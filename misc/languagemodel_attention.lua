require 'nn'
local LSTM = require 'misc.lstm_attention'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local layer, parent = torch.class('nn.LanguageModel', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.vocab_size = utils.getopt(opt,'vocab_size',9567)
  self.embedding_size = utils.getopt(opt,'embedding_size',512)
  self.zt_size = 512
  self.rnn_size = utils.getopt(opt,'rnn_size',512)
  self.num_features = 196
  self.dim_features = 512
  --self.num_layers = 1
  local dropout = utils.getopt(opt,'dropout',0)
  -- options for Language Model
  self.batch_size = utils.getopt(opt,'batch_size',100)
  self.seq_length = utils.getopt(opt,'seq_length',16)
  self.gpuid = utils.getopt(opt,'gpuid',0)
  -- create the core lstm network. note +1 for both the START and END tokens
  self.core = LSTM.lstm(self.embedding_size,self.zt_size,self.num_features,self.dim_features,self.vocab_size + 1,self.rnn_size,dropout,self.batch_size)
  self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.embedding_size)

  --self:_createInitState(1) -- will be lazily resized later during forward passes
end

function layer:_createInitState(init_h,init_c)
	self.init_state = {} 
  self.init_state[1] = init_c
  self.init_state[2] = init_h
end

function layer:createClones()
  -- construct the net clones
  print('constructing clones inside the LanguageModel')
  self.clones = {self.core}
  self.lookup_tables = {self.lookup_table} 
  for t=2,self.seq_length+2 do
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.lookup_tables[t] = self.lookup_table:clone('weight','gradWeight')
  end
end

function layer:getModulesList()
  return {self.core,self.lookup_table}
end

function layer:parameters()

  local p1,g1 = self.core:parameters()
  local p2,g2 = self.lookup_table:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end  
  for k,v in pairs(p2) do table.insert(params, v) end
  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end

  return params, grad_params
end

function layer:training()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:training() end
  for k,v in pairs(self.lookup_tables) do v:training() end
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:evaluate() end
  for k,v in pairs(self.lookup_tables) do v:evaluate() end
end


function layer:sample(img,init_h,init_c,a)
	self:_createInitState(init_h,init_c)
  local state = self.init_state
  batch_size = img:size(1)
  -- we will write output predictions into tensor seq
  local seq = torch.LongTensor(self.seq_length,batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length,batch_size)
  if self.gpuid>=0 then
    seq = seq:cuda()
    seqLogprobs = seqLogprobs:cuda()
  end
  local logprobs -- probs predicted in last time step
  local alpha = {}
  for t=1,self.seq_length+1 do

    local it, sampleLogprobs,inputs
    if t == 1 then
      --start tokem
      it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      if self.gpuid >=0 then it=it:cuda() end
      xt = self.lookup_table:forward(it)
    else
      sampleLogprobs, it = torch.max(logprobs,2)
      it = it:view(-1):long()
      xt = self.lookup_table:forward(it)
    end

    if t >= 2 then 
      seq[t-1] = it -- record the samples
      seqLogprobs[t-1] = sampleLogprobs:view(-1):float() -- and also their log likelihoods
    end

    inputs = {img,xt,a,unpack(state)}
	  local out = self.core:forward(inputs)
    logprobs = out[4] -- last element is the output vector
    alpha[t] = out[3]:clone()
    -- if(t==3) then
    --   print(alpha[1],alpha[2],alpha[3])
    -- end

    state = {}
    for i=1,2 do table.insert(state, out[i]) end
  end
  -- return the samples and their log likelihoods
  return seq,seqLogprobs,alpha
end


--[[
input is a tuple of:
1. torch.Tensor of size 512*14*14 (K is dim of image code)
2. torch.LongTensor of size 16, --]]

function layer:updateOutput(input)
  local imgs = input[1]
  local seq = input[2]
  local init_h = input[3]
  local init_c = input[4]
  local a = input[5]
  local alpha = {}
  if self.clones == nil then self:createClones() end -- lazily create clones on first forward pass
  assert(seq:size(1) == self.seq_length)
  local batch_size = seq:size(2)
  self.output:resize(self.seq_length+1,batch_size,self.vocab_size+1)
  self:_createInitState(init_h,init_c)
  self.state = {[0] = self.init_state}
  self.inputs = {}
  self.lookup_tables_inputs = {}
  self.tmax = 0 -- we will keep track of max sequence length encountered in the data for efficiency
  for t=1,self.seq_length+1 do

    local can_skip = false
    if t == 1 then
      local it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      if self.gpuid>=0 then it=it:cuda() end
      self.lookup_tables_inputs[t] = it
      xt = self.lookup_tables[t]:forward(it)
    else
      -- feed in the rest of the sequence...
      local it = seq[t-1]:clone()
      if torch.sum(it) == 0 then
        -- computational shortcut for efficiency. All sequences have already terminated and only
        -- contain null tokens from here on. We can skip the rest of the forward pass and save time
        can_skip = true 
      end
      it[torch.eq(it,0)] = 1

      if not can_skip then
        self.lookup_tables_inputs[t] = it
        xt = self.lookup_tables[t]:forward(it)
      end
    end

    if not can_skip then
      -- construct the inputs
    	self.inputs[t] = {imgs,xt,a,unpack(self.state[t-1])}
      local out = self.clones[t]:forward(self.inputs[t])
      -- process the outputs
      self.output[t] = out[4] -- last element is the output vector
      alpha[t] = out[3]:clone()
      self.state[t] = {} -- the rest is state
      for i=1,2 do table.insert(self.state[t], out[i]) end
      self.tmax = t
    end
  end
  return self.output,alpha
end


--[[
gradOutput is an (D+1)xNx(M+1) Tensor.
--]]
function layer:updateGradInput(gradOutput,alpha)
  -- print (gradOutput:size())
  -- print (alpha[1]:size())
  local dimgs = torch.zeros(gradOutput:size(1),gradOutput:size(2),self.num_features,self.dim_features)-- grad on input images
  local da = torch.zeros(gradOutput:size(1),gradOutput:size(2)*self.num_features,self.dim_features)
  -- go backwards and lets compute gradients
  local init_state = {}
  if self.gpuid>=0 then
    dimgs = dimgs:cuda()
    da = da:cuda()
    table.insert(init_state,torch.CudaTensor(self.batch_size,self.rnn_size):zero())
    table.insert(init_state,torch.CudaTensor(self.batch_size,self.rnn_size):zero())
  else
    table.insert(init_state,torch.zeros(self.batch_size,self.rnn_size))
    table.insert(init_state,torch.zeros(self.batch_size,self.rnn_size))
  end

  local dstate = {[self.tmax] = init_state} -- this works when init_state is all zeros
  for t=self.tmax,1,-1 do
    -- concat state gradients and output vector gradients at time step t
    local dout = {}
    for k=1,#dstate[t] do table.insert(dout, dstate[t][k]) end
    table.insert(dout,alpha[t])
    table.insert(dout, gradOutput[t])
    --print(t,#gradOutput[t],#dout,#self.inputs[t])
    -- for k,v in pairs(self.inputs[t]) do v:float() end
    -- for k,v in pairs(dout) do v:float() end
    -- self.clones[t]:cuda()
    local dinputs = self.clones[t]:backward(self.inputs[t], dout)
    -- split the gradient to xt and to state
    dimgs[t] = dinputs[1] -- first element is the input vector
    dx_t = dinputs[2]
    da[t] = dinputs[3]
    dinit_c = dinputs[4]
    dinit_h = dinputs[5]
     

    local it = self.lookup_tables_inputs[t]
    self.lookup_tables[t]:backward(it,dx_t)

    dstate[t-1] = {} -- copy over rest to state grad
    for k=4,5 do table.insert(dstate[t-1], dinputs[k]) end
  end
  -- we have gradient on image, but for LongTensor gt sequence we only create an empty tensor - can't backprop
  da = torch.sum(da,1)[1]
  dimgs = torch.sum(dimgs,1)[1]
  self.gradInput = {dimgs,dinit_c,dinit_h,da,torch.Tensor()}
  --print(#dinit_c,#dinit_h,#dimgs)
  return self.gradInput
end

-------------------------------------------------------------------------------
-- Language Model-aware Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.LanguageModelCriterion', 'nn.Criterion')
function crit:__init()
  parent.__init(self)
end

--[[
input is a Tensor of size (D+1)x(M+1)
seq is a LongTensor of size D. The way we infer the target
in this criterion is as follows:
- at first time step the output is ignored (loss = 0). It's the image tick
- the label sequence "seq" is shifted by one to produce targets
- at last time step the output is always the special END token (last dimension)
The criterion must be able to accomodate variably-sized sequences by making sure
the gradients are properly set to zeros where appropriate.
--]]
function crit:updateOutput(input, seq)
  self.gradInput:resizeAs(input):zero() -- reset to zeros
  local L,N,Mp1 = input:size(1), input:size(2) , input:size(3)
  local D = seq:size(1)
  assert(D == L-1, 'input Tensor should be 1 larger in time')

  local loss = 0
  local n = 0
  for b=1,N do
    local first_time = true
    for t=1,L do -- iterate over sequence time (ignore t=1, dummy forward for the image)

      -- fetch the index of the next token in the sequence
      local target_index
      if t > D then -- we are out of bounds of the index sequence: pad with null tokens
        target_index = 0
      else
        target_index = seq[{t,b}] -- t is correct, since at t=1 START token was fed in and we want to predict first word 
      end
      -- the first time we see null token as next index, actually want the model to predict the END token
      if target_index == 0 and first_time then
        target_index = Mp1
        first_time = false
      end

      -- if there is a non-null next token, enforce loss!
      if target_index ~= 0 then
        -- accumulate loss
        loss = loss - input[{t,b,target_index}] -- log(p)
        self.gradInput[{ t,b,target_index }] = -1
        n = n + 1
      end
    end
  end
  self.output = loss / n -- normalize by number of predictions that were made
  self.gradInput:div(n)
  return self.output
end

function crit:updateGradInput(input, seq)
  return self.gradInput
end


