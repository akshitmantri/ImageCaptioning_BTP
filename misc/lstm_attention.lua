require 'nn'
require 'nngraph'

local LSTM = {}
function LSTM.lstm(embedding_size,zt_size,num_features,dim_features,vocab_size,rnn_size,dropout,batch_size)
  dropout = dropout or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- e (contexted features) 100x196x512
  table.insert(inputs, nn.Identity()()) -- Eyt_1 100x512
  table.insert(inputs, nn.Identity()())  -- original features a 100x196x14x14
	table.insert(inputs, nn.Identity()()) -- prev_c 100x512
  table.insert(inputs, nn.Identity()()) -- prev_h 100x512
	

  local  Eyt_1, a
  local outputs = {}
    -- c,h from previos timesteps
  local prev_h = inputs[5]
  local prev_c = inputs[4]
  -- the input to this layer
  e = inputs[1] --converted to 100x196x512 after contexting
  Eyt_1 = inputs[2] 
  a = inputs[3]
  a_ = nn.View(-1,num_features,dim_features)(a) --100x196x512
  --context of h
  h_context = nn.Replicate(num_features,2,zt_size)(nn.Linear(rnn_size,zt_size)(prev_h))
  -- creating the eti given in the paper
  z = nn.Tanh()(nn.CAddTable()({e,h_context}))--100x196x512
  --flattening z
  z_flat = nn.View(-1,zt_size)(z) --19600x512
  --alpha
  alpha = nn.View(-1,num_features)(nn.Linear(zt_size,1)(z_flat)) --19600x1-> 100x196
  alpha = nn.SoftMax()(alpha) --100x196
  --final z_t
  weighted_context = nn.Sum(2)(nn.CMulTable()({a_,nn.Replicate(dim_features,3,num_features)(alpha)})) --100x196x512->100x512

  -- evaluate the input sums at once for efficiency 	
  local Eyt_12h = nn.Linear(embedding_size, 4 * rnn_size)(Eyt_1):annotate{name='Eyt_12h_'} --100,512x4
  local zt2h = nn.Linear(zt_size, 4*rnn_size)(weighted_context):annotate{name='zt2h_'} --100,512x5
  local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'} --100,512x4
  local all_input_sums = nn.CAddTable()({Eyt_12h, zt2h, h2h}) --100,512x4

  local reshaped = nn.Reshape(4, rnn_size)(all_input_sums) --100x4x512
  local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4) --100x512
  -- decode the gates
  local in_gate = nn.Sigmoid()(n1)
  local forget_gate = nn.Sigmoid()(n2)
  local out_gate = nn.Sigmoid()(n3)
  -- decode the write inputs
  local in_transform = nn.Tanh()(n4)
  -- perform the LSTM update
  local next_c = nn.CAddTable()
  									(
  										{nn.CMulTable()
  											({forget_gate, prev_c}),
  										nn.CMulTable()
  											({in_gate, in_transform})
  										}
  									)
  -- gated cells form the output
  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  
  table.insert(outputs, next_c) --100x512
  table.insert(outputs, next_h) --100x512
  table.insert(outputs,alpha) --100x196

  -- set up the decoder
  --local top_h = outputs[#outputs]
  logits = nn.ReLU()(nn.Linear(rnn_size,rnn_size)(next_h)) --100x512
  logit_words = nn.Linear(rnn_size,vocab_size)(logits) --100xvsize

  -- local proj = nn.Linear(embedding_size, vocab_size)
  -- 										(nn.CAddTable()
  -- 											({Eyt_1,
  -- 													nn.Linear(rnn_size,embedding_size)(next_h),
  -- 														nn.Linear(zt_size,embedding_size)(zt)
  -- 											})
  -- 										):annotate{name='decoder'}

  local y_t = nn.LogSoftMax()(logit_words) --100xvsize
  table.insert(outputs, y_t)

  return nn.gModule(inputs, outputs)
end

return LSTM