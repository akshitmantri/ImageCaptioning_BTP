require 'torch'
require 'nn'
require 'nngraph'
require 'loadcaffe'
local utils = require 'misc.utils'

require 'misc.DataLoader'

require 'misc.languagemodel_attention'

local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'
image = require 'image'

-- Data input settings
cmd = torch.CmdLine()
cmd:option('-input_h5','dataset/data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','dataset/data.json','path to the json file containing additional info and vocab')
cmd:option('-cnn_proto','cnnmodel/model.prototxt','path to CNN prototxt file in Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-cnn_model','cnnmodel/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')

-- Model settings
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-embedding_size',512,'the encoding size of each token in the vocabulary, and the image.')

-- Optimization: General
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',15,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_lm', 0.5, 'strength of dropout in the Language Model RNN')
cmd:option('-finetune_cnn_after', 30000, 'After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
cmd:option('-seq_per_img',5,'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
-- Optimization: for the Language Model
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',4e-4,'learning rate')
cmd:option('-learning_rate_decay_start', 50000, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 20000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
-- Optimization: for the CNN
cmd:option('-cnn_optim','adam','optimization to use for CNN')
cmd:option('-cnn_optim_alpha',0.8,'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta',0.999,'alpha for momentum of CNN')
cmd:option('-cnn_learning_rate',1e-5,'learning rate for the CNN')
cmd:option('-cnn_weight_decay', 0, 'L2 weight decay just for the CNN')

-- Evaluation/Checkpointing
cmd:option('-val_images_use', 100, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 200, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'checkpoints', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-language_eval', 0, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-losses_log_every', 500, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

cmd:text()

-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}

local protos = {}

if string.len(opt.start_from) > 0 then
  -- load protos from file
  print('initializing weights from ' .. opt.start_from)
  local loaded_checkpoint = torch.load(opt.start_from)
  protos = loaded_checkpoint.protos
  net_utils.unsanitize_gradients(protos.cnn)
  local lm_modules = protos.lm:getModulesList()
  for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end
  net_utils.unsanitize_gradients(protos.init_h)
  net_utils.unsanitize_gradients(protos.init_c)
  net_utils.unsanitize_gradients(protos.context_encode)
  protos.crit = nn.LanguageModelCriterion() -- not in checkpoints, create manually
  protos.expander = nn.FeatExpander(opt.seq_per_img) -- not in checkpoints, create manually
  --batch wise mean model
  model_meanfeat = nn.Sequential()
  model_meanfeat:add(nn.View(-1,512,196))
  model_meanfeat:add(nn.Transpose({2,3}))
  model_meanfeat:add(nn.Mean(2))
  protos.meanfeat = model_meanfeat
  
  -- pre context encoding for batchwise
  model_pre_context_encode_batch = nn.Sequential()
  model_pre_context_encode_batch:add(nn.View(-1,512,196))
  model_pre_context_encode_batch:add(nn.Transpose({2,3}))
  model_pre_context_encode_batch:add(nn.View(-1,512))
  protos.pre_context_encode_batch = model_pre_context_encode_batch
  --post context encode for batchwise
  protos.post_context_encode_batch = nn.View(-1,196,512)



else
	-- create protos from scratch
  -- intialize language model
  local lmOpt = {}
  lmOpt.vocab_size = loader:getVocabSize()
  lmOpt.embedding_size = opt.embedding_size
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.num_layers = 1
  lmOpt.dropout = opt.drop_prob_lm
  lmOpt.seq_length = loader:getSeqLength()
  lmOpt.batch_size = opt.batch_size * opt.seq_per_img
  lmOpt.gpuid = opt.gpuid
  protos.lm = nn.LanguageModel(lmOpt)
  -- initialize the ConvNet
  local cnn_backend = opt.backend
  if opt.gpuid == -1 then cnn_backend = 'nn' end -- override to nn if gpu is disabled
  local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, cnn_backend)
  protos.cnn = net_utils.build_cnn(cnn_raw, {backend = cnn_backend})
  -- initialize a special FeatExpander module that "corrects" for the batch number discrepancy 
  -- where we have multiple captions per one image in a batch. This is done for efficiency
  -- because doing a CNN forward pass is expensive. We expand out the CNN features for each sentence
  protos.expander = nn.FeatExpander(opt.seq_per_img)
  -- criterion for the language model
  protos.crit = nn.LanguageModelCriterion()
  -- model for initial states of the lstm
  -- batch wise mean
  model_meanfeat = nn.Sequential()
  model_meanfeat:add(nn.View(-1,512,196))
  model_meanfeat:add(nn.Transpose({2,3}))
  model_meanfeat:add(nn.Mean(2))
  protos.meanfeat = model_meanfeat
 
  --h0
  protos.init_h = nn.Linear(512,opt.rnn_size)
  --c0
  protos.init_c = nn.Linear(512,opt.rnn_size)
  --prepro for batchwise context encode
  model_pre_context_encode_batch = nn.Sequential()
  model_pre_context_encode_batch:add(nn.View(-1,512,196))
  model_pre_context_encode_batch:add(nn.Transpose({2,3}))
  model_pre_context_encode_batch:add(nn.View(-1,512))
  protos.pre_context_encode_batch = model_pre_context_encode_batch
  --context encode
  protos.context_encode = nn.Linear(512,512):noBias()
  -- post context encode for batches
  protos.post_context_encode_batch = nn.View(-1,196,512)

end

if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v = v:cuda() end
end

-- flatten and prepare all model parameters to a single vector. 
local params, grad_params = protos.lm:getParameters()
local cnn_params, cnn_grad_params = protos.cnn:getParameters()
local init_h_params,init_h_grad_params = protos.init_h:getParameters()
local init_c_params,init_c_grad_params = protos.init_c:getParameters()
local context_encode_params,context_encode_grad_params = protos.context_encode:getParameters()
print('total number of parameters in LM: ', params:nElement())
print('total number of parameters in CNN: ', cnn_params:nElement())
print('total number of parameters in INIT_H: ',init_h_params:nElement())
print('total number of parameters in INIT_C: ',init_c_params:nElement())
print('total number of parameters in context_encode: ',context_encode_params:nElement())
assert(params:nElement() == grad_params:nElement())
assert(cnn_params:nElement() == cnn_grad_params:nElement())
assert(init_h_params:nElement() == init_h_grad_params:nElement())
assert(init_c_params:nElement() == init_c_grad_params:nElement())
assert(context_encode_params:nElement() == context_encode_grad_params:nElement())
-- construct thin module clones that share parameters with the actual
-- modules. These thin module will have no intermediates and will be used
-- for checkpointing to write significantly smaller checkpoint files
local thin_lm = protos.lm:clone()
thin_lm.core:share(protos.lm.core, 'weight', 'bias') -- TODO: we are assuming that LM has specific members! figure out clean way to get rid of, not modular.
local thin_cnn = protos.cnn:clone('weight', 'bias')
local thin_init_h = protos.init_h:clone('weight','bias')
local thin_init_c = protos.init_c:clone('weight','bias')
local thin_context_encode = protos.context_encode:clone('weight','bias')
-- sanitize all modules of gradient storage so that we dont save big checkpoints
net_utils.sanitize_gradients(thin_cnn)
net_utils.sanitize_gradients(thin_init_c)
net_utils.sanitize_gradients(thin_init_h)
net_utils.sanitize_gradients(thin_context_encode)
local lm_modules = thin_lm:getModulesList()
for k,v in pairs(lm_modules) do net_utils.sanitize_gradients(v) end

-- create clones and ensure parameter sharing. we have to do this 
-- all the way here at the end because calls such as :cuda() and
-- :getParameters() reshuffle memory around.
protos.lm:createClones()

collectgarbage() -- "yeah, sure why not"

local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

  protos.cnn:evaluate()
  protos.lm:evaluate()
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}
  local vocab = loader:getVocab()
  while true do

    -- fetch a batch of data
    local data = loader:getBatch{batch_size = opt.batch_size, split = split, seq_per_img = opt.seq_per_img}
    data.images = net_utils.prepro(data.images, false, opt.gpuid >= 0) -- preprocess in place, and don't augment
    n = n + data.images:size(1)

    -- forward the model to get loss
    local feats = protos.cnn:forward(data.images)
    local expanded_feats = protos.expander:forward(feats)
    local mean_features = protos.meanfeat:forward(expanded_feats)
    --find the average of all the features
    --for i=1,expanded_feats:size(1) do
    	--sum_features[i] = protos.sumfeat:forward(expanded_feats[i])
    	--sum_features[i]:div(expanded_feats:size(1))
    --end
    local h_0 = protos.init_h:forward(mean_features)
    local c_0 = protos.init_c:forward(mean_features)
    -- forward the language model
    local a_ = protos.pre_context_encode_batch:forward(expanded_feats)
    local e_ = protos.context_encode:forward(a_)
    local e = protos.post_context_encode_batch:forward(e_)

	  local logprobs = protos.lm:forward{e,data.labels,h_0,c_0,a_}
	  --local loss = 0
	  --for i=1,expanded_feats:size(1) do
		 	--local logprobs[i] = protos.lm:forward{expanded_feats[i], data.labels[{{},{i}}], init_h[i], init_c[i]}
		  -- forward the language model criterion
		  --loss = loss + protos.crit:forward(logprobs[i], data.labels[{{},{i}}])
		--end
		local loss = protos.crit:forward(logprobs,data.labels)
	  loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1

    -- forward the model to also get generated samples for each image
    local temp_mask_e = torch.ByteTensor(feats:size(1)*5,e:size(2),e:size(3)):fill(0)
    local temp_mask_hc = torch.ByteTensor(feats:size(1)*5,h_0:size(2)):fill(0)
    for i = 1,feats:size(1) do
      temp_mask_e[i*5]=1
      temp_mask_hc[i*5]=1
    end
    model_a_unrepeated = nn.Sequential()
    model_a_unrepeated:add(nn.View(-1,512,196))
    model_a_unrepeated:add(nn.Transpose({2,3}))
    model_a_unrepeated:add(nn.View(-1,512))

    if opt.gpuid >= 0 then
      temp_mask_e = temp_mask_e:cuda()
      temp_mask_hc = temp_mask_hc:cuda()
      model_a_unrepeated = model_a_unrepeated:cuda()
    end
    a_unrepeated = model_a_unrepeated:forward(feats)
    -- print(temp_mask:size(),e:size(),h_0:size(),c_0:size(),a_unrepeated:size())
    e_masked_model = nn.Reshape(feats:size(1),e:size(2),e:size(3))
    hc_masked_model = nn.Reshape(feats:size(1),h_0:size(2))

    if opt.gpuid >=0 then 
      e_masked_model:cuda()
      hc_masked_model:cuda()
    end

    local e_masked = e_masked_model:forward(e[temp_mask_e])
    local h_0_masked = hc_masked_model:forward(h_0[temp_mask_hc])
    local c_0_masked = hc_masked_model:forward(c_0[temp_mask_hc])
    local seq = protos.lm:sample(e_masked,h_0_masked,c_0_masked,a_unrepeated)
    --local seq = torch.zeros(feats:size(1),data.labels:size(1))
    --for i=1,feats:size(1) do
    	--local seq[i] = protos.lm:sample(feats,init_h[i*5],init_c[i*5]) --feats has only the images without replicating
    --end
    local sents = net_utils.decode_sequence(vocab, seq)
    for k=1,#sents do
      local entry = {image_id = data.infos[k].id, caption = sents[k]}
      table.insert(predictions, entry)
      if verbose then
        print(string.format('image %s: %s', entry.image_id, entry.caption))
      end
    end

    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, val_images_use)
    if verbose then
      print(string.format('evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss))
    end

    if loss_evals % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if n >= val_images_use then break end -- we've used enough images
  end

  local lang_stats
  if opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(predictions, opt.id)
  end

  return loss_sum/loss_evals, predictions, lang_stats
end

local iter = 0
local function lossFun()
  protos.cnn:training() --this function changes the parameters of the net to the new params
  protos.lm:training()
  protos.init_c:training()
  protos.init_h:training()
  protos.context_encode:training()
  grad_params:zero() --lm grad params
 	init_c_grad_params:zero()
 	init_h_grad_params:zero()
  context_encode_grad_params:zero()
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    cnn_grad_params:zero()
  end

  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data  
  local data = loader:getBatch{batch_size = opt.batch_size, split = 'train', seq_per_img = opt.seq_per_img}
  data.images = net_utils.prepro(data.images, true, opt.gpuid >= 0) -- preprocess in place, do data augmentation
  -- data.images: Nx3x224x224 
  -- data.seq: LxM where L is sequence length upper bound, and M = N*seq_per_img

  -- forward the ConvNet on images (most work happens here)
  local feats = protos.cnn:forward(data.images)
  -- we have to expand out image features, once for each sentence
  local expanded_feats = protos.expander:forward(feats)
  --find the average of all the features
  local mean_features = protos.meanfeat:forward(expanded_feats)
  local h_0 = protos.init_h:forward(mean_features)
  local c_0 = protos.init_c:forward(mean_features)
  -- forward the language model
  local a_ = protos.pre_context_encode_batch:forward(expanded_feats) --19600x512
  local e_ = protos.context_encode:forward(a_) --19600x512
  local e = protos.post_context_encode_batch:forward(e_) --100x196x512
  logprobs,alpha = protos.lm:forward{e,data.labels,h_0,c_0,a_}
  loss = protos.crit:forward(logprobs,data.labels)
  -- print (loss)

  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dlogprobs = protos.crit:backward(logprobs,data.labels)
    -- backprop language model
  de,dc_0,dh_0,da_1,ddummy = unpack(protos.lm:backward(dlogprobs,alpha)) 
  --local dexpanded_feats = torch.zeros(expanded_feats:size(1),expanded_feats:size(2),expanded_feats:size(3),expanded_feats:size(4))
  --local dinit_c = torch.zeros(expanded_feats:size(1),opt.rnn_size)
  --local dinit_h = torch.zeros(expanded_feats:size(1),opt.rnn_size)
  --for i=1,expanded_feats:size(1) do 
  --	dlogprobs[i] = protos.crit:backward(logprobs[i], data.labels[{{},{i}}])
  --	dexpanded_feats[i],dinit_c[i],dinit_h[i],ddummy = unpack(protos.lm:backward({expanded_feats[i],data.labels[{{},{i}}]},dlogprobs[i]))
  --end
  -- backprop through the initalization h,c layers for the lstm.
  de_ = protos.post_context_encode_batch:backward(e_,de)
  da_2 = protos.context_encode:backward(a_,de_)
  da_ = da_1 + da_2

  dexpanded_feats1 = protos.pre_context_encode_batch:backward(expanded_feats,da_)

  dmean1 = protos.init_h:backward(mean_features,dh_0)
  dmean2 = protos.init_c:backward(mean_features,dc_0)
  dmean_features = dmean1 + dmean2

  dexpanded_feats2 = protos.meanfeat:backward(expanded_feats,dmean_features)
  dexpanded_feats = dexpanded_feats1 + dexpanded_feats2
  -- backprop the CNN, but only if we are finetuning
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    local dfeats = protos.expander:backward(feats, dexpanded_feats)
    local dx = protos.cnn:backward(data.images, dfeats)
  end

  -- clip gradients
  -- print(string.format('claming %f%% of gradients', 100*torch.mean(torch.gt(torch.abs(grad_params), opt.grad_clip))))
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  -- apply L2 regularization
  if opt.cnn_weight_decay > 0 then
    cnn_grad_params:add(opt.cnn_weight_decay, cnn_params)
    -- note: we don't bother adding the l2 loss to the total loss, meh.
    cnn_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end
  -----------------------------------------------------------------------------

  -- and lets get out!
  local losses = { total_loss = loss }
  return losses
end

local loss0
local optim_state = {}
local cnn_optim_state = {}
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local best_score
local save = 0

evol_images = torch.Tensor(8,3,224,224)
if(opt.gpuid>=0) then evol_images = evol_images:cuda() end

for i=1,8 do
  img = image.load('images_folder/' .. i .. '.jpg',3,'byte')
  img = image.scale(img,224,224)
  imgtemp = torch.Tensor(1,img:size()[1],img:size()[2],img:size()[3])
  imgtemp[1] = img
  imgs = net_utils.prepro(imgtemp,false,opt.gpuid>=0)
  evol_images[i] = imgs[1]
end

while true do  

  -- eval loss/gradient
  local losses = lossFun()
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses.total_loss end
  -- print (iter)
  -- print (losses.total_loss)
  print(string.format('iter %d: %f', iter, losses.total_loss))

  -- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

    -- evaluate the validation performance
    local val_loss, val_predictions, lang_stats = eval_split('val', {val_images_use = opt.val_images_use})
    print('validation loss: ', val_loss)
    print(lang_stats)
    val_loss_history[iter] = val_loss
    if lang_stats then
      val_lang_stats_history[iter] = lang_stats
    end

    local checkpoint_path = path.join(opt.checkpoint_path, 'model_id' .. opt.id)

    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.val_loss_history = val_loss_history
    checkpoint.val_predictions = val_predictions -- save these too for CIDEr/METEOR/etc eval
    checkpoint.val_lang_stats_history = val_lang_stats_history

    utils.write_json(checkpoint_path .. '.json', checkpoint)
    print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    -- write the full model checkpoint as well if we did better than ever
    local current_score
    if lang_stats then
      -- use CIDEr score for deciding how well we did
      current_score = lang_stats['CIDEr']
    else
      -- use the (negative) validation loss as a score
      current_score = -val_loss
    end
    if best_score == nil or current_score > best_score then
      best_score = current_score end
    if iter > 0 then -- dont save on very first iteration
      -- include the protos (which have weights) and save to file
      save = save+1
      local save_protos = {}
      save_protos.lm = thin_lm -- these are shared clones, and point to correct param storage
      save_protos.cnn = thin_cnn
      save_protos.init_h = thin_init_h
      save_protos.init_c = thin_init_c
      save_protos.context_encode = thin_context_encode
      checkpoint.protos = save_protos
      -- also include the vocabulary mapping so that we can use the checkpoint 
      -- alone to run on arbitrary images without the data loader
      checkpoint.vocab = loader:getVocab()
      torch.save(checkpoint_path .. save .. '.t7', checkpoint)
      print('wrote checkpoint to ' .. checkpoint_path .. save .. '.t7')
    end
  end

  if(iter%1000 == 0) then
    vocab  = loader:getVocab()
    evolve_checkpoint = {}
    local feats_evol = protos.cnn:forward(evol_images)
    local mean_features_evol = protos.meanfeat:forward(feats_evol)
    local h_0_evol = protos.init_h:forward(mean_features_evol)
    local c_0_evol = protos.init_c:forward(mean_features_evol)
    local a__evol = protos.pre_context_encode_batch:forward(feats_evol)
    local e__evol = protos.context_encode:forward(a__evol)
    local e_evol = protos.post_context_encode_batch:forward(e__evol)
    local seq_evol,lp,alpha_evol = protos.lm:sample(e_evol,h_0_evol,c_0_evol,a__evol)
    local sents_evol = net_utils.decode_sequence(vocab,seq_evol)
    table.insert(evolve_checkpoint,sents_evol)
    table.insert(evolve_checkpoint,alpha_evol)
    torch.save('evolve_checkpoints/' .. iter/1000 .. '.t7',evolve_checkpoint)
    print('saved evolution checkpoint to evolve_checkpoints/' .. iter/1000 .. '.t7')
  end



  -- decay the learning rate for both LM and CNN
  local learning_rate = opt.learning_rate
  local cnn_learning_rate = opt.cnn_learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
    cnn_learning_rate = cnn_learning_rate * decay_factor
  end

  -- perform a parameter update
  sgd(init_c_params,init_c_grad_params,learning_rate)
  sgd(init_h_params,init_h_grad_params,learning_rate)
  if opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'sgd' then
    sgd(params, grad_params, opt.learning_rate)
  elseif opt.optim == 'sgdm' then
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'sgdmom' then
    sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'adam' then
    adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
  else
    error('bad option opt.optim')
  end

  -- do a cnn update (if finetuning, and if rnn above us is not warming up right now)
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    if opt.cnn_optim == 'sgd' then
      sgd(cnn_params, cnn_grad_params, cnn_learning_rate)
    elseif opt.cnn_optim == 'sgdm' then
      sgdm(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn_optim_state)
    elseif opt.cnn_optim == 'adam' then
      adam(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn_optim_state)
    else
      error('bad option for opt.cnn_optim')
    end
  end

  -- stopping criterions
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 20 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

end
