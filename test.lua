require 'torch'
require 'nn'
require 'nngraph'
-- exotics
require 'loadcaffe'
-- local imports
local utils = require 'misc.utils'
require 'misc.languagemodel_attention'
local net_utils = require 'misc.net_utils'
image = require 'image'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:text()

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

model = torch.load('checkpoints/model_id1.t7')
protos = model.protos
vocab = model.vocab
model_meanfeat = nn.Sequential()
model_meanfeat:add(nn.View(-1,512,196))
model_meanfeat:add(nn.Transpose({2,3}))
model_meanfeat:add(nn.Mean(2))
if opt.gpuid>=0 then model_meanfeat:cuda() end
protos.meanfeat = model_meanfeat

-- pre context encoding for batchwise
model_pre_context_encode_batch = nn.Sequential()
model_pre_context_encode_batch:add(nn.View(-1,512,196))
model_pre_context_encode_batch:add(nn.Transpose({2,3}))
model_pre_context_encode_batch:add(nn.View(-1,512))
if opt.gpuid>=0 then model_pre_context_encode_batch:cuda() end 
protos.pre_context_encode_batch = model_pre_context_encode_batch
--post context encode for batchwise
protos.post_context_encode_batch = nn.View(-1,196,512)
if opt.gpuid >=0 then protos.post_context_encode_batch:cuda() end

seq = torch.Tensor(16,1)
if(opt.gpuid>=0) then
	seq = seq:cuda()
end

-- evol_images = torch.Tensor(8,3,224,224)
-- if(opt.gpuid>=0) then evol_images = evol_images:cuda() end

-- for i=1,8 do
--   img = image.load('images_folder/' .. i .. '.jpg',3,'byte')
--   imgtemp = torch.Tensor(1,img:size()[1],img:size()[2],img:size()[3])
--   imgtemp[1] = img
--   imgs = net_utils.prepro(imgtemp,false,opt.gpuid>=0)
--   evol_images[i] = imgs[1]
-- end

-- print(#evol_images)
i = image.load('images_folder/1.jpg',3,'byte')
img = torch.Tensor(1,i:size()[1],i:size()[2],i:size()[3])
img[1] = i
img = net_utils.prepro(img,false,opt.gpuid>=0)
-- print(img)
local feats = protos.cnn:forward(img)
local mean_features = protos.meanfeat:forward(feats)
local h_0 = protos.init_h:forward(mean_features)
local c_0 = protos.init_c:forward(mean_features)
local a_ = protos.pre_context_encode_batch:forward(feats)
local e_ = protos.context_encode:forward(a_)
local e = protos.post_context_encode_batch:forward(e_)
local seq,lm,alpha = protos.lm:sample(e,h_0,c_0,a_)
local sents = net_utils.decode_sequence(vocab,seq)
print(sents)
print(alpha[1])





