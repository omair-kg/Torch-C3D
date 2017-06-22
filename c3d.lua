local nninit = require 'nninit'
local function c3d(batchSize)
  ---[[
   -- Create table describing C3D configuration
   local cfg = {64 , 'M1' , 128 , 'M' , 256 , 'M' , 256 , 'M' , 256 ,'M'}
   local features = nn.Sequential()
   do
      local iChannels = 3;
      for k,v in ipairs(cfg) do
         if v == 'M' then
            features:add(nn.VolumetricMaxPooling(2,2,2,2,2,2):ceil())
         elseif v == 'M1' then
            features:add(nn.VolumetricMaxPooling(1,2,2,1,2,2):ceil())
         else
            local oChannels = v;
            features:add(nn.VolumetricConvolution(iChannels,oChannels,3,3,3,1,1,1,1,1,1):init('weight',nninit.kaiming))
            features:add(nn.ReLU(true))
            iChannels = oChannels;
         end
      end
   end

   --features:get(1).gradInput = nil

   local classifier = nn.Sequential()
   classifier:add(nn.View(256*1*4*4))
   classifier:add(nn.Linear(256*1*4*4, 2048):init('weight',nninit.kaiming))
   classifier:add(nn.ReLU(true))
   classifier:add(nn.Dropout(0.7))
   classifier:add(nn.Linear(2048, 2048):init('weight',nninit.kaiming))
   classifier:add(nn.ReLU(true))
   classifier:add(nn.Dropout(0.7))
   classifier:add(nn.Linear(2048, 101):init('weight',nninit.kaiming)) -- UCF-101
   --classifier:add(nn.LogSoftMax())

   local model = nn.Sequential()
   model:add(features):add(classifier)

   return model, {batchSize,3,16,112,112}
end
--]]
--[[
  local cfg = {64, 'M1', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}

  local features = nn.Sequential()
  do
    local iChannels = 3;
    for k,v in ipairs(cfg) do
      if v == 'M' then
        features:add(nn.VolumetricMaxPooling(2,2,2,2,2,2):ceil())
      elseif v == 'M1' then
        features:add(nn.VolumetricMaxPooling(1,2,2,1,2,2):ceil())
      else
        local oChannels = v;
        features:add(nn.VolumetricConvolution(iChannels,oChannels,3,3,3,1,1,1,1,1,1))
        features:add(nn.ReLU(true))
        iChannels = oChannels;
      end
    end
  end

  --features:get(1).gradInput = nil

  local classifier = nn.Sequential()
  classifier:add(nn.View(512*1*4*4))
  classifier:add(nn.Linear(512*1*4*4, 4096))
  classifier:add(nn.ReLU(true))
  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(4096, 4096))
  classifier:add(nn.ReLU(true))
  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(4096, 101):init('weight',nninit.kaiming)) -- UCF-101
  classifier:add(nn.LogSoftMax())
  local model = nn.Sequential()
  model:add(features):add(classifier)
  net = torch.load("/mnt/data1/Nets/DownloadedNets/conv3d_deepnetA_sport1m_iter_1900000.t7")--("/mnt/data1/c3d_experiment_overfit_control/c3d.t7")

  convs ={1,4,7,9,12,14,17,19}
  for i,v in ipairs(convs) do

    model.modules[1].modules[v].weight = net.modules[v].weight:clone()
    model.modules[1].modules[v].bias = net.modules[v].bias:clone()

  end

  fc = {2,5}
  for i,v in ipairs(fc) do

    model.modules[2].modules[v].weight = net.modules[v+21].weight
    model.modules[2].modules[v].bias = net.modules[v+21].bias

  end

  return model, {batchSize,3,16,112,112}
end
--]]
return c3d
