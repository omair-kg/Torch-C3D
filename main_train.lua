require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'paths'
require  'optim'
require 'xlua'
require 'datasources.ucf101_listloader'
require 'image'
require 'math'
local Threads = require 'threads'
torch.setdefaulttensortype('torch.FloatTensor')

-- Parameters for datasource class, batch size and LR
opt_default = {
  -- general
  saveName_model = 'model.t7',  -- save file name
  saveName_optim = 'optimState.t7',
  save_dir = "/mnt/data1/c3d_experiment_trainSet_hope_lr5e4",
  -- training
  nEpoches = 20,         -- number of "epoches" per training
  nIters = 3422,             -- number of minibatches per training epoch
  nIters_test = 1335,        -- number of minibatches per test epoch
  batchSize = 30,            -- number of samples per minibatches
  -- model
  modelOptim = 'adam',        -- delta(adadelta), grad(adagrad) or sgd
  modelConfig = {
    learningRate = 5e-4,
    learningRateDecay = 1e-4,
    weightDecay = 5e-5
  },
  params={                  -- parameters for dataloader class
    nInputFrames=16
  },
  nThreads = 16,
  from_checkPoint = false,
  save_every = 1
}

local params_ = opt_default.params
local batchSize = opt_default.batchSize
--=======================Network declaration============================
local net,optimState,input
print("\n")
-- Define network arch, type conversions and size allocations
if opt_default.from_checkPoint then
  print("\n Loading from checkpoint \n")
  net = torch.load(paths.concat(opt_default.save_dir, 'latest',opt_default.saveName_model))
  optimState = torch.load(paths.concat(opt_default.save_dir, 'latest',opt_default.saveName_optim)) 
else  
  optimState = opt_default.modelConfig
  net, isize = require 'c3d'(batchSize)
  local sizeTensor = torch.LongTensor(isize)
  input = torch.randn(sizeTensor:storage())
--net:apply(weights_init)
end
local criterion = nn.CrossEntropyCriterion()--nn.ClassNLLCriterion()--
net = net:cuda()
net = cudnn.convert(net, cudnn)
criterion = criterion:cuda()
local params, gradParams = net:getParameters()
--========================== Multi threading initialization=============
local nThreads = opt_default.nThreads

donkey = Threads(
  nThreads,
  function()
    require 'torch'
    require 'datasources.ucf101_listloader'
    require 'image'
  end,
  function(idx)
    params_={
      nInputFrames=16}
    batchSize = 30
    tid = idx
    local seed = 2 + idx
    torch.manualSeed(seed)
    torch.setnumthreads(1)
    datasource = UCF101_jpgDatasource(params_)
  end
);
--========================Declare loggers=============================
save_dir = opt_default.save_dir
local trainLogger = optim.Logger(save_dir .. '_train.log')
trainLogger:setNames{'Loss' , 'Acc.'}

local valLogger = optim.Logger(save_dir .. '_val.log')
valLogger:setNames{'Loss' , 'Acc.'}
--==========================Init schemes==============================
local function weights_init(m)
  local name = torch.type(m)
  if name:find('VolumetricConvolution') then
    m.weight:normal(0.0, 0.01)
    m.bias:fill(1)
  elseif name:find('Linear') then
    if m.weight then m.weight:normal(0.0, 0.005) end
    if m.bias then m.bias:fill(1) end
  end
end

-- declare variables to be used in script
local top1_epoch, top1_epoch_val
local loss_epoch,loss_epoch_val
local batchNumber 
local epoch=0
local confusion = optim.ConfusionMatrix(101)
local confusion_epoch = optim.ConfusionMatrix(101)

local dataTimer = torch.Timer()


-- train one batch
function trainBatch(batch,label)
  cutorch.synchronize()
  collectgarbage()
  local dataLoadingTime = dataTimer:time().real

  batchNumber = batchNumber + 1
  local input = batch--:clone()
  input = input:permute(1,3,2,4,5):cuda()
  local target=label--:clone()
  target=target:cuda()
  --===================== feval function for optimizer
  local loss,output
  feval = function (x)
    --gradParams:zero()
    net:zeroGradParameters()
    output = net:forward(input)
    loss = criterion:forward(output, target)
    local dloss_doutput = criterion:backward(output, target)
    net:backward(input, dloss_doutput)
    return loss, gradParams
  end

  optim.adam(feval,params,optimState)
  cutorch.synchronize()
  loss_epoch = loss_epoch+loss
  --top1 error
  local top1 = 0
  --=====================Accuracy calculation===================================
  do
    for i=1,batchSize do
      confusion:add(output[i], target[i])
      confusion_epoch:add(output[i],target[i])
    end
  end
  confusion:updateValids()
  confusion_epoch:updateValids()
  top1 = confusion.totalValid * 100
--=======================Batch statistics display==================================
  print((sys.COLORS.red .. 'Epoch: [%d][%d/%d]\t Err %.4f \t Top1-%%: %.2f \t Top1_Global-%%: %.2f \t Loading Time %.3f'):format(
      epoch, batchNumber, opt_default.nIters,loss, top1,confusion_epoch.totalValid * 100,
      dataLoadingTime))
  confusion:zero()
  dataTimer:reset() 
end

function testBatch(batch,label)
  cutorch.synchronize()
  collectgarbage()
  local loss_,output_
  input = batch--:clone()
  input = input:permute(1,3,2,4,5):cuda()
  target=label--:clone()
  target=target:cuda()
  output_ = net:forward(input)
  loss_ = criterion:forward(output_, target)
  loss_epoch_val = loss_epoch_val + loss_
  --=================================================================================
  do
    for i=1,batchSize do
      confusion_epoch:add(output_[i],target[i])
    end
  end
  confusion_epoch:updateValids()
end


--- Main training and validation loops from here
local oneEpoch = 102681-- number of instances in training epoch
local oneEpoch_val = 40072-- number of instances in validation epoch
local shuffle_list_train = torch.randperm(oneEpoch)--torch.range(1,oneEpoch)--
local shuffle_list_test = torch.range(1,oneEpoch_val)--torch.randperm(oneEpoch_val)

for epoch_ = 1,opt_default.nEpoches do --number of epochs to run for
  top1_epoch = 0
  top1_epoch_val=0
  confusion_epoch:zero()

  loss_epoch = 0
  loss_epoch_val=0
  time = sys.clock()

  epoch = epoch + 1
  batchNumber = 0
  batchSize = opt_default.batchSize
  net:training()

  for t = 1,oneEpoch,batchSize do
    collectgarbage()
    cutorch.synchronize()
    if (t+batchSize)< oneEpoch then
      donkey:addjob(
        function()
          if (t+batchSize)> oneEpoch then
            batchSize = oneEpoch - t+1
          end
          local batch, label = datasource:nextBatch(shuffle_list_train[{{t,t+batchSize-1}}], 'train')
          return batch,label
        end,
        trainBatch
      )
    end
  end
  donkey:synchronize()
  cutorch.synchronize()
  time = sys.clock() - time
  top1_epoch = confusion_epoch.totalValid*100--100 * top1_epoch/oneEpoch
  loss_epoch=loss_epoch/opt_default.nIters --oneEpoch/batchSize
  trainLogger:add{loss_epoch,top1_epoch}
  print((sys.COLORS.blue .."\n Training Summary:\n Epoch:[%d] Average loss: %.5f Acc.: %.2f ,Total time: %.5f"):format( epoch,loss_epoch and loss_epoch or -1,top1_epoch,time))

  time = sys.clock()
  batchSize=opt_default.batchSize
  print(sys.COLORS.green .. "\n Validation ")
  confusion_epoch:zero()

  for t = 1,oneEpoch_val,batchSize do
    collectgarbage()
    cutorch.synchronize()
    if (t+batchSize)< oneEpoch_val then
      donkey:addjob(
        function()
          if (t+batchSize)> oneEpoch_val then
            batchSize = oneEpoch_val - t+1
          end
          local batch, label = datasource:nextBatch(shuffle_list_test[{{t,t+batchSize-1}}], 'test')
          return batch,label
        end,
        testBatch
      )  
    end
    xlua.progress(t,oneEpoch_val)
  end
  donkey:synchronize()
  cutorch.synchronize()
  time = sys.clock() - time
  top1_epoch_val = confusion_epoch.totalValid*100--100 * top1_epoch_val/oneEpoch_val
  loss_epoch_val=loss_epoch_val/opt_default.nIters_test --oneEpoch_val/batchSize
  valLogger:add{loss_epoch_val,top1_epoch_val}
  print((sys.COLORS.green .. "\n Validation Summary \n Epoch:[%d] loss: %.5f Acc.: %.2f , time:%.5f"):format( epoch,loss_epoch_val and loss_epoch_val or -1,top1_epoch_val,time))


  if epoch % opt_default.save_every == 0 then
    local filename = paths.concat(save_dir, tostring(epoch),opt_default.saveName_model)
    local optimFileName = paths.concat(save_dir, tostring(epoch),opt_default.saveName_optim)
    os.execute('mkdir -p ' .. sys.dirname(filename))
    net:clearState()
    cudnn.convert(net, nn)
    torch.save(filename, net)
    cudnn.convert(net, cudnn)
    torch.save(optimFileName, optimState)
  end

end

