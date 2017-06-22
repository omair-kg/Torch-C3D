require 'torch'
require 'io'
require 'datasources.datasource'
require 'paths'
require 'image'
require 'math'
--[[
params.nInputFrames: number of frames in one chunk
params.datapath: Change this for other datasets, default is ucf101
--]]
local UCF101_jpgDatasource, parent = torch.class('UCF101_jpgDatasource', 'ClassDatasource')

function UCF101_jpgDatasource:__init(params)
  parent.__init(self)
  assert(params.nInputFrames ~= nil, "UCF101Dataset: must specify nInputFrames")
  self.datapath = params.datapath or '/mnt/data0/poseEstimation_torch/h5_files/UCF/UCF-101_jpgs'
  local setfiles = {train = 'train_02.txt', test = 'test_01.txt'}
  assert(paths.dirp(self.datapath), 'Path ' .. self.datapath .. ' does not exist')
  self.listPath = "/mnt/data0/poseEstimation_torch/h5_files/UCF/ucfTrainTestlist/"
  assert(paths.dirp(self.listPath), 'Path ' .. self.listPath .. ' does not exist')

  self.label = {train = {}, test = {}}
  self.fileList = {train = {}, test = {}}
  self.frameNumber = {train = {}, test = {}}

  for _, set in pairs{'train', 'test'} do
    local f = io.open(paths.concat(self.listPath, setfiles[set]), 'r')
    assert(f ~= nil, 'File ' .. paths.concat(self.listPath, setfiles[set]) .. ' not found.')
    local N = 0
    for line in f:lines() do
      if string.byte(line:sub(-1,-1)) == 13 then
        --remove the windows carriage return
        line = line:sub(1,-2)
      end
      local line_path = line:sub(1,line:find(' ')-2)
      local line_im = line_path:sub(line:find('/')+1,-1)
      local filepath = (paths.concat(self.datapath,line_path,line_im))
      line_ = line:sub(line:find(' ')+1,-1)
      local framenumber = tonumber(line_:sub(1,line_:find(' ')))
      local class = tonumber(line_:sub(line_:find(' ')+1,-1))
      table.insert(self.fileList[set],filepath)
      table.insert(self.frameNumber[set],framenumber)
      table.insert(self.label[set],class)
      N = N+1
    end
    f:close()
    if set == 'train' then
      assert(N==102681,'Not all Training files read' .. N)
    else
      assert(N==40072,'Not all Test files read' .. N)
    end
  end
  self.nInputFrames = params.nInputFrames
  self.nChannels, self.nClasses = 3, 101
  self.h, self.w = 112, 112
  self.mean ={0.39676561420733 , 0.38167717973734 , 0.35316953393671}
  --self.mean= {101.16265612, 97.28164335, 89.96576999}
  --self.std =  {71.89510996, 70.41822383, 71.29522751}
end

function UCF101_jpgDatasource:randomCrop(im,y,x)
  local weird_h ,weird_w = 128 , 171      
  im = image.scale(im, weird_w,weird_h)

  local h, w = im:size(2), im:size(3)
  if h > self.h then
    --local y = torch.random(h - self.h + 1)
    im = im:narrow(2, y, self.h)
  end
  if w > self.w then
    --local x = torch.random(w - self.w + 1)
    im = im:narrow(3, x, self.w)
  end
  if im:size(1) == 1 then
    im = im:expand(3, im:size(2), im:size(3))
  end
  return im
end

function UCF101_jpgDatasource:centerCrop(im)
  local weird_h ,weird_w = 128 , 171      
  im = image.scale(im, weird_w,weird_h) 
  local h, w = im:size(2), im:size(3)
  if h > self.h then
    local y = math.floor((h - self.h)/2+1)
    im = im:narrow(2, y, self.h)
  end
  if w > self.w then
    local x = math.floor((w - self.w)/2+1)
    im = im:narrow(3, x, self.w)
  end
  if im:size(1) == 1 then
    im = im:expand(3, im:size(2), im:size(3))
  end
  return im
end

function UCF101_jpgDatasource:nextBatch(batch,set)
  batchSize = batch:size(1)
  assert(batchSize ~= nil, 'nextBatch: must specify batchSize')
  self.output_cpu:resize(batchSize, self.nInputFrames, self.nChannels, self.h, self.w)
  self.labels_cpu:resize(batchSize)
  for i = 1, batchSize do

    index = batch[i]
    start_frame = self.frameNumber[set][index]
    filepath = self.fileList[set][index]
    self.labels_cpu[i] = self.label[set][index]
    
    local should_I_flip = torch.uniform()<0.5
    local y = torch.random(128 - self.h + 1)
    local x = torch.random(171 - self.w + 1)
    for j = 1, self.nInputFrames do
      impath = (filepath .. '_%06d.jpg'):format(j+start_frame-1)
      im = image.load(impath)
      for threeChannels=1,3 do -- channels
        im[{{threeChannels},{},{}}]:add(-self.mean[threeChannels])
        im[{{threeChannels},{},{}}]:div(self.std[threeChannels]) 
      end
      -- assign the image to the returned output variable
      if string.match(set, "train") then
        if should_I_flip then
          self.output_cpu[i][j]:copy(self:randomCrop(im,y,x)) 
        else -- randomly flip image
          self.output_cpu[i][j]:copy(image.hflip(self:randomCrop(im,y,x))) 
        end

      else
        self.output_cpu[i][j]:copy(self:centerCrop(im)) 
      end
    end

  end
  return self:typeResults(self.output_cpu, self.labels_cpu)
end

