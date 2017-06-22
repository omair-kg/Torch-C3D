--[[
Code in this file borrows from https://github.com/MichaelMathieu/datasources
I have made changes so that the class can handle loading images instead of loading videos and extracting frames.
--]]
require 'torch'
require 'io'
require 'datasources.datasource'
require 'paths'
require 'image'
require 'math'
require 'xlua'
--[[
params.nInputFrames: number of frames in one chunk
params.datapath: Change this for other datasets, default is ucf101
--]]
local UCF101_jpgDatasource, parent = torch.class('UCF101_jpgDatasource', 'ClassDatasource')

function UCF101_jpgDatasource:__init(params)
  parent.__init(self)
  assert(params.nInputFrames ~= nil, "UCF101Dataset: must specify nInputFrames")
  self.datapath = params.datapath or '/mnt/data0/poseEstimation_torch/h5_files/UCF/UCF-101_jpgs'
  local setfiles = {train = 'trainlist01.txt', test = 'testlist01.txt'}
  assert(paths.dirp(self.datapath), 'Path ' .. self.datapath .. ' does not exist')
  -- create index list for classes
  f = io.open("/mnt/data0/poseEstimation_torch/h5_files/UCF/ucfTrainTestlist/classInd.txt",'r')
  assert(f ~= nil, 'File not found.')
  self.classList = {}
  for line in f:lines() do
    if string.byte(line:sub(-1,-1)) == 13 then
      --remove the windows carriage return
      line = line:sub(1,-2)
    end
    local class, classidx
    class =line:sub(line:find(' ')+1, -1)
    classidx = tonumber(line:sub(1, line:find(' ')))
    if self.classList[class] == nil then
      self.classList[class] = {}
    end
    table.insert(self.classList[class], classidx)
  end
  f:close()

  local classes = paths.dir(self.datapath)
  self.classes = {}
  self.sets = {train = {}, test = {}}
  self.fileList = {train = {} , test = {}}
  self.labelsList = {train = {} , test = {}}
  for _, set in pairs{'train', 'test'} do
    local f = io.open(paths.concat('/mnt/data0/poseEstimation_torch/h5_files/UCF/ucfTrainTestlist/', setfiles[set]), 'r')
    assert(f ~= nil, 'File ' .. paths.concat(self.datapath, setfiles[set]) .. ' not found.')
    for line in f:lines() do
      if string.byte(line:sub(-1,-1)) == 13 then
        --remove the windows carriage return
        line = line:sub(1,-2)
      end
      local filename, class
      if set == 'train' then
        filename = line:sub(1, line:find(' ')-5)
        classidx = tonumber(line:sub(line:find(' ')+1, -1))
        class = filename:sub(1, filename:find('/')-1)
        self.classes[classidx] = class
      else
        filename = line:sub(1, #line-4)
        class = filename:sub(1, filename:find('/')-1)
      end
      local avifile = filename:sub(filename:find('/')+1,-1)
      --local completePath = (filename .. '/' .. avifile)
      local classIndex = self.classList[class]

      if self.sets[set][class] == nil then
        self.sets[set][class] = {}
      end
      table.insert(self.sets[set][class], avifile)
      table.insert(self.fileList[set],filename)
      table.insert(self.labelsList[set],classIndex)
    end
    f:close()
    local n = 0
    for _, _ in pairs(self.sets[set]) do
      n = n + 1
    end
    assert(n == 101)
  end
  self.nbframes = {}
  assert(#self.classes == 101)
  self.nInputFrames = params.nInputFrames
  self.nChannels, self.nClasses = 3, 101
  self.h, self.w = 112, 112
  self.mean= {0.39676561420733 , 0.38167717973734 , 0.35316953393671}
  self.std =  {0.23639146288122,0.2311516201128,0.22791352961204}
  
end

function UCF101_jpgDatasource:randomCrop(im)
  local weird_h ,weird_w = 128 , 171      
  im = image.scale(im, weird_w,weird_h)

  local h, w = im:size(2), im:size(3)
  if h > self.h then
    local y = torch.random(h - self.h + 1)
    im = im:narrow(2, y, self.h)
  end
  if w > self.w then
    local x = torch.random(w - self.w + 1)
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
  assert(self.sets[set] ~= nil, 'Unknown set ' .. set)
  self.output_cpu:resize(batchSize, self.nInputFrames, self.nChannels, self.h, self.w)
  self.labels_cpu:resize(batchSize)
  -- populate one batch
  --time = sys.clock()
  for i = 1, batchSize do
    index = batch[i]
    -- select a class at random
    self.labels_cpu[i] = self.labelsList[set][index][1]

    -- determine the number of images in the selected instance
    local temp_FN = self.fileList[set][index]
    local filepath = paths.concat(self.datapath, self.fileList[set][index])
    local avifile = temp_FN:sub(temp_FN:find('/')+1,-1)
    local nframes =  #paths.dir(filepath)-3
    -- choose a random start point for reading in frames
    if (nframes - (self.nInputFrames-1)>0) then
      local istart = torch.random(nframes - (self.nInputFrames-1))
      for j = 1, self.nInputFrames do
        impath = (paths.concat(filepath,avifile) .. '_%06d.jpg'):format(j+istart)
        im = image.load(impath)
        for threeChannels=1,3 do -- channels
          im[{{threeChannels},{},{}}]:add(-self.mean[threeChannels])
          im[{{threeChannels},{},{}}]:div(self.std[threeChannels]) 
        end
        if string.match(set, "train") then
          self.output_cpu[i][j]:copy(self:randomCrop(im)) 
        else
          self.output_cpu[i][j]:copy(self:centerCrop(im)) 
        end

      end
    end
  end
  return self:typeResults(self.output_cpu, self.labels_cpu)
end

function UCF101_jpgDatasource:nextBatch_legacy(batchSize,set)
  assert(batchSize ~= nil, 'nextBatch: must specify batchSize')
  assert(self.sets[set] ~= nil, 'Unknown set ' .. set)
  self.output_cpu:resize(batchSize, self.nInputFrames, self.nChannels, self.h, self.w)
  self.labels_cpu:resize(batchSize)
  -- populate one batch
  --time = sys.clock()
  for i = 1, batchSize do
    local done = false
    while not done do
      -- select a class at random
      local iclass = torch.random(self.nClasses)
      local class = self.classes[iclass]
      -- select an instance of the class at random
      local idx = torch.random(#self.sets[set][class])

      -- determine the number of images in the selected instance
      local filepath = paths.concat(self.datapath, class, self.sets[set][class][idx])
      local nframes =  #paths.dir(filepath)-3
      self.labels_cpu[i] = iclass
      -- choose a random start point for reading in frames
      if (nframes - (self.nInputFrames-1)>0) then
        
        local istart = torch.random(nframes - (self.nInputFrames-1))
        for j = 1, self.nInputFrames do
          impath = (paths.concat(filepath,self.sets[set][class][idx]) .. '%04d.jpg'):format(j+istart)
          im = image.load(impath)
          for threeChannels=1,3 do -- channels
            im[{{threeChannels},{},{}}]:add(-self.mean[threeChannels])
            im[{{threeChannels},{},{}}]:div(self.std[threeChannels]) 
           
          end
          if string.match(set, "train") then
            self.output_cpu[i][j]:copy(self:randomCrop(im)) 
            
          else
            self.output_cpu[i][j]:copy(self:centerCrop(im)) 
          end
        end
        done = true
      end
    end
  end
  return self:typeResults(self.output_cpu, self.labels_cpu)
end

function UCF101_jpgDatasource:calculate_mean()
  local frame_mean = {0,0,0}
  local frame_std = {0,0,0}
  local set = 'train'
  local total_images = 0
  for iclass = 1,self.nClasses do -- cycle though classes
    local class = self.classes[iclass]
    for num_seqs = 1,#self.sets[set][class] do -- cycle through sequences in classes
      local filepath = paths.concat(self.datapath,class,self.sets[set][class][num_seqs])
      local nframes = #paths.dir(filepath)-3
      for idx = 1,nframes do
        impath = (paths.concat(filepath,self.sets[set][class][num_seqs]) .. '_%06d.jpg'):format(idx)
        im = image.load(impath)
        for threeChannels=1,3 do 
          frame_mean[threeChannels] = frame_mean[threeChannels] + im[threeChannels]:mean()
          frame_std[threeChannels] = frame_std[threeChannels] + im[threeChannels]:std()
        end
        total_images = total_images+1
            xlua.progress(total_images,1642896)
      end      
    end   
  end
  for threeChannels=1,3 do 
    frame_mean[threeChannels] = frame_mean[threeChannels]/total_images
    frame_std[threeChannels] = frame_std[threeChannels]/total_images
  end
  print(frame_mean)
  print(frame_std)
  return frame_mean,frame_std
end

