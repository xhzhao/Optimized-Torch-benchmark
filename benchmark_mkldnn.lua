require 'sys'

require 'nn'
--require 'mkldnn'
--require 'cunn'

--require 'cudnn'
--cudnn.benchmark = true -- run manual auto-tuner provided by cudnn
--cudnn.verbose = false

-- require 'fbcunn'
-- require 'nnbhwd' -- not compiling anymore, file an issue



--sys.compare = false--false
--sys.compare = true--false
--sys.timerEnable = true--true
--sys.timerEnable = false--true

sys.totalTime = 0
sys.convTime = 0
sys.maxpoolingTime = 0
sys.avgpoolingTime = 0
sys.reluTime = 0
sys.sbnTime = 0
sys.linearTime = 0
sys.dropTime = 0
sys.concatTime = 0 -- ConcatTable.lua
sys.concatTime2 = 0 -- Concat.lua



local nets = {}
nets[#nets+1] = require 'alexnet'
--nets[#nets+1] = require 'vgg_e'
--nets[#nets+1] = require 'googlenet'
--nets[#nets+1] = require 'resnet'
--nets[#nets+1] = require 'resnet_test'


local libs = {}
--libs[#libs+1] = {cudnn.SpatialConvolution, cudnn.SpatialMaxPooling, cudnn.ReLU, 'BDHW', 'cudnn'}
-- libs[#libs+1] = {fbnn.SpatialConvolution, cudnn.SpatialMaxPooling, cudnn.ReLU, 'BDHW', 'fbnn'}
libs[#libs+1] = {nn.SpatialConvolutionMKLDNN, nn.SpatialMaxPoolingMKLDNN, nn.ReLUMKLDNN, 'BDHW', 'nn'}
--libs[#libs+1] = {mkldnn.SpatialConvolutionMM, mkldnn.SpatialMaxPooling, mkldnn.ReLU, 'BDHW', 'nn'}
-- libs[#libs+1] = {nn.SpatialConvolutionBHWD, nn.SpatialMaxPoolingBHWD, nn.ReLU, 'BHWD', 'nnBHWD'}
print('Running on CPU...')
--print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)

steps = 10 -- nb of steps in loop to average perf
nDryRuns = 15

torch.setdefaulttensortype('torch.FloatTensor')

function makeInput(config, size)
   local layout = config[4]
   local osize
   if layout == 'BDHW' then
      osize = size
   elseif layout == 'DHWB' then
      osize = {size[2],size[3],size[4],size[1]}
   elseif layout == 'BHWD' then
      osize = {size[1], size[3], size[4], size[2]}
   end
   return torch.randn(torch.LongStorage(osize))
   --return torch.randn(torch.FloatStorage(osize))
end





for i=1,#nets do
   for j=1,#libs do
      collectgarbage()
      local model,model_name,size = nets[i](libs[j])
      --model=model:cuda()
      local input = makeInput(libs[j],size)
      local lib_name = libs[j][5]
      print('ModelType: ' .. model_name, 'Kernels: ' .. lib_name,
            'Input shape: ' .. input:size(1) .. 'x' .. input:size(2) ..
               'x' .. input:size(3) .. 'x' .. input:size(4))

--	print(model)

	local timeStart,timeEnd
      -- dry-run
      if sys.timerEnable then
         for i=1,nDryRuns do
         print("forward start")
	 timeStart = sys.clock()
         model:zeroGradParameters()
         local output = model:updateOutput(input)
         print("backward start 1")
         local gradInput = model:updateGradInput(input, output)
         print("backward start 2")
         model:accGradParameters(input, output)
	 timeEnd = sys.clock()

         --cutorch.synchronize()
         collectgarbage()
	--print("total time = ",sys.totalTime,", convTime = ", sys.convTime, ", sys.maxpoolingTime = ",sys.maxpoolingTime,", sys.avgpoolingTime = ",sys.avgpoolingTime,", sys.reluTime = ",sys.reluTime, ", sys.sbnTime = ",sys.sbnTime,",sys.linearTime=",sys.linearTime,",sys.dropTime=",sys.dropTime,", sys.concatTime = ",sys.concatTime,",sys.concatTime2=",sys.concatTime2,", sum = ",(sys.convTime+sys.maxpoolingTime+sys.avgpoolingTime+sys.reluTime+sys.sbnTime+sys.linearTime+sys.dropTime+sys.concatTime+sys.concatTime2))

	print("sys.totalTime =		",sys.totalTime)
	print("ys.convTime =		",sys.convTime)
	print("sys.maxpoolingTime =	",sys.maxpoolingTime)
	print("sys.avgpoolingTime =	",sys.avgpoolingTime)
	print("sys.reluTime =		",sys.reluTime)
	print("sys.sbnTime =		",sys.sbnTime)
	print("sys.linearTime =	",	sys.linearTime)
	print("sys.dropTime=		",sys.dropTime)
	print("sys.concatTime=		",sys.concatTime)
	print("sys.concatTime2 =	",sys.concatTime2)
	print("sum = 			",sys.convTime+sys.maxpoolingTime+sys.avgpoolingTime+sys.reluTime+sys.sbnTime+sys.linearTime+sys.dropTime+sys.concatTime+sys.concatTime2)
	print("------")


	sys.totalTime = timeEnd - timeStart
	sys.convTime = 0
	sys.maxpoolingTime = 0
	sys.avgpoolingTime = 0
	sys.reluTime = 0
	sys.sbnTime = 0
	sys.linearTime = 0
	sys.dropTime = 0
	sys.concatTime = 0
	sys.concatTime2 = 0


         end
      else

         for i=1,nDryRuns do
         sys.tic()
	 model:zeroGradParameters()
         local output = model:updateOutput(input)
         local gradInput = model:updateGradInput(input, output)
         model:accGradParameters(input, output)
	 print("time = ",sys.toc())
         --cutorch.synchronize()
         collectgarbage()
         end

      end

	sys.timerEnable = false

      local tmf, tmbi, tmbg
      sys.tic()
      for t = 1,steps do
         output = model:updateOutput(input)
      end
      --cutorch.synchronize()
      tmf = sys.toc()/steps
      print(string.format("%-30s %25s %10.2f", lib_name, ':updateOutput():', tmf*1000))

      collectgarbage()
      sys.tic()
      for t = 1,steps do
         model:updateGradInput(input, output)
         collectgarbage()
      end
      --cutorch.synchronize()
      tmbi = sys.toc()/steps
      print(string.format("%-30s %25s %10.2f", lib_name, ':updateGradInput():', tmbi*1000))

      collectgarbage()
      sys.tic()
      local ok = 1
      for t = 1,steps do
         ok = pcall(function() model:accGradParameters(input, output) end)
      end
      --cutorch.synchronize()
      tmbg = sys.toc()/steps
      if not ok then
         print(string.format("%-30s %25s %s", lib_name, ':accGradParameters():', 'FAILED!'))
      else
         print(string.format("%-30s %25s %10.2f", lib_name, ':accGradParameters():', tmbg*1000))
      end
      print(string.format("%-30s %25s %10.2f", lib_name, ':Forward:', (tmf)*1000))
      print(string.format("%-30s %25s %10.2f", lib_name, ':Backward:', (tmbi+tmbg)*1000))
      print(string.format("%-30s %25s %10.2f", lib_name, ':TOTAL:', (tmf+tmbi+tmbg)*1000))
      print()
   end
end

print('')
