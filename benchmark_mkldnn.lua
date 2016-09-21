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
sys.convTime_forward = 0
sys.convTime_backward = 0
sys.maxpoolingTime_forward = 0
sys.maxpoolingTime_backward = 0
sys.avgpoolingTime_forward = 0
sys.avgpoolingTime_backward = 0
sys.reluTime_forward = 0
sys.reluTime_backward = 0
sys.sbnTime_forward = 0
sys.sbnTime_backward = 0
sys.lrnTime_forward = 0
sys.lrnTime_backward = 0
sys.linearTime_forward = 0
sys.linearTime_backward = 0
sys.dropTime_forward = 0
sys.dropTime_backward = 0
sys.concatTableTime_forward = 0
sys.concatTableTime_backward = 0
sys.concatTime_forward = 0
sys.concatTime_backward = 0
sys.thresholdTime_forward = 0
sys.thresholdTime_backward = 0


local nets = {}
--nets[#nets+1] = require 'alexnet'
--nets[#nets+1] = require 'alexnet_g1'
--nets[#nets+1] = require 'alexnet_g2'
--nets[#nets+1] = require 'vgg_e'
nets[#nets+1] = require 'googlenet'
--nets[#nets+1] = require 'resnet'


local libs = {}
--libs[#libs+1] = {cudnn.SpatialConvolution, cudnn.SpatialMaxPoolingMKLDNN, cudnn.ReLU, 'BDHW', 'cudnn'}
-- libs[#libs+1] = {fbnn.SpatialConvolution, cudnn.SpatialMaxPooling, cudnn.ReLU, 'BDHW', 'fbnn'}
libs[#libs+1] = {nn.SpatialConvolutionMKLDNN, nn.SpatialMaxPoolingMKLDNN, nn.ReLUMKLDNN, 'BDHW', 'nn'}
--libs[#libs+1] = {nn.SpatialConvolution, nn.SpatialMaxPooling, nn.ReLU, 'BDHW', 'nn'}
--libs[#libs+1] = {mkldnn.SpatialConvolutionMM, mkldnn.SpatialMaxPooling, mkldnn.ReLU, 'BDHW', 'nn'}
-- libs[#libs+1] = {nn.SpatialConvolutionBHWD, nn.SpatialMaxPoolingBHWD, nn.ReLU, 'BHWD', 'nnBHWD'}
print('Running on CPU...')
--print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)

steps = 10 -- nb of steps in loop to average perf
nDryRuns = 5

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
	print("sys.convTime_forward =		",sys.convTime_forward)
	print("sys.convTime_backward =		",sys.convTime_backward)
	print("sys.maxpoolingTime_forward =	",sys.maxpoolingTime_forward)
	print("sys.maxpoolingTime_backward =	",sys.maxpoolingTime_backward)
	print("sys.avgpoolingTime_forward =	",sys.avgpoolingTime_forward)
	print("sys.avgpoolingTime_backward =	",sys.avgpoolingTime_backward)
	print("sys.reluTime_forward =		",sys.reluTime_forward)
	print("sys.reluTime_backward =		",sys.reluTime_backward)
	print("sys.lrnTime_forward =		",sys.lrnTime_forward)
	print("sys.lrnTime_backward =		",sys.lrnTime_backward)
	print("sys.sbnTime_forward =		",sys.sbnTime_forward)
	print("sys.sbnTime_backward =		",sys.sbnTime_backward)
	print("sys.linearTime_forward =	",	sys.linearTime_forward)
	print("sys.linearTime_backward =	",	sys.linearTime_backward)
	print("sys.dropTime_forward=		",sys.dropTime_forward)
	print("sys.dropTime_backward=		",sys.dropTime_backward)
	print("sys.concatTableTime_forward=		",sys.concatTableTime_forward)
	print("sys.concatTableTime_backward=		",sys.concatTableTime_backward)
	print("sys.concatTime_forward =		",sys.concatTime_forward)
	print("sys.concatTime_backward=		",sys.concatTime_backward)
	print("sys.thresholdTime_forward =      ",sys.thresholdTime_forward)
	print("sys.thresholdTime_backward =      ",sys.thresholdTime_backward)
	print("sum = 			",sys.convTime_forward+sys.convTime_backward+sys.maxpoolingTime_forward+sys.maxpoolingTime_backward+sys.avgpoolingTime_forward+sys.avgpoolingTime_backward+sys.reluTime_forward+sys.reluTime_backward+sys.sbnTime_forward+sys.sbnTime_backward+sys.linearTime_forward+sys.linearTime_backward+sys.dropTime_forward+sys.dropTime_backward+sys.concatTime_forward+sys.concatTime_backward+sys.concatTableTime_forward+sys.concatTableTime_backward+sys.thresholdTime_forward+sys.thresholdTime_backward+sys.lrnTime_forward+sys.lrnTime_backward)
	print("------")


	sys.totalTime = timeEnd - timeStart
	sys.convTime_forward = 0
	sys.convTime_backward = 0
	sys.maxpoolingTime_forward = 0
	sys.maxpoolingTime_backward = 0
	sys.avgpoolingTime_forward = 0
	sys.avgpoolingTime_backward = 0
	sys.reluTime_forward = 0
	sys.reluTime_backward = 0
	sys.sbnTime_forward = 0
	sys.sbnTime_backward = 0
	sys.lrnTime_forward = 0
	sys.lrnTime_backward = 0
	sys.linearTime_forward = 0
	sys.linearTime_backward = 0
	sys.dropTime_forward = 0
	sys.dropTime_backward = 0
	sys.concatTableTime_forward = 0
	sys.concatTableTime_backward = 0
	sys.concatTime_forward = 0
	sys.concatTime_backward = 0
	sys.thresholdTime_forward = 0
	sys.thresholdTime_backward = 0
         end
      else

         for i=1,nDryRuns do
         sys.tic()
	 model:zeroGradParameters()
         local output = model:updateOutput(input)
         local gradInput = model:updateGradInput(input, output)
         model:accGradParameters(input, output)
	 print("totaltime = ",sys.toc())
         --cutorch.synchronize()
         collectgarbage()
         end

      end

      if steps > 0 then
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
end

print('haha end')
