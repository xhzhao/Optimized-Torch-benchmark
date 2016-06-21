require 'sys'

require 'nn'
--require 'mkldnn'
--require 'cunn'

--require 'cudnn'
--cudnn.benchmark = true -- run manual auto-tuner provided by cudnn
--cudnn.verbose = false

-- require 'fbcunn'
-- require 'nnbhwd' -- not compiling anymore, file an issue
local nets = {}
nets[#nets+1] = require 'alexnet'
--nets[#nets+1] = require 'overfeat'
--nets[#nets+1] = require 'vgg_a'
--nets[#nets+1] = require 'googlenet'

local libs = {}
libs[#libs+1] = {nn.SpatialConvolutionMM, nn.SpatialMaxPooling, nn.ReLU, 'BDHW', 'nn'}

libs[#libs+1] = {nn.SpatialConvolutionMKLDNN, nn.SpatialMaxPoolingMKLDNN, nn.ReLUMKLDNN, 'BDHW', 'nn'}
--libs[#libs+1] = {nn.SpatialConvolutionMKLDNN, nn.SpatialMaxPoolingMKLDNN, nn.ReLUMKLDNN, 'BDHW', 'nn'}
print('Running on CPU...')
--print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)

steps = 20 -- nb of steps in loop to average perf
nDryRuns = 3

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
      collectgarbage()
      local nn_model,nn_model_name,nn_size = nets[i](libs[1])
      local mkldnn_model,mkldnn_model_name,mkldnn_size = nets[i](libs[2])
      
      local input = makeInput(libs[1],nn_size)
      local lib_name = libs[1][5]
      print('ModelType: ' .. nn_model_name, 'Kernels: ' .. lib_name,
            'Input shape: ' .. input:size(1) .. 'x' .. input:size(2) ..
               'x' .. input:size(3) .. 'x' .. input:size(4))

      -- dry-run
      local newInput = input

      for i=1,nDryRuns do
         nn_model:zeroGradParameters()
         mkldnn_model:zeroGradParameters()

         local nn_output = nn_model:updateOutput(input)
         local mkldnn_output = mkldnn_model:updateOutput(input)

	 print(nn_output:nDimension(),tonumber(nn_output:cdata().size[0]),tonumber(nn_output:cdata().size[1]))
	 local size=tonumber(nn_output:cdata().size[0]) * tonumber(nn_output:cdata().size[1])
	 nn_output.THNN.SpatialConvolutionMM_compare(nn_output:cdata(), mkldnn_output:cdata(), size)
	--print(nn_output.nDimensions,nn_output.size[0])
         --local gradInput = model:updateGradInput(input, output)
         --model:accGradParameters(input, output)
         collectgarbage()
      end

end

print('')
