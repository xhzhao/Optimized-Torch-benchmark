function alexnet(lib)

   local SpatialConvolution = lib[1]
   local SpatialMaxPooling = lib[2]
   local ReLU = lib[3]
   local SBatchNorm = nn.SpatialBatchNormalizationMKLDNN


   local features = nn.ConcatMKLDNN(2)
   local fb1 = nn.Sequential() -- branch 1
   --fb1:add(SpatialConvolution(3,48,11,11,4,4,2,2))       -- 224 -> 55
   fb1:add(SpatialConvolution(3,48,11,11,4,4,0,0))       -- 227 -> 55
   fb1:add(ReLU(true))
   fb1:add(SBatchNorm(48))
   fb1:add(SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   fb1:add(SpatialConvolution(48,128,5,5,1,1,2,2))       --  27 -> 27
   fb1:add(ReLU(true))
   fb1:add(SBatchNorm(128))
   fb1:add(SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   fb1:add(SpatialConvolution(128,192,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(ReLU(true))
   fb1:add(SpatialConvolution(192,192,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(ReLU(true))
   fb1:add(SpatialConvolution(192,128,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(ReLU(true))
   fb1:add(SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

  


   local fb2 = fb1:clone() -- branch 2
   for k,v in ipairs(fb2:findModules('nn.SpatialConvolutionMKLDNN')) do
      v:reset() -- reset branch 2's weights
   end

   fb1:get(1).gradInput = nil
   fb2:get(1).gradInput = nil
   features:add(fb1)
   features:add(fb2)
   --features = makeDataParallel(features, nGPU) -- defined in util.lua

   -- 1.3. Create Classifier (fully connected layers)
   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(256*6*6, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Linear(4096, 1000))
  -- classifier:add(nn.LogSoftMax())
   --classifier:cuda()


   -- 1.4. Combine 1.1 and 1.3 to produce final model
   local model = nn.Sequential():add(features):add(classifier)

   return model,'AlexNet_g2',{256,3,227,227}
end

return alexnet
