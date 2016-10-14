function alexnet(lib)

   local features = nn.Sequential()

   local SpatialConvolution = cudnn.SpatialConvolution
   local ReLU = cudnn.ReLU
   local SpatialMaxPooling = cudnn.SpatialMaxPooling
   local SBatchNorm = cudnn.SpatialBatchNormalization
   local LRN = cudnn.SpatialCrossMapLRN

   features:add(SpatialConvolution(3,96,11,11,4,4))       -- 224 -> 55
   features:add(ReLU(true))
   features:add(LRN(5,0.0001,0.75))
   features:add(SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   features:add(SpatialConvolution(96,256,5,5,1,1,2,2,2))       --  27 -> 27
   features:add(ReLU(true))
   features:add(LRN(5,0.0001,0.75))
   features:add(SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   features:add(SpatialConvolution(256,384,3,3,1,1,1,1))      --  13 ->  13
   features:add(ReLU(true))
   features:add(SpatialConvolution(384,384,3,3,1,1,1,1,2))      --  13 ->  13
   features:add(ReLU(true))
   features:add(SpatialConvolution(384,256,3,3,1,1,1,1,2))      --  13 ->  13
   features:add(ReLU(true))
   features:add(SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6


   features:get(1).gradInput = nil

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
   classifier:add(nn.LogSoftMax())
   --classifier:cuda()


   -- 1.4. Combine 1.1 and 1.3 to produce final model
   local model = nn.Sequential():add(features):add(classifier)

   return model,'AlexNet_g2',{256,3,227,227}
end

return alexnet
