1. install cuda, cudnn, tensorrt
   
   just copy {lib,include} of {cuda,cudnn,tensorrt} to some place, e.g
   /opt/anaconda3/envs/cuda-11/{include,lib}
   
2. patch tensorrt caffeParser.cpp and build libnvcaffeparser.so, make sure the
   patched libnvcaffeparser.so is linked
   
   ```
   else if (layerMsg.type() == "Softmax")
   {
       pluginName = "SOFTMAX";
   }   
   ```

