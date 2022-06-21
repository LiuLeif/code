1. install cuda, cudnn, tensorrt
   
   just copy {lib,include} of {cuda,cudnn,tensorrt} to some place, e.g
   /opt/anaconda3/envs/cuda-11/{include,lib}
   
2. patch tensorrt

   1. git clone https://github.com/NVIDIA/TensorRT/, checkout to
      156c59ae86d454fa89146fe65fa7332dbc8c3c2b and apply `tensorrt.diff`
      
   2. build TensorRT 
   
   3. make sure `libnvcaffeparser.so` built from TensorRT is linked by setting
      LD_LIBRARY_PATH (check Makefile for details)
   
   
