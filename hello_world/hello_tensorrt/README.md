# install cuda, cudnn, tensorrt

copy {lib,include} of {cuda,cudnn,tensorrt} to some place, e.g
/opt/anaconda3/envs/cuda-11/{include,lib}

# patch tensorrt

1.  git clone https://github.com/NVIDIA/TensorRT/, checkout to
    156c59ae86d454fa89146fe65fa7332dbc8c3c2b and apply `tensorrt.diff`

2.  build TensorRT

3.  change Makefile based on your local config

# run

make run\_mnist

make run\_googlenet

# run with int8

1.  turn on `CPPFLAGS += -DINT8` in Makefile
2.  make clean
3.  make run\_mnist
