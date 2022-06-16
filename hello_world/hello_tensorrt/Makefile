all:mnist.elf

CPPFLAGS:=-I /opt/anaconda3/envs/cuda-11/include/
CXXFLAGS:=-g -O0 -MMD
LDFLAGS:= -L/opt/anaconda3/envs/cuda-11/lib -L/opt/anaconda3/envs/cuda-11/lib64 -L/home/sunway/source/TensorRT/build/out
LDLIBS:=-lnvcaffeparser -lnvinfer -lnvinfer_plugin -lcudnn -lcudart -lstdc++ 

SRC:=mnist.cpp
OBJ := $(patsubst %.cpp,%.o,${SRC})

DEP := $(OBJ:.o=.d)
-include ${DEP}

CUDA_KERNEL_SRC:=$(wildcard *.cu)
CUDA_OBJ := $(patsubst %.cu,%.o,${CUDA_KERNEL_SRC})

%.o:%.cu
	nvcc -c $^ -o $@

mnist.elf:${OBJ} ${CUDA_OBJ}
	gcc $^ ${LDFLAGS} ${LDLIBS} -o $@

run:mnist.elf
	LD_LIBRARY_PATH="/home/sunway/source/TensorRT/build/out:/opt/anaconda3/envs/cuda-11/lib64:/opt/anaconda3/envs/cuda-11/lib" ./mnist.elf

clean:
	rm ${OBJ} mnist.elf
	rm ${DEP}
