CPPFLAGS := -I /opt/anaconda3/envs/cuda-11/include/ -I/usr/include/opencv4
# CPPFLAGS += -DINT8
CXXFLAGS := -g -O0 -MMD
LDFLAGS := -L/opt/anaconda3/envs/cuda-11/lib -L/opt/anaconda3/envs/cuda-11/lib64 -L/home/sunway/source/TensorRT/build/out
LDLIBS := -lnvcaffeparser -lnvinfer -lnvinfer_plugin -lcudnn -lcudart -lstdc++ -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

SRC := $(wildcard *.cpp)
OBJ := $(patsubst %.cpp,%.o,${SRC})
APP := $(patsubst %.cpp,%.elf,${SRC})
RUN_APP := $(patsubst %.cpp,run_%.elf,${SRC})

all: ${APP}

DEP := $(OBJ:.o=.d)
-include ${DEP}

CUDA_KERNEL_SRC:=$(wildcard *.cu)
CUDA_OBJ := $(patsubst %.cu,%.o,${CUDA_KERNEL_SRC})

%.o:%.cu
	nvcc -c $^ -o $@

%.elf:%.o ${CUDA_OBJ}
	gcc $^ ${LDFLAGS} ${LDLIBS} -o $@

${RUN_APP}:run_%.elf:%.elf
	LD_LIBRARY_PATH="/home/sunway/source/TensorRT/build/out:/opt/anaconda3/envs/cuda-11/lib64:/opt/anaconda3/envs/cuda-11/lib"  ./$<

clean:
	rm ${OBJ} ${APP} ${DEP}
