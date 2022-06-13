CPPFLAGS:=-I /opt/anaconda3/envs/cuda-11/include/
LDFLAGS:=-L/opt/anaconda3/envs/cuda-11/lib -L/opt/anaconda3/envs/cuda-11/lib64
LDLIBS:=-lnvcaffe_parser -lnvinfer -lcudnn -lstdc++

mnist:mnist.o

run:mnist
	./mnist

clean:
	rm mnist.o mnist
