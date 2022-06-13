CPPFLAGS:=-I /opt/anaconda3/envs/cuda-11/include/
LDFLAGS:=-L/opt/anaconda3/envs/cuda-11/lib -L/opt/anaconda3/envs/cuda-11/lib64
LDLIBS:=-lnvcaffe_parser -lnvinfer -lcudnn -lcudart -lstdc++

mnist.elf:mnist.o
	gcc $^ ${LDFLAGS} ${LDLIBS} -o $@

run:mnist.elf
	./mnist.elf

clean:
	rm mnist.o mnist.elf
