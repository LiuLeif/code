#include <stdio.h>

__global__ void Exp(float* output, float* input) {
    output[threadIdx.x] = exp(input[threadIdx.x]);
}

__global__ void Divid(float* output, float* sum) {
    output[threadIdx.x] /= *sum;
}

__global__ void Sum(float* data, float* result) {
    int id = threadIdx.x;
    atomicAdd(result, data[id]);
}

void Softmax(float* output, float* input, int N) {
    float* sum;
    cudaMalloc(&sum, sizeof(float));
    Exp<<<1, N>>>(output, input);
    Sum<<<1, N>>>(output, sum);
    Divid<<<1, N>>>(output, sum);
    cudaFree(sum);
}
