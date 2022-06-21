#include <float.h>
#include <stdio.h>

__global__ void ConvKernel(
    float* dst, const float* src, int input_channel, int output_channel, int h,
    int w, int kernel_h, int kernel_w, int stride_h, int stride_w, int output_h,
    int output_w, float* kernel, float* bias) {
    int output_x = blockIdx.x * blockDim.x + threadIdx.x;
    int output_y = blockIdx.y * blockDim.y + threadIdx.y;
    int output_c = threadIdx.z;

    if (output_c >= output_channel || output_x >= output_h ||
        output_y >= output_w) {
        return;
    }
    // input channel: 1 output channel: 20 h: 28 w: 28 kernel: 5 5 stride: 1 1
    // NCHW
    float sum = bias[output_c];
    for (int k = 0; k < input_channel; k++) {
        for (int i = 0; i < kernel_h; i++) {
            for (int j = 0; j < kernel_w; j++) {
                float src_value =
                    src[k * h * w + (output_x * stride_h + i) * w +
                        (output_y * stride_w + j)];
                // OIHW
                float kernel_value = kernel
                    [output_c * input_channel * kernel_h * kernel_w +
                     k * kernel_h * kernel_w + i * kernel_w + j];
                sum += src_value * kernel_value;
            }
        }
    }
    dst[output_c * output_h * output_w + output_x * output_w + output_y] = sum;
}

void Convolution(
    float* dst, const float* src, int input_channel, int output_channel, int h,
    int w, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, float* kernel, float* bias, cudaStream_t stream) {
    float* kernelWeights;
    float* biasWeights;
    //  input channel: 1 output channel: 20 h: 28 w: 28 kernel: 5 5 stride: 1 1
    // 20, 24, 24
    cudaMalloc(
        &kernelWeights,
        input_channel * output_channel * kernel_h * kernel_w * 4);
    cudaMalloc(&biasWeights, output_channel * 4);
    cudaMemcpy(
        kernelWeights, kernel,
        input_channel * output_channel * kernel_h * kernel_w * 4,
        cudaMemcpyHostToDevice);
    cudaMemcpy(biasWeights, bias, output_channel * 4, cudaMemcpyHostToDevice);

    int output_h = (h - kernel_h) / stride_h + 1;
    int output_w = (w - kernel_w) / stride_w + 1;

    int block_x = output_h + 1;
    int block_y = output_w + 1;

    ConvKernel<<<
        dim3(block_x, block_y), dim3(1, 1, output_channel), 0, stream>>>(
        dst, src, input_channel, output_channel, h, w, kernel_h, kernel_w,
        stride_h, stride_w, output_h, output_w, kernelWeights, biasWeights);
}
