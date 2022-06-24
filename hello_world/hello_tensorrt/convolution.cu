#include <float.h>
#include <stdio.h>
#include <unistd.h>

__global__ void ConvKernel(
    float* dst, const float* src, int input_channel, int output_channel, int h,
    int w, int kernel_h, int kernel_w, int stride_h, int stride_w, int output_h,
    int output_w, int padding_h, int padding_w, float* kernel, float* bias) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    int channel = global_id / output_h / output_w;
    int output_x = global_id % (output_h * output_w) / output_w;
    int output_y = global_id % (output_h * output_w) % output_w;

    if (channel >= output_channel || output_x >= output_h ||
        output_y >= output_w) {
        return;
    }
    // input channel: 1 output channel: 20 h: 28 w: 28 kernel: 5 5 stride: 1 1
    // NCHW
    float sum = bias[channel];
    for (int k = 0; k < input_channel; k++) {
        for (int i = 0; i < kernel_h; i++) {
            for (int j = 0; j < kernel_w; j++) {
                int orig_x = output_x * stride_h + i;
                int orig_y = output_y * stride_w + j;

                float src_value = 0.0;
                if (orig_x >= padding_h && orig_x < padding_h + h &&
                    orig_y >= padding_w && orig_y < padding_w + w) {
                    src_value =
                        src[k * h * w + (orig_x - padding_h) * w + orig_y -
                            padding_w];
                }
                // OIHW
                float kernel_value = kernel
                    [channel * input_channel * kernel_h * kernel_w +
                     k * kernel_h * kernel_w + i * kernel_w + j];
                sum += src_value * kernel_value;
            }
        }
    }
    dst[channel * output_h * output_w + output_x * output_w + output_y] = sum;
}

void Convolution(
    float* dst, const float* src, int input_channel, int output_channel, int h,
    int w, int kernel_h, int kernel_w, int stride_h, int stride_w,
    int padding_h, int padding_w, float* kernel, float* bias,
    cudaStream_t stream) {
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

    // NOTE: `floor` for convolution
    int output_h = (h - kernel_h + 2 * padding_h) / stride_h + 1;
    int output_w = (w - kernel_w + 2 * padding_w) / stride_w + 1;

    int total_size = output_channel * output_h * output_w;
    ConvKernel<<<(int)(total_size / 128) + 1, 128, 0, stream>>>(
        dst, src, input_channel, output_channel, h, w, kernel_h, kernel_w,
        stride_h, stride_w, output_h, output_w, padding_h, padding_w,
        kernelWeights, biasWeights);
}
