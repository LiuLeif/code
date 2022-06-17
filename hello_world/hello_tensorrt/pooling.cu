#include <float.h>
#include <stdio.h>

__global__ void Max(
    float* dst, const float* src, int h, int w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int output_h, int output_w) {
    int channel = blockIdx.x;
    int output_x = threadIdx.x;
    int output_y = threadIdx.y;
    float max_value = -1000000.0;
    for (int i = 0; i < kernel_h; i++) {
        for (int j = 0; j < kernel_w; j++) {
            float curr_value =
                src[channel * h * w + (output_x * stride_h + i) * w +
                    (output_y * stride_w + j)];
            max_value = max(max_value, curr_value);
        }
    }
    dst[channel * output_h * output_w + output_x * output_w + output_y] =
        max_value;
}

void Pooling(
    float* dst, const float* src, int channel, int h, int w, int method,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, cudaStream_t stream) {
    int output_h = (h - kernel_h) / stride_h + 1;
    int output_w = (w - kernel_w) / stride_w + 1;

    Max<<<channel, dim3(output_h, output_w), 0, stream>>>(
        dst, src, h, w, kernel_h, kernel_w, stride_h, stride_w, output_h,
        output_w);
}
