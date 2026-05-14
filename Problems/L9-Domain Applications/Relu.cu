#include <stdio.h>
#include <cuda_runtime.h>

// ReLU activation function kernel
__global__ void relu(float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmaxf(0.0f, in[idx]);
    }
}

// Sigmoid activation function kernel
__global__ void sigmoid(float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0f / (1.0f + expf(-in[idx]));
    }
}

// TanH activation function kernel
__global__ void tanh_activation(float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tanhf(in[idx]);
    }
}

// 2D Convolution kernel
#define TILE_SIZE 16
__global__ void Conv2D(float *input, float *kernel, float *output, int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.x;
    int col = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (row < input_rows - kernel_rows + 1 && col < input_cols - kernel_cols + 1) {
        float sum = 0.0f;
        for (int i = 0; i < kernel_rows; i++) {
            for (int j = 0; j < kernel_cols; j++) {
                sum += input[(row + i) * input_cols + (col + j)] * kernel[i * kernel_cols + j];
            }
        }
        output[row * (input_cols - kernel_cols + 1) + col] = sum;
    }
}