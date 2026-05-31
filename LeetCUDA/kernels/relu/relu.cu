#include <stdio.h>
#include <float.h>
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torhch/types.h>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// fp 32
__global__ void relu_f32_kernel(float *x, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = fmaxf(0.0f, x[idx]);
    }
}

// fp 32 x 4
__global__ void relu_f32x4_kernel(float *x, float *y, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        float4 reg_x = FLOAT4(x[idx]);
        float4 reg_y;
        reg_y.x = fmaxf(0.0f, reg_x.x);
        reg_y.y = fmaxf(0.0f, reg_x.y);
        reg_y.z = fmaxf(0.0f, reg_x.z);
        reg_y.w = fmaxf(0.0f, reg_x.w);
        FLOAT4(y[idx]) = reg_y;
    }
}