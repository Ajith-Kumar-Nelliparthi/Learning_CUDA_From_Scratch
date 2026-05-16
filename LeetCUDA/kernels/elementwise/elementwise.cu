#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <float.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <torch/types.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<bfloat2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// fp32
// elementwise add
__global__ void elementwise_add_f32_kernel(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// elementwise add + Vec4
__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if ((idx + 3) < N) {
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        FLOAT4(c[idx]) = reg_c;
    } else if (idx < N) {
        for (int i = 0; (idx + i) < N; i++) {
            c[idx + i] = a[idx + i] + b[idx + i];
        }
    }
}

// FP16
// elementwise add
__global__ void elementwise_add_fp16_kernel(half *a, half *b, half *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

// fp16 x 2
__global__ void elementwise_add_fp16x2_kernel(half *a, half *b, half *c, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if ((idx + 1) < N) {
        half2 reg_A = HALF2(a[idx]);
        half2 reg_B = HALF2(b[idx]);
        half2 reg_c;
        reg_c.x = __hadd(reg_A.x, reg_B.x);
        reg_c.y = __hadd(reg_A.y, reg_B.y);
        HALF2(c[idx]) = reg_c;
    } else if (idx < N) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

// fp16 x 8
__global__ void elementwise_add_fp16x8_kernel(half *a, half *b, half *c, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if ((idx + 7) < N) {
        half2 reg_a_0 = HALF2(a[idx + 0]);
        half2 reg_a_1 = HALF2(a[idx + 2]);
        half2 reg_a_2 = HALF2(a[idx + 4]);
        half2 reg_a_3 = HALF2(a[idx + 6]);
        half2 reg_b_0 = HALF2(b[idx + 0]);
        half2 reg_b_1 = HALF2(b[idx + 2]);
        half2 reg_b_2 = HALF2(b[idx + 4]);
        half2 reg_b_3 = HALF2(b[idx + 6]);
        half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3;
        reg_c_0.x = __hadd(reg_a_0.x, reg_b_0.x);
        reg_c_0.y = __hadd(reg_a_0.y, reg_b_0.y);
        reg_c_1.x = __hadd(reg_a_1.x, reg_b_1.x);
        reg_c_1.y = __hadd(reg_a_1.y, reg_b_1.y);
        reg_c_2.x = __hadd(reg_a_2.x, reg_b_2.x);
        reg_c_2.y = __hadd(reg_a_2.y, reg_b_2.y);
        reg_c_3.x = __hadd(reg_a_3.x, reg_b_3.x);
        reg_c_3.y = __hadd(reg_a_3.y, reg_b_3.y);
        HALF2(c[idx + 0]) = reg_c_0;
        HALF2(c[idx + 2]) = reg_c_1;
        HALF2(c[idx + 4]) = reg_c_2;
        HALF2(c[idx + 6]) = reg_c_3;
    } else if (idx < N) {
        for (int i = 0; (idx + i) < N; i++) {
            c[idx + i] = __hadd(a[idx + i], b[idx + i]);
        }
    }
}