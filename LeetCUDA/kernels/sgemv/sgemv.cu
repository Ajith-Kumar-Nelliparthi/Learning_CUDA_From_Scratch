#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>
#include <math.h>
#include <algorithm>
#include <vector>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define WARP_SIZE 32

// FP 32
// warp reduce sum
template <const int kwarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int mask = kwarpSize >> 1, mask >= 1, mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// Y = α*(A[MK] * X[K]) + β.y
__global__ void sgemv_k32_f32_kernel(float *x, float *a, float *y, int M, int K) {
    int tx = threadIdx.x;       // 0 - 31
    int ty = threadIdx.y;       // 0 - 4
    int bid = blockIdx.x;        // 0 - M/4
    int m = bid * blockDim.y + threadIdx.y;    // (0-M/4) * 4 + (0-3)
    int lane = tx % WARP_SIZE;   // 0-31
    if (m < M) {
        float sum = 0.0f;
        int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
#pragma unroll
        for (int w=0; w<NUM_WARPS; w++) { 
            int k = w * WARP_SIZE + lane;              // 0-32, 32-64, 64-96, 96-128
            sum += a[m * K + k] * x[k];             // m=rows(0-4) K=32, k=(32,64,96,128) 
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        if (lane == 0) {
            y[m] = sum;
        }
    }
}

__global__ void sgemv_k128_f32x4_kernel(float *x, float *a, float *y, int M,int K) {
    int tx = threadIdx.x; // 0-31
    int ty = threadIdx.y ;// 0-4
    int bx = blockIdx.x;  // 0-M/4
    int lane = tx % WARP_SIZE; // 0-31
    int m = bx * blockDim.y + threadIdx.y; // (0-M/4) * 4 (No of rows in y) + (0-3)
    if (m < M) {
        float sum = 0.0f;
        int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
#pragma unroll
        for (int w=0; w<NUM_WARPS; w++) {
            int k = (w * WARP_SIZE + lane) * 4;
            float4 reg_x = FLOAT4(x[k]);
            float4 reg_a = FLOAT4(a[m * K + k]);
            sum += (reg_a.x * reg_x.x + reg_a.y * reg_x.y + reg_a.w * reg_x.w + reg_a.z * reg_x.z);
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        if (lane == 0) y[m] = sum;
    } 
}