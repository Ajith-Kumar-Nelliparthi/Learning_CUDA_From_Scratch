#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <torch/types.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP 32
// WARP REDUCE SUM
template <const int kwarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = kwarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// BLOCK REDUCE SUM FOR FP 32
template <const int NUM_THREADS = 256>
__device__ __forceinline__ float block_reduce_sum_f32(float val) {
    const int tid = threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];

    val = warp_reduce_sum_f32<WARP_SIZE>(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();

    val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
    val = warp_reduce_sum_f32<NUM_WARPS>(val);
    return val;
}

template <const int NUM_THREADS = 256>
__global__ void rms_norm_f32_kernel(float *x, float *y, int g, int N, int K) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    const float epsilon = 1e-5f;
    __shared__ float s_var;

    float val = (idx < N * K) ? x[idx] : 0.0f;
    float variance = val * val;
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    if (tid == 0) s_var = rsqrtf(variance / (float)K + epsilon);
    __syncthreads();
    if (idx < N * K) {
        y[idx] = (val * s_var) + g;
    }
}

template <const int NUM_THREADS = 256>
__global__ void rms_norm_f32x4_kernel(float *x, float *y, int g, int N, int K) {
    int tid = threadIdx.x;
    int idx = 4 * (blockIdx.x * blockDim.x + tid);
    constexpr float epsilon = 1e-5f;
    __shared__ float s_var;

    float4 reg_x = FLOAT4(x[idx]);
    float variance = ((idx < N * K) ? (reg_x.x * reg_x.x + reg_x.y * reg_x.y 
                                    + reg_x.z * reg_x.z + reg_x.w * reg_x.w) : 0.0f);
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    if (tid == 0) s_var = rsqrtf(variance / (float)K + epsilon);
    __syncthreads();
    
    float4 reg_y;
    reg_y.x = reg_x.x * s_var + g;
    reg_y.y = reg_x.y * s_var + g;
    reg_y.z = reg_x.z * s_var + g;
    reg_y.w = reg_x.w * s_var + g;
    if (idx < N * K) FLOAT4(y[idx]) = reg_y;
} 