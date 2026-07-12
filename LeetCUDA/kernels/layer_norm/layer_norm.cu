#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <float.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <torch/types.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat16 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP 32
// warp reduce sum
template <const int kwarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = kwarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// block reduce sum
template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
    const int tid = threadIdx.x;
    const int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];

    val = warp_reduce_sum_f32<WARP_SIZE>(val);
    if (lane_id == 0) shared[warp_id] = val;
    __syncthreads();

    val = (lane_id < NUM_WARPS) ? shared[lane_id] : 0.0f;
    val = warp_reduce_sum_f32<NUM_WARPS>(val);
    return val;
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f32_kernel(float *x, float *y, float g, float b, int N, int K) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = blockIdx.x* blockDim.x + tid;
    const float epsilon = 1e-5f;

    __shared__ float s_mean;
    __shared__ float s_var;
    float value = (idx < N * K) ? x[idx] : 0.0f;
    float sum = block_reduce_sum_f32<NUM_THREADS>(value);
    if (tid == 0) s_mean = sum / (float)K;
    __syncthreads();

    // variance
    float variance = (value - s_mean) * (value - s_mean);
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    if (tid == 0) s_var = rsqrtf(variance / (float)K + epsilon);
    __syncthreads();

    if (idx < N * K) {
        y[idx] = ((value - s_mean) * s_var) * g + b;
    }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f32x4_kernel(float *x, float *y, float g, float b, int N, int K) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = (bid * blockDim.x + tid) * 4;
    const float epsilon = 1e-5f;

    __shared__ float s_mean;
    __shared__ float s_var;
    float4 reg_x = FLOAT4(x[idx]);
    float value = (idx < N * K) ? (reg_x.x + reg_x.y + reg_x.w + reg_x.z) : 0.0f;
    float sum = block_reduce_sum_f32<NUM_THREADS>(value);
    if (tid == 0) s_mean = sum / (float)K;
    __syncthreads();

    float4 reg_var;
    reg_var.x = reg_x.x - s_mean;
    reg_var.y = reg_x.y - s_mean;
    reg_var.w = reg_x.w - s_mean;
    reg_var.z = reg_x.z - s_mean;
    float variance = reg_var.x * reg_var.x + reg_var.y * reg_var.y + 
                    reg_var.w * reg_var.w + reg_var.z * reg_var.z;
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    if (tid == 0) s_var = rsqrtf(variance / (float)K + epsilon);
    __syncthreads();

    float4 reg_y;
    reg_y.x = reg_var.x * s_var * g + b;
    reg_y.y = reg_var.y * s_var * g + b;
    reg_y.w = reg_var.w * s_var * g + b;
    reg_y.z = reg_var.z * s_var * g + b;
    if (idx < N * K) {
        float4(y[idx]) = reg_y;
    }
}

// FP 16
// warp reduce sum : half
template <const int kwarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
#pragma unroll
    for (int mask = kwarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <const int kwarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {
    float val_f32 = __half2float(val);
#pragma unroll
    for (int mask = kwarpSize >> 1; mask >= 1; mask >>= 1) {
        val_f32 += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val_f32;
}

template <const int NUM_THREADS = 256>
__device__ half block_reduce_sum_f16_f16(half val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ half shared[NUM_WARPS];
    val = warp_reduce_sum_f16_f16<WARP_SIZE>(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();

    val = (lane < NUM_WARPS) ? shared[lane] : __float2half(0.0f);
    val = warp_reduce_sum_f16_f16<NUM_WARPS>(val);
    return val;
}