#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <vector>
#include <algorithm>
#include <float.h>
#include <math.h>
#include <torch/types.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP 32
struct __align__(8) MD{
    float m;
    float d;
};
// warp reduce for online softmax
template <const int kwarpSize = WARP_SIZE>
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
    unsigned int mask = 0xffffffff;
#pragma unroll
    for (int stride = kwarpSize >> 1; stride >= 1; stride >>= 1) {
        MD other;
        other.m = __shfl_xor_sync(mask, value.m, stride);
        other.d = __shfl_xor_sync(mask, value.d, stride);

        bool value_bigger = (value.m > other.m);
        MD bigger_m = value_bigger ? value : other;
        MD smaller_m = value_bigger ? other : value;

        value.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
        value.m = bigger_m.m;
    }
    return value;
}

// warp reduce sum
template <const int kwarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = kwarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// warp reduce max
template <const int kwarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
#pragma unroll
    for (int mask = kwarpSize >> 1; mask >= 1; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// block reduction (sum)
template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];

    // warp reduce
    float value = warp_reduce_sum_f32<WARP_SIZE>(val);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();

    value = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
    value = warp_reduce_sum_f32<NUM_WARPS>(value);
    // thread 0  shares the value to all the threads within warp
    value = __shfl_sync(0xffffffff, value, 0, 32);
    return value;
}

// block reduction (max)
template <const int NUM_THREADS = 256>
__device__ float block_reduce_max_f32(float val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];

    float value = warp_reduce_max_f32<WARP_SIZE>(val);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();

    value = (lane < NUM_WARPS) ? shared[lane] : -FLT_MAX;
    value = warp_reduce_max_f32<NUM_WARPS>(value);
    value = __shfl_sync(0xffffffff, value, 0, 32);
    return value;
}

template <const int NUM_THREADS = 256>
__global__ void softmax_f32_per_token_kernel(float *x, float *y, int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;

    float exp_val = (idx < N) ? expf(x[idx]) : 0;
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);

    if (idx < N) {
        y[idx] = exp_val / exp_sum;
    }
}

template <const int NUM_THREADS = 256>
__global__ void softmax_f32x4_per_token_kernel(float *x, float *y, int N) {
    const int tid = threadIdx.x;
    const int idx = 4 * (blockIdx.x * blockDim.x + tid);
    
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_exp;
    reg_exp.x = (idx + 0 < N) ? expf(reg_x.x) : 0.0f;
    reg_exp.y = (idx + 1 < N) ? expf(reg_x.y) : 0.0f;
    reg_exp.w = (idx + 2 < N) ? expf(reg_x.w) : 0.0f;
    reg_exp.z = (idx + 3 < N) ? expf(reg_x.z) : 0.0f;

    float exp_val = (reg_exp.x + reg_exp.y + reg_exp.w + reg_exp.z);
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);

    if (idx + 3 < N) {
        float4 reg_y;
        reg_y.x = reg_exp.x / (exp_sum);
        reg_y.y = reg_exp.y / (exp_sum);
        reg_y.w = reg_exp.w / (exp_sum);
        reg_y.z = reg_exp.z / (exp_sum);
        FLOAT4(y[idx]) = reg_y;
    }
}