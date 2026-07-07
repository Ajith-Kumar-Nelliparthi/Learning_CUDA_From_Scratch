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