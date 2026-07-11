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
#define epsilion 1e-5f

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