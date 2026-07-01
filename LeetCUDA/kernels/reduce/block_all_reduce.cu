#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <torch/types.h>
#include <torch/extension.h>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP 32
// Warp Reduce Sum
template <const int kwarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int offset = kwarpSize >> 1; offset >= 1; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// block all reduce sum
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    float sum = (idx < N) ? a[idx] : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (lane == 0) {
        reduce_smem[warp] = sum;
    }
    __syncthreads();

    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0) {
        sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    }
    if (tid == 0)
        atomicAdd(y, sum);
}

// block all reduce sum + float4
template <const int NUM_THREADS = 256 / 4>
__global__ void block_all_reduce_sum_f32x4_f32_kernel(float *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = 4 * (blockIdx.x * NUM_THREADS + tid);
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    float4 reg_a = FLOAT4(a[idx]);
    float sum = (idx < N) ? (reg_a.x + reg_a.y + reg_a.z + reg_a.w) : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    // perform warp reduce sum
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    // write warp reduce sum to shared memory
    if (lane == 0) {
        reduce_smem[warp] = sum;
    }
    __syncthreads();
    // perform block reduce sum
    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0) {
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    }
    if (tid == 0)
        atomicAdd(y, sum);
}