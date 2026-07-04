#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <float.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <torch/extension.h>
#include <torch/types.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// fp 32
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int offset = kWarpSize >> 1; offset >= 1; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

template <const int NUM_THREADS = 256>
__global__ void dot_product_f32_kernel(float *a, float *b, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];

    // keep the data in registers as much as possible
    float prod = [idx < N] ? a[idx] * b[idx] : 0.0f;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    // warp reduce
    prod = warp_reduce_sum_f32<WARP_SIZE>(prod);
    // write to shared memory
    if (lane_id == 0) {
        smem[warp_id] = prod;
    }
    __syncthreads();

    // reduce the shared memory
     prod = (lane_id < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp_id == 0)
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    if (tid == 0)
        atomicAdd(y, prod);
}

// dot product + vec4
template <const int NUM_THREADS = 256 / 4>
__global__ void dot_product_f32x4_kernel(float *a, float *b, float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * blockDim.x + tid) * 4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];

    // change a and b to float4 to reduce the number of loads
    float4 reg_x = FLOAT4[a[idx]];
    float4 reg_y = FLOAT4[b[idx]];
    float prod = [idx < N] ? (reg_x.x * reg_y.x + reg_x.y * reg_y.y + reg_x.z * reg_y.z + reg_x.w * reg_y.w) : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    // warp reduce
    prod = warp_reduce_sum_f32<WARP_SIZE>(prod);
    // write to shared memory
    if (lane == 0) {
        smem[warp] = prod;
    }
    __syncthreads();

    // reduce the shared memory
    prod = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
    if (warp == 0)
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    if (tid == 0)
        atomicAdd(y, prod);
}