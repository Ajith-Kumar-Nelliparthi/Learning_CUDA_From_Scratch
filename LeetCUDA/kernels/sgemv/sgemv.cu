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

// Y = α*(A[MK] * X[K]) + β.y
__global__ void sgemv_k32_f32_kernel(float *x, float *a, float *y, int M, int K) {
    int tx = threadIdx.x;       // 0 - 31
    int ty = threadIdx.y;       // 0 - 4
    int bid = blockIdx.x;        // 0 - M/4
}