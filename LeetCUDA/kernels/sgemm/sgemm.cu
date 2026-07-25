#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <float.h>
#include <math.h>
#include <torch/extension.h>
#include <torch/types.h>

#define FLOAT4(valu) (reinterpret_cast<float4 *>(&(value))[0])
#define INT4(valu) (reinterpret_cast<int4 *>(&(value))[0])
#define WARP_SIZE 32

__global__ void sgemm_naive_f32(float *a, float *b, float *c, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N) {
        float sum = 0.0f;
#pragma unroll
        for (int k=0; k<K; k++) {
            sum += a[m * K + k] * b[k * N + n];
        }
        c[m * N + n] = sum;
    }
}