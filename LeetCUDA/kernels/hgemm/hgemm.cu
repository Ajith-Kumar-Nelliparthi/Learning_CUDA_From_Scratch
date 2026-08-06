#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <float.h>
#include <cuda_fp16.h>
#include <math.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <vector>
#include <torch/types.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// fp 16
// hgemm naive kernel
__global__ void hgemm_naive_kernel(half *a, half *b, half *c, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M && n < N) {
        half sum = 0.0f;
#pragma unroll
        for (int k=0; k<K; k++) {
            sum += a[m * K + k] * b[k * N + n];
        }
        c[m * N + n] = sum;
    }
}

template <const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void hgemm_sliced_k_f16_kernel(half *a, half *b, half *c, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx; // thread id in block

    __shared__ half s_a[BM][BK], s_b[BK][BN];

    int load_smem_a_m = tid / 32;
    int load_smem_a_k = tid % 32;
    int load_smem_b_k = tid / 32;
    int load_smem_b_n = tid % 32;
    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;

    if (load_gmem_a_m >= M || load_gmem_b_n >= N) {
        return;
    }

    half sum = __float2half(0.0f);
    for (int bk=0; bk<BK; bk++) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_addr];
        s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_addr];
        __syncthreads();

    #pragma unroll
        for (int k=0; k<BK; k++) {
            int comp_smem_a_m = load_smem_a_m;
            int comp_smem_b_n = load_smem_b_n;
            sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
        __syncthreads();
    }
    int store_gmem_c_m = load_gmem_a_m;
    int store_gmem_c_n = load_gmem_b_n;
    int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
    c[store_gmem_c_addr] = sum;
}