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

// define slices of BM,BN,BK
template <const int BM=32, const int BN=32, const int BK=32>
__global__ void sgemm_slice_k_f32_kernel(float *a, float *b, float *c, int M, int N, int K) {
    __shared__ float s_a[BM][BK], s_b[BK][BN];

    // define thread & block indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    // tid within the block upto 1024 (31 * 32 + 31 = 1023)
    int tid = ty * blockDim.y + tx; 
    // shared memory indices (to write values)
    int load_smem_a_m = tid / 32;
    int load_smem_a_k = tid % 32;
    int load_smem_b_k = tid / 32;
    int load_smem_b_n = tid % 32;
    // global memory indices (read values from A and B)
    int load_gmem_a_m = by * BM + load_smem_a_m;     // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n;     // global col of a and c

    float sum = 0.0f;
    for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;                        // calculate k index in a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;           // calculate linear index address of A in 1D      (r * col + c)
        s_a[load_smem_a_m][load_smem_a_k] = load_gmem_a_addr;

        int load_gmem_b_k = bk * BK + load_smem_b_k;                        // calculate k index in b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;           // calculate linear index address of B in 1D
        s_b[load_smem_b_k][load_smem_b_n] = load_gmem_b_addr;
        __syncthreads();
        
#pragma unroll
        for (int k=0; k<BK; ++k) {
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

template <const int BM = 128, const int BN = 128, const int BK = 8,
    const int TM = 8, const int TN = 8>
__global__ void sgemm_t_8x8_sliced_k_f32x4_kernel(float *a, float *b, float *c, int M, int N, int K) {
    __shared__ float s_a[BM][BK], s_b[BK][BM];    // initalize shared memory of 128x8 and 8x128 tiles

    // indexing
    const int tx = threadIdx.x;   // 0 to 127
    const int ty = threadIdx.y;   // 0 to 7
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    int tid = ty * blockDim.x + tx; // thread within the block

    int load_smem_a_m = tid / 2; // (128 / 8) * (128 / 8) = 256 threads per block
                                // tid / 2 = 256 / 2 = 128 -> (0-127)
    int load_smem_a_k = (tid % 2 == 0) ? 0:4; // a_k loads or one thread reads 0-4 & the other reads 4-7 indices at once 
    int load_smem_b_k = tid / 32;                   // 0-7 threads in y-dir
    int load_smem_b_n =  (tid % 32) * 4;           // each thread loads 4 elements in x-dir
    // 1D flat index calculation m and n
    int load_gmem_a_m = by * BM + load_smem_a_m;    // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n;    // global col of b and c

    float r_c[TM][TN] - {0.0};
// #pragma unroll
    for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;   // global memory address
        FLOAT4(s_a[load_gmem_a_m][load_gmem_a_k]) = FLOAT4(a[load_gmem_a_addr]);

        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;   // global memory address
        FLOAT4(s_b[load_gmem_b_k][load_gmem_b_n]) = FLOAT4(b[load_gmem_b_addr]);
        __syncthreads();

#pragma unroll
        for (int k=0; k<BK; ++k) {
#pragma unroll
            for (int m=0; m<TM; ++m) {
#pragma unroll
                for (int n=0; n<TN; ++n) {
                    int comp_smem_a_m = ty * TM + m;
                    int comp_smem_b_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int m=0; m<TM; ++m) {
        int store_gmem_c_m = by * BM + ty * TM + m;
#pragma unroll
        for (int n=0; n<TN; ++n) {
            int store_gmem_c_n = bx * BN + tx * TN + n;
            int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;    // global memory address
            FLOAT4(c[store_gmem_c_addr]) = FLOAT4(r_c[m][n]);
        }
    }
}
