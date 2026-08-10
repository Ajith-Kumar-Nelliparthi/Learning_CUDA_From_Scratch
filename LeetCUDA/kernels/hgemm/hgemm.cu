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
__global__ void hgemm_naive_f16_kernel(half *a, half *b, half *c, int M, int N, int K) {
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

template <const int BM=128, const int BN=128, const int BK=8,
          const int TM=8, const int TN=8>
__global__ void hgemm_t_8x8_sliced_k_f16x4_kernel(half *a, half *b, half *c, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx; // thread id with in the block
    __shared__ half s_a[BM][BK], s_b[BK][BN];

    int load_smem_a_m = tid / 2; // 0~127
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 4; // 0,4
    int load_smem_b_k = tid / 32; // 0~7
    int load_smem_b_n = (tid % 32) * 4; // 0,4,8,...,124
    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;

    if (load_gmem_a_m >= M || load_gmem_b_n >= N) {
        return;
    }
    half r_c[TM][TN] = {__float2half(0.0f)};
    
    for (int bk=0; bk < (K + BK - 1) / BK; bk++) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        HALF2(s_a[load_smem_a_m][load_smem_a_k + 0]) = HALF2(a[load_gmem_a_addr + 0]);
        HALF2(s_a[load_smem_a_m][load_smem_a_k + 2]) = HALF2(a[load_gmem_a_addr + 2]);
        HALF2(s_b[load_smem_b_k][load_smem_b_n + 0]) = HALF2(b[load_gmem_b_addr + 0]);
        HALF2(s_b[load_smem_b_k][load_smem_b_n + 2]) = HALF2(b[load_gmem_b_addr + 2]);
        __syncthreads();

    #pragma unroll
        for (int k=0; k<BK; k++) {
    #pragma unroll
            for (int m=0; m<TM; m++) {
    #pragma unroll
                for (int n=0; n<TN; n++) {
                    int comp_smem_a_m = ty * TM + m;
                    int comp_smem_b_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int m=0; m<TM; m++) {
        int store_gmem_c_m = by * BM + ty * TM + m;
#pragma unroll
        for (int n=0; n<TN; n++) {
            int store_gmem_c_n = bx * BN + tx * TN + n;
            int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
            c[store_gmem_c_addr] = r_c[m][n];
        }
    }
}

template <const int BM=128, const int BN=128, const int BK=8,
            const int TM=8, const int TN=8>
__global__ void hgemm_t_8x8_sliced_k_f16x4_pack_kernel(half *a, half *b, half *c, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y; 
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx; // thread id with in the block
    __shared__ half s_a[BM][BK], s_b[BK][BN];

    int load_smem_a_m = tid / 2; // 0~127
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 4; // 0,4
    int load_smem_b_k = tid / 32; // 0~7
    int load_smem_b_n = (tid % 32) * 4; // 0,4,8,...,124
    half r_c[TM][TN] = {__float2half(0.0f)};

    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;

    if (load_gmem_a_m >= M || load_gmem_b_n >= N) {
        return;
    }

    for (int bk=0; bk < (K + BK - 1) / BK; bk++) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        LDST64BITS(s_a[load_smem_a_m][load_smem_a_k]) = LDST64BITS(a[load_gmem_a_addr]);
        LDST64BITS(s_b[load_smem_b_k][load_smem_b_n]) = LDST64BITS(b[load_gmem_b_addr]);
        __syncthreads();

    #pragma unroll
        for (int k=0; k<BK; k++) {
    #pragma unroll
            for (int m=0; m<TM; m++) {
    #pragma unroll
                for (int n=0; n<TN; n++) {
                    int comp_smem_a_m = ty * TM + m;
                    int comp_smem_b_n = tx * TN + n;
                    r_c[m][n] += __hfma(s_a[comp_smem_a_m][k] , s_b[k][comp_smem_b_n]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int m=0; m<TM; m++) {
        int store_gmem_c_m = by * BM + ty * TM + m;
#pragma unroll
        for (int n=0; n<TN; n++) {
            int store_gmem_c_n = bx * BN + tx * TN + n;
            int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
            LDST64BITS(c[store_gmem_c_addr]) = LDST64BITS(r_c[m][n]);
        }
    }
}

template <const int BM=128, const int BN=128, const int BK=8,
            const int TM=8, const int TN=8>
__global__ void hgemm_t_8x8_sliced_k_f16x4_bcf_kernel(half *a, half *b, half *c, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    __shared__ half s_a[BK][BM], s_b[BK][BN];

    half r_load_a[TM / 2];
    half r_load_b[TN / 2];
    half r_comp_a[TM];
    half r_comp_b[TN];
    half r_load_c[TM][TN] = {__float2half(0.0f)};

    int load_smem_a_m = tid / 2; // 0,1,2,..127
    int load_smem_a_k = (tid & 1) << 2; // 0 or 4
    int load_smem_b_k = tid / 32;
    int load_smem_b_n = (tid & 31) << 2; 

    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;

    if (load_gmem_a_m >= M || load_gmem_b_n >= N) {
        return;
    }

    for (int bk=0; bk<(K + BK - 1) / BK; bk++) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        // load data from GBM to Registers
        HALF2(r_load_a[0]) = HALF2(a[load_gmem_a_addr + 0]);
        HALF2(r_load_a[2]) = HALF2(a[load_gmem_a_addr + 2]);
        HALF2(r_load_b[0]) = HALF2(b[load_gmem_b_addr + 0]);
        HALF2(r_load_b[2]) = HALF2(b[load_gmem_b_addr + 2]);

        // registers to SM
        s_a[load_smem_a_k + 0][load_smem_a_m] = r_load_a[0];
        s_a[load_smem_a_k + 1][load_smem_a_m] = r_load_a[1];
        s_a[load_smem_a_k + 2][load_smem_a_m] = r_load_a[2];
        s_a[load_smem_a_k + 3][load_smem_a_m] = r_load_a[3];
        HALF2(s_b[load_smem_b_k][load_smem_b_n + 0]) = HALF2(r_load_b[2]);
        HALF2(s_b[load_smem_b_k][load_smem_b_n + 2]) = HALF2(r_load_b[2]);

        __syncthreads();

#pragma unroll
        for (int tk=0; tk < BK; tk++) {
            HALF2(r_comp_a[0]) = HALF2(s_a[tk][ty * TM / 2]);
            HALF2(r_comp_a[2]) = HALF2(s_a[tk][ty * TM / 2 + 2]);
            HALF2(r_comp_a[4]) = HALF2(s_a[tk][ty * TM / 2 + BM / 2]);
            HALF2(r_comp_a[6]) = HALF2(s_a[tk][ty * TM / 2 + BM / 2 + 2]);

            HALF2(r_comp_b[0]) = HALF2(s_b[tk][tx * TN / 2]);
            HALF2(r_comp_b[2]) = HALF2(s_b[tk][tx * TN / 2 + 2]);
            HALF2(r_comp_b[4]) = HALF2(s_b[tk][tx * TN / 2 + BN / 2]);
            HALF2(r_comp_b[6]) = HALF2(s_b[tk][tx * TN / 2 + BN / 2 + 2]);
#pragma unroll
            for (int tm=0; tm<TM; tm++) {
#pragma unroll
                for (int tn=0; tn<TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i=0; i < TM / 2; i++) {
        int store_gmem_c_m = by * BM + ty * TM / 2 + i;
        int store_gmem_c_n = bx * BN + tx * TN / 2;
        int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
        HALF2(c[store_c_gmem_addr + 0]) = HALF2(r_c[i][0]);
        HALF2(c[store_c_gmem_addr + 2]) = HALF2(r_c[i][2]);
        HALF2(c[store_c_gmem_addr + BN / 2 + 0]) = HALF2(r_c[i][4]);
        HALF2(c[store_c_gmem_addr + BN / 2 + 2]) = HALF2(r_c[i][6]);
    }
#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        HALF2(c[store_c_gmem_addr + 0]) = HALF2(r_c[i + TM / 2][0]);
        HALF2(c[store_c_gmem_addr + 2]) = HALF2(r_c[i + TM / 2][2]);
        HALF2(c[store_c_gmem_addr + BN / 2 + 0]) = HALF2(r_c[i + TM / 2][4]);
        HALF2(c[store_c_gmem_addr + BN / 2 + 2]) = HALF2(r_c[i + TM / 2][6]);
    }
}

template <const int BM=128, const int BN=128, const int BK=8,
    const int TM=8, const int TN=8>
__global__ void hgemm_t_8x8_sliced_k_f16x4_pack_bcf_kernel(half *a, half *b, half *c, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    __shared__ half s_a[BK][BM], s_b[BK][BN];

    half r_load_a[TM / 2];
    half r_load_b[TN / 2];
    half r_comp_a[TM];
    half r_comp_b[TN];
    half r_c[TM][TN] = {__float2half(0.0f)};

    int load_smem_a_m = tid / 2;
    int load_smem_a_k = (tid & 1) << 2;
    int load_smem_b_k = tid / 32;
    int load_smem_b_n = (tid & 31) << 2; // 0,4,8,,,,124
    
    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;

    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    for (int bk=0; bk< (K + BK - 1) / BK; bk++) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        // read data from HBM to Registers
        LDST64BITS(r_load_a[0]) = LDST64BITS(a[load_gmem_a_addr]);
        LDST64BITS(r_load_b[0]) = LDST64BITS(b[load_gmem_b_addr]);

        // read data from reg to SM
        s_a[load_smem_a_k][load_smem_a_m] = r_load_a[0];
        s_a[load_smem_a_k + 1][load_smem_a_m] = r_load_a[1];
        s_a[load_smem_a_k + 2][load_smem_a_m] = r_load_a[2];
        s_a[load_smem_a_k + 3][load_smem_a_m] = r_load_a[3];

        LDST64BITS(s_b[load_smem_b_k][load_smem_b_n]) = LDST64BITS(r_load_b[0]);
        __syncthreads();

#pragma unroll
        for (int tk=0; tk < BK; tk++) {
            LDST64BITS(r_comp_a[0]) = LDST64BITS(s_a[tk][ty * TM / 2]);
            LDST64BITS(r_comp_a[4]) = LDST64BITS(s_a[tk][ty * TM / 2 + BM / 2]);

            LDST64BITS(r_comp_b[0]) = LDST64BITS(s_b[tk][tx * TN / 2]);
            LDST64BITS(r_comp_b[4]) = LDST64BITS(s_b[tk][tx * TN / 2 + BN / 2]);
#pragma unroll
            for (int tm=0; tm < TM; tm++) {
#pragma unroll
                for (int tn=0; tn < TN; tn++) {
                    r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i=0; i < TM / 2; i++) {
        int store_gmem_c_m = by * BM + ty * TM / 2 + i;
        int store_gmem_c_n = bx * BN + tx * TN / 2;
        int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
        LDST64BITS(c[store_c_gmem_addr]) = LDST64BITS(r_c[i][0]);
        LDST64BITS(c[store_c_gmem_addr + BN / 2]) = LDST64BITS(r_c[i][4]);
    }
#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        LDST64BITS(c[store_c_gmem_addr]) = LDST64BITS(r_c[i + TM / 2][0]);
        LDST64BITS(c[store_c_gmem_addr + BN / 2]) = LDST64BITS(r_c[i + TM / 2][4]);
    }
}

template <const int BM=128, const int BN=128, const int BK=8,
            const int TM=8, const int TN=8, const int offset = 0>
__global__ void hgemm_t_8x8_sliced_k_f16x8_pack_bcf_kernel(half *a, half *b, hlaf *c, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    __shared__ half s_a[BK][BM + offset], s_b[BK][BN + offset];

    half r_load_a[TM / 2];
    half r_load_b[TN / 2];
    half r_comp_a[TM];
    half r_comp_b[TN];
    half r_c[TM][TN] = {__float2half(0.0f)};

    int load_smem_a_m = tid / 2;
    int load_smem_a_k = (tid & 1) << 2;
    int load_smem_b_k = tid / 32;
    int load_smem_b_n = (tid & 31) << 2;

    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;

    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    for (int bk=0; bk < (K + BK - 1) / BK; bk++) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        // read data from HBM to reg
        LDST64BITS(r_load_a[0]) = LDST64BITS(a[load_gmem_a_addr]);
        LDST64BITS(r_load_b[0]) = LDST64BITS(b[load_gmem_b_addr]);

        // write data to SM
        s_a[load_smem_a_k][load_smem_a_m] = r_load_a[0];
        s_a[load_smem_a_k + 1][load_smem_a_m] = r_load_a[1];
        s_a[load_smem_a_k + 2][load_smem_a_m] = r_load_a[2];
        s_a[load_smem_a_k + 3][load_smem_a_m] = r_load_a[3];

        LDST64BITS(s_b[load_smem_b_k][load_smem_b_n]) = LDST64BITS(r_load_b[0]);
        __syncthreads();

#pragma unroll
        for (int tk=0; tk < BK; tk++) {
            LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[tk][ty * TM]);
            LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[tk][tx * TN]);
#pragma unroll
            for (int tm=0; tm<TM; tm++) {
#pragma unroll
                for (int tn=0; tn<TN; tn++) {
                    r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i=0; i<TM; i++) {
        int store_gmem_c_m = by * BM + ty * TM + i;
        int store_gmem_c_n = bx * BN + ty * TN;
        int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
        LDST128BITS(c[store_gmem_c_addr]) = LDST128BITS(r_c[i][0]);
    }
}

template <const int BM=128, const int BN=128, const int BK=8,
        const int TM=8, const int TN=8, const int OFFSET=0>
__global__ void hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf_kernel(half *a, half *b, half *c, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    __shared__ half s_a[2][BK][BM + OFFSET];
    __shared__ half s_b[2][BK][BN + OFFSET];

    half r_load_a[TM / 2];
    half r_load_b[TN / 2];
    half r_comp_a[TM];
    half r_comp_b[TN];
    half r_c[TM][TN] = {__float2half(0.0f)};

    int load_smem_a_m = tid / 2;
    int load_smem_a_k = (tid & 1) << 2;
    int load_smem_b_k = tid / 32;
    int load_smem_b_n = (tid & 31) << 2;

    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) {
        return;
    }

    // load buffer 0 explicit
    {
        int load_gmem_a_k = load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        // read data from HBM to Registers
        LDST64BITS(r_load_a[0]) = LDST64BITS(a[load_gmem_a_addr]);
        LDST64BITS(r_load_b[0]) = LDST64BITS(b[load_gmem_b_addr]);

        // read to SM
        s_a[0][load_smem_a_k + 0][load_smem_a_m] = r_load_a[0];
        s_a[0][load_smem_a_k + 1][load_smem_a_m] = r_load_a[1];
        s_a[0][load_smem_a_k + 2][load_smem_a_m] = r_load_a[2];
        s_a[0][load_smem_a_k + 3][load_smem_a_m] = r_load_a[3];
        LDST64BITS(s_b[0][load_smem_b_k][load_smem_b_n]) = LDST64BITS(r_load_b[0]);
    }
    __syncthreads();

    for (int bk=1; bk < (K + BK - 1) / BK; bk++) {
        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        LDST64BITS(r_load_a[0]) = LDST64BITS(a[load_gmem_a_addr]);
        LDST64BITS(r_load_b[0]) = LDST64BITS(b[load_gmem_b_addr]);

#pragma unroll
        for (int tk=0; tk<BK; tk++) {
            LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[smem_sel][tk][ty * TM]);
            LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[smem_sel][tk][tx * TN]);
#pragma unroll
            for (int tm=0; tm<TM; tm++) {
#pragma unroll
                for (int tn=0; tn<TN; tn++) {
                    r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }
        s_a[smem_sel_next][load_smem_a_k + 0][load_smem_a_m] = r_load_a[0];
        s_a[smem_sel_next][load_smem_a_k + 1][load_smem_a_m] = r_load_a[1];
        s_a[smem_sel_next][load_smem_a_m + 2][load_smem_a_m] = r_load_a[2];
        s_a[smem_sel_next][load_smem_a_k + 3][load_smem_a_m] = r_load_a[3];
        LDST64BITS(s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]) = LDST64BITS(r_load_b[0]);
        __syncthreads();
    }
    {
        int smem_sel_last = ((K + BK - 1) / BK - 1) & 1;
    #pragma unroll
        for (int tk=0; tk<BK; tk++) {
            LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[smem_sel_last][tk][ty * TM]);
            LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[smem_sel_last][tk][tx * TN]);
    #pragma unroll
            for (int tm=0; tm<TM; tm++) {
    #pragma unroll
                for (int tn=0; tn<TN; tn++) {
                    r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }
    }
#pragma unroll
    for (int i=0; i<TM; i++) {
        int store_gmem_c_m = by * BM + ty * TM + i;
        int store_gmem_c_n = bx * BN + tx * TN;
        int store_gmem_c_adr = store_gmem_c_m * N + store_gmem_c_n;
        LDST128BITS(c[store_gmem_c_adr]) = LDST128BITS(r_c[i][0]);
    }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

// HGEMM naive: compute one c[i,j] element per threads, all row major
void hgemm_naive_f16(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 32;
  constexpr int BN = 32;

  dim3 block(BN, BM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_naive_f16_kernel<<<grid, block>>>(
      reinterpret_cast<half *>(a.data_ptr()),
      reinterpret_cast<half *>(b.data_ptr()),
      reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}

void hgemm_sliced_k_f16(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;

  dim3 block(BN, BM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_sliced_k_f16_kernel<BM, BN, BK>
      <<<grid, block>>>(reinterpret_cast<half *>(a.data_ptr()),
                        reinterpret_cast<half *>(b.data_ptr()),
                        reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}

// t 8x8 fp16x4
void hgemm_t_8x8_sliced_k_f16x4(torch::Tensor a, torch::Tensor b,
                                torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_t_8x8_sliced_k_f16x4_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(reinterpret_cast<half *>(a.data_ptr()),
                        reinterpret_cast<half *>(b.data_ptr()),
                        reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}

// t 8x8 fp16x4 pack
void hgemm_t_8x8_sliced_k_f16x4_pack(torch::Tensor a, torch::Tensor b,
                                     torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_t_8x8_sliced_k_f16x4_pack_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(reinterpret_cast<half *>(a.data_ptr()),
                        reinterpret_cast<half *>(b.data_ptr()),
                        reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}

// reduce bank conflicts
void hgemm_t_8x8_sliced_k_f16x4_bcf(torch::Tensor a, torch::Tensor b,
                                    torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_t_8x8_sliced_k_f16x4_bcf_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(reinterpret_cast<half *>(a.data_ptr()),
                        reinterpret_cast<half *>(b.data_ptr()),
                        reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}

// reduce bank conflicts, f16x4 pack, t 8x8
void hgemm_t_8x8_sliced_k_f16x4_pack_bcf(torch::Tensor a, torch::Tensor b,
                                         torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_t_8x8_sliced_k_f16x4_pack_bcf_kernel<BM, BN, BK, TM, TN, 4>
      <<<grid, block>>>(reinterpret_cast<half *>(a.data_ptr()),
                        reinterpret_cast<half *>(b.data_ptr()),
                        reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}

// reduce bank conflicts, t 8x8 fp16x8 pack, pad
void hgemm_t_8x8_sliced_k_f16x8_pack_bcf(torch::Tensor a, torch::Tensor b,
                                         torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_t_8x8_sliced_k_f16x8_pack_bcf_kernel<BM, BN, BK, TM, TN, 8>
      <<<grid, block>>>(reinterpret_cast<half *>(a.data_ptr()),
                        reinterpret_cast<half *>(b.data_ptr()),
                        reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}

void hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf(torch::Tensor a, torch::Tensor b,
                                              torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf_kernel<BM, BN, BK, TM, TN, 8>
      <<<grid, block>>>(reinterpret_cast<half *>(a.data_ptr()),
                        reinterpret_cast<half *>(b.data_ptr()),
                        reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}