#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <vector>
#include <cuda_bf16.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <torch/types.h>
using namespace nvcuda;

#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)        \
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_CA(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))

HOST_DEVICE_INLINE
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

 // only one warp per block (32 threads)
 template <const int WMMA_M=16, const int WMMA_N=16, const int WMMA_K=16>
 __global__ void hgemm_wmma_m16n16k16_naive_kernel(half *A, half *B, half *C, int M, int N, int K) {
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    const int load_gmem_a_m = blockIdx.y * WMMA_M;
    const int load_gmem_b_n = blockIdx.x * WMMA_N;
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    // load data / tile from HBM / GBM
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);

#pragma unroll
    for (int k=0; k<NUM_K_TILES; k++) {
        // load tiles into fragments
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
        A_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
        B_frag;

        wmma::load_matrix_sync(A_frag, A + load_gmem_a_m * K + k * WMMA_K, K);
        wmma::load_matrix_sync(B_frag, B + (k * WMMA_K) + N + load_gmem_b_n, N);

        // perform MMA (multiply - accumulate)
        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        __syncthreads();
    }
    // store results
    wmma::store_matrix_sync(C + load_gmem_a_m * N + load_gmem_b_n, C_frag, N, wmma::mem_row_major)
 }

 template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16, const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2>
 __global__ void hgemm_wmma_m16n16k16_mma4x2_kernel(half *A, half *B, half *C, int M, int N, int K) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y; // block id's in x dir
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M; // 16 x 4 = 64
    constexpr int BN = WMMA_N * WMMA_TILE_N; // 16 x 2 = 32
    constexpr int BK = WMMA_K;
    __shared__ half s_a[BM][WMMA_K], s_b[WMMA_K][BN];     // 64x16x2bytes=2kb 16x32x2bytes=1kb

    // 1D thread index
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;    // 0 to 7
    const int lane_id = tid % WARP_SIZE;   // 0 to 31
    const int warp_m = warp_id / 2;     // 0,1,2,3
    const int warp_n = warp_id % 2;     // 0,1

    // shared memory index calculation
    int load_smem_a_m = tid / 4;   //0-63
    int load_smem_a_k = (tid % 4) * 4;  // 0,4,8,..
    int load_smem_b_k = tid / 16;
    int load_smem_b_n = (tid % 16) * 2;     // 0,2,4,,32
    int load_gmem_a_m = by * BM + load_smem_a_m;    // global m
    int load_gmem_b_n = bx * BN + load_smem_b_n;    // global n
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);

#pragma unroll
    for (int k=0; k<NUM_K_TILES; k++) {
        int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        // 64 bits sync memory - gbm to smem
        LDST64BITS(s_a[load_smem_a_m][load_smem_a_k]) = LDST64BITS(A[load_gmem_a_addr]);
        // 32 bits sync memory
        LDST32BITS(s_b[load_smem_b_k][load_smem_b_n]) = LDST32BITS(B[load_gmem_b_addr]);
        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
        A_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
        B_frag;

        wmma::load_matrix_sync(A_frag, &s_a[warp_m * WMMA_M][0], BK);
        wmma::load_matrix_sync(B_frag, &s_b[0][warp_n * WMMA_N], BN);
        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        __syncthreads();
    }
    // store results
    const int store_gmem_a_m = by * BM + warp_m * WMMA_M;
    const int store_gmem_b_n = bx * BN + warp_n * WMMA_N;
    wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_b_n, C_frag, N, wmma::mem_row_major);
 }

 template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
            const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2, 
            const int WARP_TILE_M = 2, const int WARP_TILE_N = 4>
__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel(half *A, half *B, half *C, int M, int N, int K) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;  // 16x4x2 = 128
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;  // 16*2*4 = 128
    constexpr BK = WMMA_K;
    __shared__ half s_a[BM][BK], s_b[BK][BN];   // 16x128x2bytes = 4KB

    // index cal
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;        // 0 to 7
    const int lane_id = tid % WARP_SIZE;        // 0 to 31
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;

    // shared mem index cal
    int load_smem_a_m = tid / 2;    // row 0 to 127
    // 1 thread reads first 8 elements and the other half reads the left total covering 16 values in a row
    // 128 rows x 2 threads per row = 256 (total threads)
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0-7 , 8-15
    int load_smem_b_k = tid / 16;   // row 0 to 15
    int load_smem_b_n = (tid % 16) * 8;

    // global index
    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    // load into fragment
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>C_frag[WARP_TILE_M][WARP_TILE_N];
#pragma unroll
    for (int i=0; i< WARP_TILE_M; i++) {
        for (int j=0; j< WARP_TILE_N; j++) {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

#pragma unroll
    for (int k=0; k<NUM_K_TILES; k++) {
        int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        LDST128BITS(s_b[load_smem_b_k][load_smem_b_n]) = LDST128BITS(B[load_gmem_b_addr]);
        LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) = LDST128BITS(A[load_gmem_a_addr]);
        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>B_frag[WARP_TILE_N];

    #pragma unroll
        for (int i=0; i<WARP_TILE_M; i++) {
            // load 2 tiles -> reg, smem a -> frags a
            // each warp loads 2 16x16 tiles of A from SMEM
            const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            wmma::load_matrix_sync(A_frag[i], &s_a[warp_smem_a_m][0], BK);
        }
    #pragma unroll
        for (int j=0; j < WARP_TILE_N; j++) {
            // load 4 tiles -> reg, smem b -> frags B
            const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            wmma::load_matrix_sync(B_frag[j], &s_b[0][warp_smem_b_n], BN);
        }

    #pragma unroll
        for (int i=0; i<WARP_TILE_M; i++) {
    #pragma unroll
            for (int j=0; j<WARP_TILE_N; j++) {
                wmma::mma_sync(C_frag[i][j], A_frag[i][j], B_frag[i][j], C_frag[i][j]);
            }
        }
        __syncthreads();
    }

    // store results
#pragma unroll
    for (int i=0; i<WARP_TILE_M; i++) {
#pragma unroll
        for (int j=0; j<WARP_TILE_N; j++) {
            const int store_gmem_a_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            const int store_gmem_b_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_b_n, C_frag[i][j], N, wmma::mem_row_major);
        }
    }
}
