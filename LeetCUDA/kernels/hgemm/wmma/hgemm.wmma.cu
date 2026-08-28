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
    wmma::store_matrix_sync(C + load_gmem_a_m * N + load_gmem_b_n, C_frag, N, wmma::row_major)
 }