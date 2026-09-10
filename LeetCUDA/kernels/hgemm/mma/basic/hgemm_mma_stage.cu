// load header files
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <float.h>
#include <cuda_bf16.h>
#include <vector>
#include <algorithm>
#include <math.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
using namespace nvcuda;

// Common vector types and inline PTX helpers used by the tensor-core kernel.
#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
// Commit and wait for asynchronous global-to-shared-memory copies.
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) \
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_CA(dst, src, bytes) \
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
// smem -> gmem: requires sm_90 or higher.
#define CP_ASYNC_BULK_COMMIT_GROUP() asm volatile("cp.async.bulk.commit_group;\n" ::)
#define CP_ASYNC_BULK_WAIT_ALL() asm volatile("cp.async.bulk.wait_all;\n" ::)
#define CP_ASYNC_BULK_WAIT_GROUP(n)                                            \
  asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_BULK(dst, src, bytes)                                         \
  asm volatile(                                                                \
      "cp.async.bulk.global.shared::cta.bulk_group.L2::128B [%0], [%1], "      \
      "%2;\n" ::"r"(dst),                                                      \
      "l"(src), "n"(bytes))
// ldmatrix
#define LDMATRIX_X1(R, addr)                                                   \
  asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"        \
               : "=r"(R)                                                       \
               : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr)                                              \
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"    \
               : "=r"(R0), "=r"(R1)                                            \
               : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                      \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"     \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))
#define LDMATRIX_X1_T(R, addr)                                                 \
  asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"  \
               : "=r"(R)                                                       \
               : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr)                                            \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"       \
      : "=r"(R0), "=r"(R1)                                                     \
      : "r"(addr))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr)                                    \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, "      \
      "[%4];\n"                                                                \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))
// stmatrix: requires sm_90 or higher.
#define STMATRIX_X1(addr, R)                                                   \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n" ::"r"(addr),    \
      "r"(R))
#define STMATRIX_X2(addr, R0, R1)                                              \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n" ::"r"(      \
          addr),                                                               \
      "r"(R0), "r"(R1))
#define STMATRIX_X4(addr, R0, R1, R2, R3)                                      \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::  \
          "r"(addr),                                                           \
      "r"(R0), "r"(R1), "r"(R2), "r"(R3))
#define STMATRIX_X1_T(addr, R)                                                 \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x1.trans.m8n8.shared.b16 [%0], {%1};\n" ::"r"(    \
          addr),                                                               \
      "r"(R))
#define STMATRIX_X2_T(addr, R0, R1)                                            \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [%0], {%1, %2};\n" ::    \
          "r"(addr),                                                           \
      "r"(R0), "r"(R1))
#define STMATRIX_X4_T(addr, R0, R1, R2, R3)                                    \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, "     \
      "%4};\n" ::"r"(addr),                                                    \
      "r"(R0), "r"(R1), "r"(R2), "r"(R3))
// mma m16n8k16
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)            \
  asm volatile(                                                                \
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, "  \
      "%4, %5}, {%6, %7}, {%8, %9};\n"                                         \
      : "=r"(RD0), "=r"(RD1)                                                   \
      : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0),  \
        "r"(RC1))
HOST_DEVICE_INLINE
int div_ceil(int a, int b) {return (a % b != 0) ? (a / b + 1) : (a / b);}

// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle
template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
          const int MMA_TILE_M = 2, const int MMA_TILE_N = 4,
          const int WARP_TILE_M = 4, const int WARP_TILE_N = 4,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 2,
          bool BLOCK_SWIZZLE = false>
__global__ void __launch_bounds__(256)
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_kernel(half *A, half *B, half *C, int M, int N, int K) {
  const int bx = ((int)BLOCK_SWIZZLE) * blockDim.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;  //16x2x4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;  //8x4x4=128
  constexpr int BK = MMA_K;

  __shared__ half s_a[K_STAGE][BM][BK + A_PAD];
  __shared__ half s_b[K_STAGE][BK][BN + B_PAD];
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);
  constexpr int s_b_stage_offset = BK * (BN + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int warp_m = warp_id % 2;   //0,1
  const int warp_n = warp_id / 2;

  int load_smem_a_m = tid / 2;
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;
  int load_smem_b_k = tid / 16;
  int load_smem_b_n = (tid % 16) * 8;
  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;
  if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
    for (int j = 0 ; j < WARP_TILE_N; j++) {
      RC[i][j][0] = 0;
      RC[i][j][1] = 0;
    }
  }
   
  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

#pragma unroll
  for (int k = 0; k < (K_STAGE - 1); k++) {
    int load_gmem_a_k = k * MMA_K + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * MMA_K + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    uint32_t load_smem_a_ptr = (smem_a_base_ptr + 
            (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(half));
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
    uint32_t load_smem_b_ptr = (smem_b_base_ptr + 
            (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) * sizeof(half));
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

#pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) {
    int smem_sel = (k + 1) % K_STAGE;
    int smem_sel_next = k % K_STAGE;

    int load_gmem_a_k = k * MMA_K + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * MMA_K + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    uint32_t load_smem_a_ptr = (smem_a_base_ptr + 
        (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(half))
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
    uint32_t load_smem_b_ptr = (smem_b_base_ptr + 
        (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + A_PAD) + load_smem_b_n) * sizeof(half));
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();

    uint32_t RA[WARP_TILE_M][4];
    uint32_t RB[WARP_TILE_N][2];
  }
}