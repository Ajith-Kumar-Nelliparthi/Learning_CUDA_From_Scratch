// load header files
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <float.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
using namespace nvcuda;

// Common vector types and inline PTX helpers used by the tensor-core kernel.
#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST32BITS(valu) (reinterpret_cast<half2 *>(&(value))[0])
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
// Load one or more 8x8 half tiles from shared memory into lane registers.
#define LDMATRIX_X1(R, addr) \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"   \
                : "=r"(R)   \
                : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"   \
                : "=r"(R0), "=r"(R1)    \
                : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr)   \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"   \
                : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)    \
                : "r"(addr))
#define LDMATRIX_X1_T(R, addr) \
    asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"   \
                : "=r"(R)   \
                : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"   \
                : "=r"(R0), "=r"(R1)    \
                : "r"(addr))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr)   \
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"   \
                : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)    \
                : "r"(addr))
// Issue the 16x8x16 half-precision matrix multiply-accumulate instruction.
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)     \
    asm volatil("mma.sync.aligned.shared.m8n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                :"=r"(RD0), "=r"(RD1)   \
                :"=r"(RA0), "=r"(RA1), "=r"(RA2), "=r"(RA3), "=r"(RB0), "=r"(RB1), "=r"(RC0), "=r"(RC1))
HOST_DEVICE_INLINE
int div_ceil(int a, int b) {return (a % b != 0) ? (a / b + 1) : (a / b);}

// One warp computes one 16x8 output tile using an m16n8k16 MMA operation.
template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16>
__global__ void hgemm_mma_m16n8k16_naive_kernel(half *A, half *B, half *C, int M, int N, int K) {
    // Each block identifies one output tile in the logical C matrix.
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, MMA_K);
    constexpr int BM = MMA_M;
    constexpr int BN = MMA_N;
    constexpr int BK = MMA_K;

    // Shared-memory tiles for A, B, and the block's partial output.
    __shared__ half s_a[MMA_M][MMA_K], s_b[MMA_K][MMA_N], s_c[MMA_M][MMA_N];

    // The block is expected to contain exactly one warp; lane_id is used by
    // ldmatrix and MMA instructions to distribute tile fragments.
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int lane_id = tid % WARP_SIZE;

    // Each pair of lanes loads one row of the A tile, while each lane owns
    // one row of the B tile. The offsets provide the global tile origin.
    const int load_smem_a_m = tid / 2;  // row 0-15
    const int load_smem_a_k = (tid % 2) * 8;    //col 0,8
    const int load_smem_b_k = tid;
    const int load_smem_b_n = 0;
    const int load_gmem_a_m = by * BM + load_smem_a_m;
    const int load_gmem_b_n = bx * BN + load_smem_b_n;
    // Avoid out-of-bounds accesses for partial tiles at the matrix edges.
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    uint32_t RC[2] = {0, 0};

    #pragma unroll
    for (int k = 0; k < NUM_K_TILES; k++) {
        int load_gmem_a_k = k * MMA_K + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) = LDST128BITS(A[load_gmem_a_addr]);

        if (lane_id < MMA_K) {
            int load_gmem_b_k = k * MMA_K + load_smem_b_k;
            int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
            LDST128BITS(s_b[load_smem_b_k][load_smem_b_n]) = (LDST128BITS(B[load_gmem_b_addr]));
        }
        __syncthreads();

        uint32_t RA[4];
        uint32_t RB[2];

        // ldmatrix for s_a, ldmatrix.trans for s_b.
        uint32_t load_smem_a_ptr = __cvta_generic_to_shared(&s_a[lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], load_smem_a_ptr);
        uint32_t load_smem_b_ptr = __cvta_generic_to_shared(&s_b[lane_id % 16][0]);
        LDMATRIX_X2_T(RB[0], RB[1], load_smem_b_ptr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
        __syncthreads();
    }

    // s_c[16][8]
    LDST32BITS(s_c[lane_id / 4][(lane_id % 4) * 2]) = LDST32BITS(RC[0]);
    LDST32BITS(s_c[lane_id / 4 + 8][(lane_id % 4) * 2]) = LDST32BITS(RC[1]);
    __syncthreads();

    // store s_c[16][8]
    if (lane_id < MMA_M) {
        // store 128 bits  per memory issue
        int store_gmem_c_m = by * BM + lane_id;
        int store_gmem_c_n = bx * BN;
        int store_gmem_c_Addr = store_gmem_c_m * N + store_gmem_c_n;
        LDST128BITS(C[store_gmem_c_Addr]) = LDST128BITS(s_c[lane_id][0]);
    }
}

// 128x128
template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
            const int MMA_TILE_M = 2, const int MMA_TILE_N = 4,
            const int WARP_TILE_M = 4, const int WARP_TILE_N = 4,
            const int A_PAD = 0, const int B_PAD = 0>
__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel(half *A, half *B, half *C, int M, int N, int K) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, MMA_K);
    constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;    //16x2x4 = 128
    constexpr int BN = MMA_N * MMA_TILE_N ?* WARP_TILE_N;   //8x4x4 = 128
    constexpr int BK = MMA_K;

    __shared__ half s_a[BM][BK + A_PAD];
    __shared__ half s_b[BK][BN + B_PAD];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_m = warp_id % 2;     //2 warps in M dir (0,1)
    const int warp_n = warp_id / 2;     //4 warps in N dir (0,1,2,3)

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
        for (int j = 0; j < WARP_TILE_N; j++) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }

#pragma unroll
    for (int k = 0; k < NUM_K_TILES; k++) {
        int load_gmem_a_k = k * MMA_K + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * MMA_K + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) = (LDST128BITS(A[load_gmem_a_addr]));
        LDST128BITS(s_b[load_smem_b_k][load_smem_b_n]) = (LDST128BITS(B[load_gmem_b_addr]));
        __syncthreads();

        // ld.matrix for s_a, ld.mat trans for s_b
        uint32_t RA[WARP_TILE_M][4];
        uint32_t RB[WARP_TILE_N][2];

        // smem -> reg
    #pragma unroll
        for (int i = 0; i < WARP_TILE_M; i++) {
            int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            int lane_smem_a_m = warp_smem_a_m + lane_id % 16;   //0-15
            int lane_smem_a_k = (lane_id / 16) * 8; //0,8
            uint32_t load_smem_a_ptr = __cvta_generic_to_shared(&s_a[load_smem_a_m][load_smem_a_k]);
            LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
        }

    #pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++) {
            int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            int lane_smem_b_k = lane_id % 16;
            int lane_smem_b_n = warp_smem_b_n;
            uint32_t load_smem_b_ptr = __cvta_generic_to_shared(&s_b[load_smem_b_k][load_smem_b_n]);
             LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
        }

    #pragma unroll
        for (int i = 0; i < WARP_TILE_M; i++) {
    #pragma unroll
            for (int j = 0; j < WARP_TILE_N; j++) {
                HMMA16816(RC[i][j][0], RC[i][j][1], RA[i][0], RA[i][1], RA[i][2],
                  RA[i][3], RB[j][0], RB[j][1], RC[i][j][0], RC[i][j][1]);
            }
        }
        __syncthreads();
    }
    // store result
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;

      int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
      int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n + (lane_id % 4) * 2;
      int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
      int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;

      LDST32BITS(C[store_gmem_c_addr_0]) = LDST32BITS(RC[i][j][0]);
      LDST32BITS(C[store_gmem_c_addr_1]) = LDST32BITS(RC[i][j][1]);
    }
  }
}
