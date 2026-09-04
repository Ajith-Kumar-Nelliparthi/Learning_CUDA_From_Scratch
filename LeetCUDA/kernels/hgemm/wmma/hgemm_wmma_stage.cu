// define macros and constants
#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <float.h>
#include <cuda_fp8.h>
#include <math.h>
#include <mma.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
using namespace nvcuda;


#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 * >(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 * >(&(value))[0])
// commit & weight macros
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" :: "n"(n))
// async copy macros
#define CP_ASYNC_CA(dst, src, bytes) \
    asm volatile( \
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2,\n" ::"r"(dst), \
        "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) \
    asm volatile( \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2,\n" ::"r"(dst), \
        "l"(src), "n"(bytes))

// helper for ceil division
HOST_DEVICE_INLINE 
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// configuration for the kernel
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
            const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
            const int WARP_TILE_M = 2, const int WARP_TILE_N = 4,
            const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 2,
            const bool BLOCK_SWIZZLE = false>
__global__ void __launch_bounds__(256)
        hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_kernel(half *A, half *B, half *C, int M, int N, int K) {
    // 256 threads (8 warps) per block
    // index calculation (thread & warp)
    // block swizzle 0/1 control use block swizlle or not
    const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;  // 16x4x2 = 128
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;  // 16x2x4 = 128
    constexpr int BK = WMMA_K;

    __shared__ half s_a[K_STAGE][BM][BK + A_PAD], s_b[K_STAGE][BK][BN + B_PAD];
    /* To find a specific element in a multi-dimensional array 
    like s_a[stage][m][k], the GPU needs to know how many elements are in one "stage." 
    */
    constexpr int s_a_stage_offset = BM * (BK + A_PAD);
    constexpr int s_b_stage_offset = BK * (BN + B_PAD);
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_m = warp_id / 2; // 0,1,2,3
    const int warp_n = warp_id % 2; // 0,1

    // shared memory and global memory index calculation
    int load_smem_a_m = tid / 2;    // row 0-127
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;     // col 0,8
    int load_smem_b_k = tid / 16;   // row 0-15
    int load_smem_b_n = (tid % 16) * 8;     // col 0,8,16,,,,120

    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    // fill fragment C
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_M, WMMA_K, half>C_frag[WARP_TILE_M][WARP_TILE_N];
#pragma unroll
    for (int i=0; i<WARP_TILE_M; i++) {
#pragma unroll
        for (int j=0; j<WARP_TILE_N; j++) {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }
}