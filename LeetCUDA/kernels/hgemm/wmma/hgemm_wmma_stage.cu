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

    // stage 0 load (the prefill)
    // convert shared memory pointers to generic addresses for cp.async
    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

#pragma unroll
    for (int k=0; k < (K_STAGE - 1); k++) { // 0,1
        int load_gmem_a_k = k * WMMA_K + load_smem_a_k;     // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k;     // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_a_ptr =
            (smem_a_base_ptr + 
                (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
                        sizeof(half));
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr = 
            (smem_b_base_ptr + 
                (k + s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) * 
                        sizeof(half));
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

        CP_ASYNC_COMMIT_GROUP();
    }
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);   //stag2->0,stage3->1,stage4->2
    __syncthreads();

    // main pipeline
#pragma unroll
    for (int k = (K_STAGE - 1); k<NUM_K_TILES; k++) {
        // identify which stage stats to use
        int smem_sel = (k - 1) % K_STAGE;
        int smem_sel_next = k % K_STAGE;

        int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_a_ptr = 
            (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(half));
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr = 
            (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) * sizeof(half));
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
        CP_ASYNC_COMMIT_GROUP();

        // define fragments for A,B
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>B_frag[WARP_TILE_N];

        // compute stage 0
    #pragma unroll
        for (int i=0; i<WARP_TILE_M; i++) {
            const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][0], BK + A_PAD);
        }
    #pragma unroll
        for (int j=0; j<WARP_TILE_N; j++) {
            const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            wmma::load_matrix_sync(B_frag[j], &s_b[smem_sel][0][warp_smem_b_n], BN + B_PAD);
        }

    #pragma unroll
        for (int i=0; i<WARP_TILE_M; i++) {
    #pragma unroll
            for (int j=0; j<WARP_TILE_N; j++) {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }
    // make sure all memory issues ready
    if ((K_STAGE - 2) > 0) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // processing last (K_STAGE - 1) tile iters
    {
    #pragma unroll
        for (int k=0; k<(K_STAGE - 1); k++) {
            const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>A_frag[WARP_TILE_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>B_frag[WARP_TILE_N];

        #pragma unroll
            for (int i=0; i<WARP_TILE_M; i++) {
                const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
                wmma::load_matrix_sync(A_frag[i], &s_a[stage_sel][warp_smem_a_m][0], BK + A_PAD);
            }
        #pragma unroll
            for (int j=0; j<WARP_TILE_N; j++) {
                const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
                wmma::load_matrix_sync(B_frag[j], &s_b[stage_sel][0][warp_smem_b_n], BN + B_PAD);
            }
        
        #pragma unroll
            for (int i=0; i<WARP_TILE_M; i++) {
        #pragma unroll
                for (int j=0; j<WARP_TILE_N; j++) {
                    wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
                }
            }
        }
    }

    // store back to c
#pragma unroll
    for (int i=0; i<WARP_TILE_M; i++) {
#pragma unroll
        for (int j=0; j<WARP_TILE_N; j++) {
            const int store_gmem_c_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            const int store_gmem_c_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            wmma::store_matrix_sync(C + store_gmem_c_m * N + store_gmem_c_n, C_frag[i][j], N, wmma::mem_row_major)
        }
    }
}

template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
            const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
            const int WARP_TILE_M = 2, const int WARP_TILE_N = 4,
            const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 2,
            const bool BLOCK_SWIZLLE = false>
__global__ void __launch_bounds__(256)
    hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel(half *A, half *B, half *C, int M, int N, int K) {
    // 256 threads (8 warps) per block
    const int bx = ((int)BLOCK_SWIZLLE) * blockIdx.z * gridDim.x + blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;  // 16x4x2 = 128
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;  // 16x2x4 = 128
    constexpr int BK = WMMA_K;

    // dynamic shared memory allocation
    extern __shared__ half smem[];
    half *s_a = smem; // start at byte 0
    // Calculate how many 'half' elements are in all stages of s_a
    half *s_b = smem + K_STAGE * BM * (BK + A_PAD); // Start s_b right where s_a ends so that s_b not overwrite s_a

    /* To find a specific element in a multi-dimensional array 
    like s_a[stage][m][k], the GPU needs to know how many elements are in one "stage." 
    */
   constexpr int s_a_stage_offset = BM * (BK + A_PAD);
   constexpr int s_b_stage_offset = BK * (BN + B_PAD);

   // thread and warp index
   const int tid = threadIdx.y * blockDim.x + threadIdx.x;
   const int warp_id = tid / WARP_SIZE;
   const int lane_id = tid % WARP_SIZE;
   const int warp_m = warp_id / 2;
   const int warp_n = warp_id % 2;

   // smem and hbm index
   int load_smem_a_m = tid / 2;
   int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;
   int load_smem_b_k = tid / 16;
   int load_smem_b_n = (tid % 16) * 8;

   int load_gmem_a_m = by * BM + load_smem_a_m;
   int load_gmem_b_n = bx * BN + load_smem_b_n;
   if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

   // fill fragment c
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>C_frag[WARP_TILE_M][WARP_TILE_N];
#pragma unroll
    for (int i=0; i<WARP_TILE_M; i++) {
#pragma unroll
        for (int j=0; j<WARP_TILE_N; j++) {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    // convert shared memory pointer addresses to generic addresses for cp.async
    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

    // stage 0 load (the prefill)
#pragma unroll
    for (int k=0; k < (K_STAGE - 1); k++) {
        int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        // calculate linear offset: base + (stage * stage_size) + (row * row_size)
        uint32_t load_smem_a_ptr = (
            smem_a_base_ptr + (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(half));
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr = (
            smem_b_base_ptr + (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) * sizeof(half));
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

        CP_ASYNC_COMMIT_GROUP();
    }
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);   // s2->0, s3->1, s4->2
    __syncthreads();

    // main pipeline
#pragma unroll
    for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) {
        int smem_sel = (k + 1) % K_STAGE;
        int smem_sel_next = k % K_STAGE;
        
        int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        // load stage 2
        uint32_t load_smem_a_ptr = (
            smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(half));
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr = (
            smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_k) * sizeof(half));
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
        CP_ASYNC_COMMIT_GROUP();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>B_frag[WARP_TILE_N];

        // compute stage 0
    #pragma unroll
        for (int i=0; i<WARP_TILE_M; i++) {
            // load 2 tiles
            int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            half *load_smem_a_frag_ptr = (s_a + smem_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD) + 0);
            wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
        }
    #pragma unroll
        for (int j=0; j<WARP_TILE_N; j++) {
            int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            half *load_smem_b_frag_ptr = (s_b + smem_sel * s_b_stage_offset + 0 * (BN + B_PAD) + warp_smem_b_n);
            wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
        }

    #pragma unroll
        for (int i=0; i<WARP_TILE_M; i++) {
    #pragma unroll
            for (int j=0; j<WARP_TILE_N; j++) {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }
        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    }
    if ((K_STAGE - 2) > 0) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // compute last tile
    {
#pragma unroll
        for (int k=0; k < (K_STAGE - 1); k++) {
            const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>A_frag[WARP_TILE_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>B_frag[WARP_TILE_N];

        #pragma unroll
            for (int i=0; i<WARP_TILE_M; i++) {
                int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
                half *load_smem_a_frag_ptr = (s_a + stage_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD) + 0);
                wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
            }
        #pragma unroll
            for (int j=0; j<WARP_TILE_N; j++) {
                int warp_smem_b_n = warp_n * (WMMA_N + WARP_TILE_N) + j * WMMA_N;
                half *load_smem_b_frag_ptr = (s_b + stage_sel * s_b_stage_offset + 0 * (BN + B_PAD) + warp_smem_b_n);
                wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
            }
        
        #pragma unroll
            for (int i=0; i<WARP_TILE_M; i++) {
        #pragma unroll
                for (int j=0; j<WARP_TILE_N; j++) {
                    wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
                }
            }
        }
    }

    // store results back to c
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

// stage with 256x256 block, mma4x4, warp4x4(64,64,16)
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
            const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 4,
            const int WARP_TILE_M = 4, const int WARP_TILE_N = 4,
            const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 2,
            bool BLOCK_SWIZZLE = false>
__global__ void __launch_bounds__(512) 
    hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem_kernel(half *A, half *B, half *C, int M, int N, int K) {
        // 512 threads (16 warps) per block
        const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
        const int by = blockIdx.y;
        const int NUM_K_TILES = div_ceil(K, WMMA_K);
        constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;  //16x4x4 = 256
        constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;  //16x4x4 = 256
        constexpr int BK = WMMA_K;

        extern half __shared__ smem[];
        half *s_a = smem;
        half *s_b = smem + K_STAGE * WMMA_M * BM * (BK + A_PAD);

        // calculate how many elements in a stage
        constexpr int s_a_stage_offset = BM * (BK + A_PAD);
        constexpr int s_b_stage_offset = BK * (BN + B_PAD);

        // thread and warp index
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int warp_id = tid / WARP_SIZE;    //0-16
        const int lane_id = tid % WARP_SIZE;    //0-31
        const int warp_m = warp_id / 4;     //0,1,2,3
        const int warp_n = warp_id % 4;     //0,1,2,3

        // smem and global memory index
        int load_smem_a_m = tid / 2;                    //row 0-255
        int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;     //col 0,8
        int load_smem_b_k = tid / 32;                   //row 0-15
        int load_smem_b_n = (tid % 32) * 8;             // col 0,8,16,32,,,
        int load_gmem_a_m = by * BM + load_smem_a_m;    // global row of a and c
        int load_gmem_b_n = bx * BN + load_smem_b_n;    // global col of b and c
        if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

        // fragment c
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>C_frag[WARP_TILE_M][WARP_TILE_N];
    #pragma unroll
        for (int i=0; i<WARP_TILE_M; i++) {
    #pragma unroll
            for (int j=0; j<WARP_TILE_N; j++) {
                wmma::fill_fragment(C_frag[i][j], 0.0);
            }
        }

        // convert smem pointer addresses to generic addresses
        uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
        uint32_t smem_b_base_ptr = __cvta_global_to_generic(s_b);

        // stage 0 load (the prefill)
    #pragma unroll
        for (int k=0; k < (K_STAGE - 1); k++) {
            int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
            int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
            int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
            int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

            // calculate linear offset: base + (stage * stage_size) + (row * row_size)
            uint32_t load_smem_a_ptr = (
                smem_a_base_ptr + (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(half));
            CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

            uint32_t load_smem_b_ptr = (
                smem_b_base_ptr + (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) * sizeof(half));
            CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], half);

            CP_ASYNC_COMMIT_GROUP();
        }
        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();

        // main pipeline
    #pragma unroll
        for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) {
            int smem_sel = (k + 1) % K_STAGE;
            int smem_sel_next = k % K_STAGE;

            int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
            int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
            int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
            int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

            // load stage 2
            uint32_t load_smem_a_ptr = 
                (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(half));
            CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

            uint32_t load_smem_b_ptr = 
                (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) * sizeof(half));
            CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
            CP_ASYNC_COMMIT_GROUP();

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>A_frag[WARP_TILE_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>B_frag[WARP_TILE_N];

            // compute stage 0
        #pragma unroll
            for (int i=0; i<WARP_TILE_M; i++) {
                int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;   // row offset in smem for matA
                half *load_smem_a_frag_ptr = (s_a + smem_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD) + 0);
                wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
            }
        #pragma unroll
            for (int j=0; j<WARP_TILE_N; j++) {
                int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;   //col offset in smem for matB
                half *load_smem_b_frag_ptr = (s_b * smem_sel * s_b_stage_offset + 0 * (BN + B_PAD) + warp_smem_b_n);
                wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
            }
        #pragma unroll
            for (int i=0; i<WARP_TILE_M; i++) {
        #pragma unroll
                for (int j=0; j<WARP_TILE_N; j++) {
                    wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
                }
            }
            CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
            __syncthreads();
    }
        if ((K_STAGE - 2) > 0) {
            CP_ASYNC_WAIT_GROUP(0);
            __syncthreads();
        }
    {
#pragma unroll
        for (int k=0; k < (K_STAGE - 1); k++) {
            const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>A_frag[WARP_TILE_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>B_frag[WARP_TILE_N];

        #pragma unroll
            for (int i=0; i<WARP_TILE_M; i++) {
                int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
                half *load_smem_a_frag_ptr = (s_a + stage_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD) + 0);
                wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
            }
        #pragma unroll
            for (int j=0; j<WARP_TILE_N; j++) {
                int warp_smem_b_n = warp_n * (WMMA_N + WARP_TILE_N) + j * WMMA_N;
                half *load_smem_b_frag_ptr = (s_b + stage_sel * s_b_stage_offset + 0 * (BN + B_PAD) + warp_smem_b_n);
                wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
            }
        
        #pragma unroll
            for (int i=0; i<WARP_TILE_M; i++) {
        #pragma unroll
                for (int j=0; j<WARP_TILE_N; j++) {
                    wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
                }
            }
        }
    }

    // store results back to c
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
