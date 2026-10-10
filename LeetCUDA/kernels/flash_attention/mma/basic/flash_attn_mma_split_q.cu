#include <utils.h>

// Step 1: Define Template Parameters
template <
    const int kHeadDim,
    const int kMmaAtomM, // M 16
    const int kMmaAtomN, // N 8
    const int kMmaAtomK, // K 16
    const int kMmaTileSeqLenQ,  //4 , 16*4 = 64
    const int kMmaTileSeqLenK,  // 1, 16*1 = 16
    const int kMmaTileSeqLenP,  // 4, 16*4 = 64
    const int kMmaTileHeadDimV, // 1, 8 *1 = 8
    const int kWarpTileSeqLenQ, // 1, 64 * 1 = 64
    const int kWarpTileSeqLenK, // 8, 8 * 8 = 64
    const int kWarpTileSeqLenP, 
    const int kWarpTileHeadDimV,
    const int kOStorageAccFloat32,
    const int kStage, const int kPad>
// Step 2: Kernel Launch & Tile Setup
__global__ void __launch_bounds__(WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK)
    flash_attn_mma_stages_split_q_kernel(half *Q, half *K, half *V, half *O, int QKV_seqlen, int QKV_head) {
    static_assert(kMmaAtomM == 16 && kMmaAtomN == 8  && kMmaAtomK == 16);
    static_assert(kMmaTileSeqLenQ <= 8 && kMmaTileSeqLenK == 1);
    static_assert(kMmaTileSeqLenP <= 8 && kMmaTileHeadDimV == 1);
    static_assert(kWarpTileSeqLenQ == 1 && kWarpTileSeqLenK <= 16);
    static_assert(kWarpTileSeqLenP == 1 && kWarpTileHeadDimV == (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV)));
    static_assert(kOStorageAccFloat32 == 0 || kOStorageAccFloat32 == 1);
    static_assert(kStage < 3 && kStage > 0);
    static_assert(kPad >= 0 && kPad % 8 == 0);

    constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
    constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
    static_assert(Br >= Bc);
    constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
    const int Tc = div_ceil(QKV_seqlen, Bc);
    constexpr float scale = 1.0f / sqrtf((float)kHeadDim);

    // Step 3: Grid Mapping & Memory Offsets
    const int QKV_batch_id = blockIdx.y / QKV_head; // batch size
    const int QKV_head_id = blockIdx.y % QKV_head;  // head num
    const int Q_tile_id = blockIdx.x;
    const int O_tile_id = Q_tile_id;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_QP = warp_id;
    const int warp_KV = 0;

    // global memory offsets
    const int Q_gemm_offset = ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) +
                                (QKV_head_id * QKV_seqlen * kHeadDim));                 // Q[seqlen, d]
    const int K_gemm_offset = ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) +
                                (QKV_head_id * QKV_seqlen * kHeadDim));                 // K[seqlen, d]
    const int V_gemm_offset = Q_gemm_offset;                                            // V[seqlen, d]
    const int O_gemm_offset = Q_gemm_offset;                                            // O[seqlen, d]

    // Step 4: Load & Organize Tiles in Shared Memory
    int load_smem_Q_Br = (tid / (kNumThreads / Br));
    int load_smem_Q_d = (tid % (kNumThreads / Br) * (kHeadDim / (kNumThreads / Br)));
    int load_smem_K_Bc = (tid / (kNumThreads / Bc));
    int load_smem_K_d = (tid % (kNumThreads / Bc) * (kHeadDim / (kNumThreads / Bc)));
    int load_smem_V_Bc = (tid / (kNumThreads / Bc));
    int load_smem_V_d = (tid % (kNumThreads / Bc) * (kHeadDim / (kNumThreads / Bc)));

    int load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br;
    if (load_gmem_Q_Br >= QKV_seqlen) return;
    int load_gmem_K_Bc_offset = 0;
    int load_gmem_V_Bc_offset = 0;

    extern __shared__ half smem[];
    constexpr int Q_tile_size = Br * (kHeadDim + kPad);
    constexpr int KV_tile_size = Bc * (kHeadDim + kPad);
    half *Q_tile_smem = smem;
    half *K_tile_smem = Q_tile_smem + Q_tile_size;
    half *V_tile_smem = K_tile_smem + kStage * KV_tile_size;

    uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
    uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
    uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

    float lane_block_row_max_old[kWarpTileSeqLenQ][2];
    float lane_block_row_sum_old[kWarpTileSeqLenQ][2];
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_max_old, -INFINITY);
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_sum_old, 0.0f);

    // Step 5: Register Setup
    uint32_t R_Q[kWarpTileSeqLenQ][4];
    uint32_t R_K[kWarpTileSeqLenK][2];
    uint32_t R_V[kWarpTileHeadDimV][2];
    uint32_t R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][2];
    uint32_t R_O[kWarpTileSeqLenP][kWarpTileHeadDimV][2];
    uint32_t R_D[kWarpTileSeqLenP][kWarpTileHeadDimV][2];
    fill_3D_regs<uint32_t, kWarpTileSeqLenQ, kWarpTileSeqLenK>(R_S, 0);
    fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV>(R_O, 0);
    fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV>(R_D, 0);

    // Step 6: Load Q into Shared Memory
    {
        int load_gmem_Q_d = load_smem_Q_d;
        int load_gmem_Q_addr = (Q_gemm_offset + load_gmem_Q_Br * kHeadDim + load_gmem_Q_d);
        uint32_t load_smem_Q_ptr = (smem_Q_base_ptr + (load_smem_Q_Br * (kHeadDim + kPad) + load_smem_Q_d) * sizeof(half));
    #pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Br)); i+=8) {
            CP_ASYNC_CG(load_smem_Q_ptr + i * 2, &Q[load_gmem_Q_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
    }
}
