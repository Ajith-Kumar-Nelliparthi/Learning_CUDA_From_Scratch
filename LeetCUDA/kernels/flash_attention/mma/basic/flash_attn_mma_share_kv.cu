#include <utils.h>

/*
FlashAttention - 2 from scratch using Tensor Cores with MMA PTX
instruction. The i/p is Q,K,V, 4D Tensor with shape [batch_size, num_heads,
seq_len, head_dim]. The o/p is O, a 4D tensor with shape as above.

Q,K,V,O : [batch_size, num_heads, seq_len, head_dim], [B, H, N, d]
each block processes Q_tile with shape [Br, d] and full K,V with shape [N, d]

Split Q across MMA(warps) and keep access KV for all MMA(warps)

MMA = m16n8k16, Br=16x4=64, Bc=8x8=64
*/

template <
    const int kHeadDim,     //32,64,128
    const int kMmaAtomM,    //M 16
    const int kMmaAtomN,    //N 8
    const int kMmaAtomK,    //K 16
    const int kMmaTileSeqLenQ,      //4 , M = 16*4=64
    const int kMmaTileSeqLenK,      //1, N=8*1=8
    const int kMmaTileSeqLenP,      //4, M=16*4=64
    const int kMmaTileHeadDimV,     //1, N=8*1=8
    const int kWarpTileSeqLenQ,     //1, M, Br=64*1=64
    const int kWarpTileSeqLenK,     //8, N, Bc=8*8=64
    const int kWarpTileSeqLenP,
    const int kWarpTileHeadDimV,
    const int kOStorageAccFloat32,      // 0/1, MMA Acc always be fp16, but storage can be fp32 or half.
    const int kStage,
    const int kPadQ,        //Pad Q/K/V 0,8
    const int kPadK, const int kPadV>
__global__ void __launch_bounds(WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK)
    flash_attn_mma_stages_split_q_shared_kv_kernel(half *Q, half *K, half *V, half *O, int QKV_seqlen, int QKV_head) {
    // --- Matmul layout ---
    // Q[Br, d] @ K^T[d, Bc] → produces score matrix S[Br, Bc].
    // P[Br, Bc] @ V[Bc, d] → produces output O[Br, d].
    // Note: K stored row-major, so K^T is effectively col-major.
    static_assert(kMmaAtomM == 16 && kMmaAtomN == 8 && kMMaAtomK == 16);
    static_assert(kMmaTileSeqLenQ <= 8 && kMmaTileSeqLenK == 1);
    static_assert(kMmaTileSeqLenP <= 8 && kMmaTileSeqLenV == 1);
    static_assert(kWarpTileSeqLenQ == 1 && kWarpTileSeqLenK <= 16);
    static_assert(kWarpTileSeqLenP == 1 && kWarpTileHeadDimV == (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV))); // P@V
    static_assert(kOStorageAccFloat32 == 0 || kOStorageAccFloat32 == 1);
    static_assert(kStage < 3 && kStage > 0);
    static_assert(kPadQ >= 0 && kPadQ % 8 == 0); // 0,8,16
    static_assert(kPadK >= 0 && kPadK % 8 == 0); // 0,8,16
    static_assert(kPadV >= 0 && kPadV % 8 == 0); // 0,8,16

    // tile sizes
    constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;  //16*4*1 = 64
    constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;  //8*1*8 = 64
    static_assert(Br >= Bc);
    constexpr int kNumThreads = (WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK);  //32*4*1 = 128
    const int Tc = div_ceil(QKV_Seqlen, Bc);
    const float scale = 1.0f / sqrt((float)kHeadDim);

    // grid layout
    const int QKV_batch_id = blockIdx.y / QKV_head;     //batch size
    const int QKV_head_id = blockIdx.y % QKV_head;      // head num
    const int Q_tile_id = blockIdx.x;
    const int O_tile_id = Q_tile_id;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_QP = warp_id;    // which Q slice this warp owns
    const int warp_KV = 0;      // all warps owns same KV tile

    // global memory offsets
    const int Q_gmem_offset = ((QKV_batch_id * QKV_head * QKV_Seqlen * kHeadDim) + 
                                (QKV_head_id * QKV_Seqlen * kHeadDim));
    const int K_gmem_offset = ((QKV_batch_id * QKV_head * QKV_Seqlen * kHeadDIm) +
                                (QKV_head_id * QKV_Seqlen * kHeadDim));
    const int V_gmem_offset = Q_gmem_offset;
    const int O_gmem_offset = Q_gmem_offset;

    int load_smem_Q_Br = (tid / (kNumThreads / Br));    // row 0-64
    int load_smem_Q_d = (tid % (kNumThreads / Br) * (kHeadDim / (kNumThreads / Br)));   // 0,32,,,,
    int load_smem_K_Bc = (tid / (kNumThreads / Bc));    // row 0-64
    int load_smem_K_d = (tid % (kNumThreads / Bc) * (kHeadDim / (kNumThreads / Bc)));   // 0,32,....
    int load_smem_V_Bc = (tid / (kNumThreads / Bc));    // row 0-64
    int load_smem_V_d = (tid % (kNumThreads / Bc) * (kHeadDim / (kNumThreads / Bc)));   // 0,32,....

    int load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br;
    if (load_gmem_Q_Br >= QKV_seqlen) return;
    int load_gmem_K_Bc_offset = 0;
    int load_gmem_V_Bc_offset = 0;

    extern __shared__ half smem[];
    constexpr int Q_tile_size = Br * (kHeadDim * kPadQ);
    constexpr int K_tile_size = Bc * (kHeadDim * kPadK);
    constexpr int V_tile_size = Bc * (kHeadDim * kPadV);
    half *Q_tile_smem = smem;
    half *K_tile_smem = Q_tile_smem + Q_tile_size;
    half *V_tile_smem = K_tile_smem;

    uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
    uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
    uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

    float lane_block_row_max_old[kWarpTileSeqLenQ][2];
    float lane_block_row_sum_old[kWarpTileSeqLenQ][2];
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_max_old, -INFINITY);
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_sum_old, 0.0f);
}