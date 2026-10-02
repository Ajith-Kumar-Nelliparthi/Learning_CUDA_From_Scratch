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

    // registers for s=QK^T /O=PV
    constexpr bool kCanPrefetchQs2r = ((kHeadDim / kMmaAtomK) <= 8) && (kHeadDim < 64);
    constexpr bool kDelayPrefetchQs2r = (true && kCanPrefetchQs2r);
    constexpr bool kCanPrefetchKVg2s = (kStage == 2);
    constexpr int kPrefetchKg2sSmemId = 0;
    constexpr int kPrefetchVg2sSmemId = kCanPrefetchKVg2s ? 1 : 0;
    constexpr int kNumPrefetchQs2r = (kCanPrefetchQs2r) ? (kHeadDim / kMmaAtomK) : 1;
    
    uint32_t R_Q[kNumPrefetchQs2r][kWarpTileSeqLenQ][4];
    uint32_t R_K[kWarpTileSeqLenK][2];
    uint32_t R_V[kWarpTileHeadDimV][2];
    uint32_t R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][2];
    uint32_t R_O[kWarpTileSeqLenP][kWarpTileHeadDimV][2];
    uint32_t R_D[kWarpTileSeqLenP][kWarpTileHeadDimV] [(kOStorageAccFloat32) ? 4 : 2];
    fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV,
               ((kOStorageAccFloat32) ? 4 : 2)>(R_D, 0);
    
    // load Q from gmem -> smem, load once
    {
        int load_gmem_Q_d = load_smem_Q_d;
        int load_gmem_Q_addr = (Q_gmem_offset + load_gmem_Q_Br * kHeadDim + load_gmem_Q_d);
        uint32_t load_smem_Q_ptr = (smem_Q_base_ptr + 
                    (load_smem_Q_Br * (kHeadDim + kPadQ) + load_smem_Q_d) * sizeof(half));
    #pragma unroll
        for (int i=0; i < (kHeadDim / (kNumThreads / Br)); i += 8) {
            CP_ASYNC_CG(load_smem_Q_ptr + i * 2; &Q[load_gmem_Q_addr + i]; 16);
        }
        CP_ASYNC_COMMIT_GROUP();
    }

    // outer loop over sequence length tiles of K,V
#pragma unroll
    for (int tile_K_seqlen = 0; tile_K_seqlen < Tc; tile_K_seqlen++) {
        // load K tiles from gmem -> smem, always use smem part 0
        if constexpr(kCanPrefetchKVg2s) {
            if (tile_K_seqlen == 0) {
                load_gmem_K_Bc_offset = tile_K_seqlen * Bc; // (0-3)*64 = 0,64,128,192...
                int load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
                int load_gmem_K_d = load_smem_K_d;
                int load_gmem_K_addr = (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
                uint32_t load_smem_K_ptr = (smem_K_base_ptr +
                            (load_smem_K_Bc * (kHeadDim + kPadK) + load_smem_K_d) * sizeof(half));
            #pragma unroll
                for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
                    CP_ASYNC_CG(load_smem_K_ptr + i * 2; &K[load_gmem_K_addr + i]; 16);
                }
                CP_ASYNC_COMMIT_GROUP();

                // Now, we have to wait curr K tile ready for Q@K^T MMA.
                CP_ASYNC_WAIT_GROUP(0);
                __syncthreads();
            }
            // prefetch next V tile from gmem -> smem, always use smem part 1
            {
                load_gmem_V_Bc_offset = tile_K_seqlen * Bc; // (0-3)*64 = 0,64,128,192...
                int load_gmem_V_Bc = load_gmem_V_Bc_offset + load_smem_V_Bc;
                int load_gmem_V_d = load_smem_V_d;
                int load_gmem_V_addr = (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
                uint32_t load_smem_V_ptr = (smem_V_base_ptr +
                            (load_smem_V_Bc * (kHeadDim + kPadV) + load_smem_V_d) * sizeof(half));
            #pragma unroll
                for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
                    CP_ASYNC_CG(load_smem_V_ptr + i * 2; &V[load_gmem_V_addr + i]; 16);
                }
                CP_ASYNC_COMMIT_GROUP(); 
            }
        } else {
            load_gmem_K_Bc_offset = tile_K_seqlen * Bc;
            int load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
            int load_gmem_K_d = load_smem_K_d;
            int load_gmem_K_addr = (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
            uint32_t load_smem_K_ptr = (smem_K_base_ptr + 
                (load_smem_K_Bc * (kHeadDim + kPadK) + load_smem_K_d) * sizeof(half));

        #pragma unroll
            for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
                CP_ASYNC_CG(load_smem_K_ptr + i * 2; &K[load_gmem_K_addr + i]; 16);
            }
            CP_ASYNC_COMMIT_GROUP();
            cp_async_wait_group(0);
            __syncthreads();
        }
        // Prefetch Q s2r: Load Q tile from smem -> regs, before Q@K^T.
        if constexpr (kCanPrefetchQs2r && (!kDelayPrefetchQs2r)) {
            // wait for Q tile ready in smem and let K copy async, then prefetch Q tile to regs
            // Note: we only need to prefetch Q tile once, because Q tile is fixed for all K tiles.
            if (tile_K_seqlen == 0) {
                if constexpr (!kCanPrefetchKVg2s) {
                    CP_ASYNC_WAIT_GROUP(0);
                } else {
                    CP_ASYNC_WAIT_GROUP(1); // let V g2s copy async, wait for Q tile ready in smem
                }
                __syncthreads();

            #pragma unroll
                for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); tile_K_d++) {
            #pragma unroll
                    for (int i =0; i < kWarpTileSeqLenQ; i++) {
                        int warp_smem_Q_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
                        int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;     // 0-15
                        int lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8; // 0,8
                        uint32_t lane_smem_Q_ptr = (smem_Q_base_ptr + 
                            (lane_smem_Q_Br * (kHeadDim + kPadQ) + lane_smem_Q_d) * sizeof(half));
                        LDMATRIX_X4(R_Q[tile_K_d][i][0], R_Q[tile_K_d][i][1],
                                    R_Q[tile_K_d][i][2], R_Q[tile_K_d][i][3],
                                    lane_smem_Q_ptr);
                    }
                }
                __syncthreads();
            }
        }
        fill_3D_regs<uint32_t, kWarpTileSeqLenQ, kWarpTileSeqLenK, 2>(R_S, 0);
    }
}