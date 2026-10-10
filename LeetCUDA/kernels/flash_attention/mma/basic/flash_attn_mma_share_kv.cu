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
    #pragma unroll
        for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); tile_K_d++) {
            if constexpr(!kCanPrefetchQs2r) {
            #pragma unroll
                for (int i = 0; i < kWarpTileSeqLenQ; i++) {
                    int warp_smem_Q_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
                    int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;     // 0-15
                    int lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8; // 0,8
                    uint32_t lane_smem_Q_ptr = (smem_Q_base_ptr +
                        (lane_smem_Q_Br * (kHeadDim + kPadQ) + lane_smem_Q_d) * sizeof(half));
                    LDMATRIX_X4(R_Q[0][i][0], R_Q[0][i][1], R_Q[0][i][2], R_Q[0][i][3], lane_smem_Q_ptr);
                }
            } else { //kCanPrefetchQs2r == true
                if constexpr (kDelayPreftchQs2r) {
                    if (tile_K_seqlen == 0) {
                        if (tile_K_d == 0) {
                            if constexpr (!kCanPrefetchKVg2s) {
                                CP_ASYNC_WAIT_GROUP(0);
                            } else {
                                CP_ASYNC_WAIT_GROUP(1); // let V g2s copy async, wait for Q tile ready in smem
                            }
                            __syncthreads();
                        }
                    #pragma unroll
                        for (int i = 0; i < kWarpTileSeqLenQ; i++) {
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
                }
            }
        #pragma unroll
            for (int j = 0; j < kWarpTileSeqLenK; j++) {
                int warp_smem_K_Bc = warp_KV * (kMmaAtomN * kWarpTileSeqLenK) + j * kMmaAtomN;
                int lane_smem_K_Bc = warp_smem_K_Bc + lane_id % 8;      // 0-7
                int lane_smem_K_d = tile_K_d * kMmaAtomK + ((lane_id / 8) % 2) * 8; // 0,8
                uint32_t lane_smem_K_ptr = (smem_K_base_ptr +
                    (lane_smem_K_Bc * (kHeadDim + kPadK) + lane_smem_K_d) * sizeof(half));
                LDMATRIX_X2_T(R_K[j][0], R_K[j][1], lane_smem_K_ptr);
            }
            if constexpr(kCanPrefetchQs2r) {
                static_assert(kWarpTileSeqLenQ == 1);
                {
                #pragma unroll
                    for (int j = 0; j < kWarpTileSeqLenK; j++) {
                        HMMA16816(R_S[0][j][0], R_S[0][j][1], R_Q[tile_K_d][0][0], R_Q[tile_K_d][0][1],
                                R_Q[tile_K_d][0][2], R_Q[tile_K_d][0][3],
                                R_K[j][0], R_K[j][1], R_S[0][j][0], R_S[0][j][1]);
                    }
                }
            } else {
                static_assert(kWarpTileSeqLenQ == 1);
                {
                #pragma unroll
                    for (int j=0; j < kWarpTileSeqLenK; j++) {
                        HMMA16816(R_S[0][j][0], R_S[0][j][1], R_Q[0][0][0], R_Q[0][0][1],
                                R_Q[0][0][2], R_Q[0][0][3],
                                R_K[j][0], R_K[j][1], R_S[0][j][0], R_S[0][j][1]);
                    }
                }
            }
        }
        __syncthreads();
        // <w/o Prefetch V g2s>: If kCanPrefetchKVg2s is not enable,
        // we will load V g2s here, before rowmax and rowsum.
        if constexpr(!kCanPrefetchKVg2s) {
            load_gmem_V_Bc_offset = tile_K_seqlen * Bc; // (0-3)*64 = 0,64,128,192...
            int load_gmem_V_Bc = load_gmem_V_Bc_offset + load_smem_V_Bc;
            int load_gmem_V_d = load_smem_V_d;
            int load_gmem_V_addr = (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
            uint32_t load_smem_V_ptr = (smem_V_base_ptr +
                        (load_smem_V_Bc * (kHeadDim + kPadV) + load_smem_V_d) * sizeof(half));
        #pragma unroll
            for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i+=8) {
                CP_ASYNC_CG(load_smem_V_ptr + i * 2; &V[load_gmem_V_addr + i]; 16);
            }
            CP_ASYNC_COMMIT_GROUP();
        }
        // prefetch V s2r: Load V tile from smem -> regs, before P@V.
        if constexpr(kCanPrefetchKVg2s) {
            if ((tile_K_seqlen + 1) < Tc) {
                load_gmem_K_Bc_offset = (tile_K_seqlen + 1) * Bc; // (0-3)*64 = 0,64,128,192...
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
            }
        }

        // Online safe softmax, warp/block reduce max/sum, row wise
        float lane_row_max_new[kWarpTileSeqLenQ][2];
        float lane_row_sum_new[kWarpTileSeqLenQ][2];
        fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_max_new, -INFINITY);
        fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_sum_new, 0.0f);

        static_assert(kWarpTileSeqLenQ == 1);
        {
        #pragma unroll
            for (int j = 0; j < kWarpTileSeqLenK; j++) {
                half *t_hptr_S_0_1 = reinterpret_cast<half *>(&(R_S[0][j][0]));
                // row max after S = Q@K^T / scale
                float top_max_0 = __half2float(__hmax(t_hptr_S_0_1[0], t_hptr_S_0_1[1])) * scale;
                float top_max_1 = __half2float(__hmax(t_hptr_S_0_1[2], t_hptr_S_0_1[3])) * scale;
                lane_row_max_new[0][0] = fmaxf(lane_row_max_new[0][0], top_max_0);
                lane_row_max_new[0][1] = fmaxf(lane_row_max_new[0][1], top_max_1);
            }

            // warp reduce max
            lane_row_max_new[0][0] = warp_reduce_max<float, 4>(lane_row_max_new[0][0]);
            lane_row_max_new[0][1] = warp_reduce_max<float, 4>(lane_row_max_new[0][1]);
        }
        static_assert(kWarpTileSeqLenQ == 1);
        {
            float block_row_max_new_0 = lane_row_max_new[0][0];
            float block_row_max_new_1 = lane_row_max_new[0][1];
            float block_row_max_old_0 = lane_block_row_max_old[0][0];
            float block_row_max_old_1 = lane_block_row_max_old[0][1];
            block_row_new_max_0 = max(block_row_max_old_0, block_row_max_new_0);
            block_row_new_max_1 = max(block_row_max_old_1, block_row_max_new_1);

        #pragma unroll
            for (int j=0; j < kWarpTileSeqLenK; j++) {
                half *t_hptr_S_0_1 = reinterpret_cast<half *>(&(R_S[0][j][0]));
                // P = Exp(S - m_new), fmaf(x, y, z) = x * y + z;
                float4 t_reg_S_0_1;
                t_reg_S_0_1.x = __expf(__fmaf_rn(__half2float(t_hptr_S_0_1[0]) * scale - block_row_new_max_0));
                t_reg_S_0_1.y = __expf(__fmaf_rn(__half2float(t_hptr_S_0_1[1]) * scale - block_row_new_max_0));
                t_reg_S_0_1.z = __expf(__fmaf_rn(__half2float(t_hptr_S_0_1[2]) * scale - block_row_new_max_1));
                t_reg_S_0_1.w = __expf(__fmaf_rn(__half2float(t_hptr_S_0_1[3]) * scale - block_row_new_max_1));
                lane_row_sum_new[0][0] += (t_reg_S_0_1.x + t_reg_S_0_1.y);
                lane_row_sum_new[0][1] += (t_reg_S_0_1.z + t_reg_S_0_1.w);
                // Update R_S for P[Br,Bc] = Exp(S-m), point wise.
                t_hptr_S_0_1[0] = __float2half_rn(t_reg_S_0_1.x);
                t_hptr_S_0_1[1] = __float2half_rn(t_reg_S_0_1.y);
                t_hptr_S_0_1[2] = __float2half_rn(t_reg_S_0_1.z);
                t_hptr_S_0_1[3] = __float2half_rn(t_reg_S_0_1.w);
            }
            lane_row_sum_new[0][0] = warp_reduce_sum<float, 4>(lane_row_sum_new[0][0]);
            lane_row_sum_new[0][1] = warp_reduce_sum<float, 4>(lane_row_sum_new[0][1]);
        }
        if constexpr(kCanPrefetchKVg2s) {
            if (tile_K_seqlen + 1 < Tc) {
                CP_ASYNC_WAIT_GROUP(1);     // we have send V & K g2s, wait V and let K async.
            } else {
                CP_ASYNC_WAIT_GROUP(0);    // we have only send V g2s.
            }
        } else {
            CP_ASYNC_WAIT_GROUP(0);
        }
        __syncthreads();

        // hgemm in registers
        fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV, 2>(R_O, 0);
    #pragma unroll
        for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); tile_V_Bc++) {
        #pragma unroll
            for (int j =0; j < kWarpTileHeadDimV; j++) {
                int warp_smem_V_Bd = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
                int lane_smem_V_Bc = tile_V_Bc * kMmaAtomK + lane_id % 16;
                int lane_smem_V_d = warp_smem_V_Bd;
                uint32_t lane_smem_V_ptr = (smem_V_base_ptr +
                    (lane_smem_V_Bc * (kHeadDim + kPadV) + lane_smem_V_d) * sizeof(half));
                LDMATRIX_X2_T(R_V[j][0], R_V[j][1], lane_smem_V_ptr);
            }

            int w = tile_V_Bc * 2;
            static_assert(kWarpTileSeqLenP == 1);
            {
            #pragma unroll
                for (int j=0; j<kWarpTileHeadDimV; j++) {
                    HMMA16816(R_O[0][j][0], R_O[0][j][1], R_S[0][w][0], R_S[0][w][1],
                        R_S[0][w + 1][0], R_S[0][w + 1][1], R_V[j][0], R_V[j][1],
                        R_O[0][j][0], R_O[0][j][1]);
                }
            }
        }
        __syncthreads();

        static_assert(kWarpTileSeqLenP == 1);
        {
            float block_row_max_new_0 = lane_row_max_new[0][0];
            float block_row_max_new_1 = lane_row_max_new[0][1];
            float block_row_sum_new_0 = lane_row_sum_new[0][0];
            float block_row_sum_new_1 = lane_row_sum_new[0][1];

            float block_row_max_old_0 = lane_block_row_max_old[0][0];
            float block_row_max_old_1 = lane_block_row_max_old[0][1];
            block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
            block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);
            // Avoid inf value while using m_old for rescaling O.
            block_row_max_old_0 = (tile_K_seqlen > 0) ? block_row_max_old_0 : block_row_max_new_0;
            block_row_max_old_1 = (tile_K_seqlen > 0) ? block_row_max_old_1 : block_row_max_new_1;

            // rescale factor for O and l, exp(m_old - m)
            float rescale_o_factor_0 = __expf(block_row_max_old_0 - block_row_max_new_0);
            float rescale_o_factor_1 = __expf(block_row_max_old_1 - block_row_max_new_1);

        // 0. Rescale O: Online rescaling O each tile_K_seqlen step, need m_new, m_old.
        // m = max(m_old, m_new), O_new[Br,d] = exp(m_old - m) * O_old + P@V
        #pragma unroll
            for (int j = 0; j < kWarpTileHeadDimV; j++) {
                half *t_hptr_O_0_1 = reinterpret_cast<half *>(&(R_O[o][j][0]));
                if constexpr(kOStorageAccFloat32) {
                    float *t_fptr_D_0_1 = reinterpret_cast<float *>(&(R_D[0][j][0]));
                    t_fptr_D_0_1[0] = __fmaf_rn(rescale_o_factor_0, t_fptr_D_0_1[0], __half2float(t_hptr_O_0_1[0]));
                    t_fptr_D_0_1[1] = __fmaf_rn(rescale_o_factor_0, t_fptr_D_0_1[1], __half2float(t_hptr_O_0_1[1]));
                    t_fptr_D_0_1[2] = __fmaf_rn(rescale_o_factor_1, t_fptr_D_0_1[2], __half2float(t_hptr_O_0_1[2]));
                    t_fptr_D_0_1[3] = __fmaf_rn(rescale_o_factor_1, t_fptr_D_0_1[3], __half2float(t_hptr_O_0_1[3]));
                } else {
                    half *t_hptr_D_0_1 = reinterpret_cast<half *>(&(R_D[0][j][0]));
                    t_hptr_D_0_1[0] = __float2half_rn(__fmaf_rn(rescale_o_factor_0, __half2float(t_hptr_D_0_1[0]), __half2float(t_hptr_O_0_1[0])));
                    t_hptr_D_0_1[1] = __float2half_rn(__fmaf_rn(rescale_o_factor_0, __half2float(t_hptr_D_0_1[1]), __half2float(t_hptr_O_0_1[1])));
                    t_hptr_D_0_1[2] = __float2half_rn(__fmaf_rn(rescale_o_factor_1, __half2float(t_hptr_D_0_1[2]), __half2float(t_hptr_O_0_1[2])));
                    t_hptr_D_0_1[3] = __float2half_rn(__fmaf_rn(rescale_o_factor_1, __half2float(t_hptr_D_0_1[3]), __half2float(t_hptr_O_0_1[3])));
                }
            }

            float block_row_sum_old_0 = lane_block_row_sum_old[0][0];
            float block_row_sum_old_1 = lane_block_row_sum_old[0][1];
            lane_block_row_sum_old[0][0] = (__fmaf_rn(rescale_factor_0, block_row_sum_old_0, block_row_sum_new_0));
            lane_block_row_sum_old[0][1] = (__fmaf_rn(rescale_factor_1, block_row_sum_old_1, block_row_sum_new_1));
            lane_block_row_max_old[0][0] = block_row_max_new_0;
            lane_block_row_max_old[0][1] = block_row_max_new_1;
        }
        if constexpr(kCanPrefetchKVg2s) {
            if (tile_K_seqlen + 1 < Tc) {
                CP_ASYNC_WAIT_GROUP(0);
                __syncthreads();
            }
        }
    }
    __syncthreads();

    // Finaly, we still have to rescale O once more.
    // O_output(D) = ( 1/l_final ) * O_final (FA2 paper)
    // NOTE: Here, we choose to reuse R_O as final output
    // in order to reduce regs usage.

    static_assert(kWarpTileSeqLenP == 1);
    {
        float rescale_factor_0 = __frcp_rn(lane_block_row_sum_old[0][0]);
        float rescale_factor_1 = __frcp_rn(lane_block_row_sum_old[0][1]);
    #pragma unroll
        for (int j=0; j < kWarpTileHeadDimV; j++) {
            if constexpr(kOStorageAccFloat32) {
                float *t_fptr_D_0_1 = reinterpret_cast<float *>(&(R_D[0][j][0]));
                half *t_hptr_D_0_1 = reinterpret_cast<half *>(&(R_D[0][j][0]));
                t_hptr_D_0_1[0] = __float2half_rn(rescale_factor_0 * t_fptr_D_0_1[0]);
                t_hptr_D_0_1[1] = __float2half_rn(rescale_factor_0 * t_fptr_D_0_1[1]);
                t_hptr_D_0_1[2] = __float2half_rn(rescale_factor_1 * t_fptr_D_0_1[2]);
                t_hptr_D_0_1[3] = __float2half_rn(rescale_factor_1 * t_fptr_D_0_1[3]);
            } else {
                half *t_hptr_D_0_1 = reinterpret_cast<half *>(&(R_D[0][j][0]));
                t_hptr_D_0_1[0] = __float2half_rn(rescale_factor_0 * __half2float(t_hptr_D_0_1[0]));
                t_hptr_D_0_1[1] = __float2half_rn(rescale_factor_0 * __half2float(t_hptr_D_0_1[1]));
                t_hptr_D_0_1[2] = __float2half_rn(rescale_factor_1 * __half2float(t_hptr_D_0_1[2]));
                t_hptr_D_0_1[3] = __float2half_rn(rescale_factor_1 * __half2float(t_hptr_D_0_1[3]));
            }
        }
    }

    // Store O(D): Write O[Br,d] from regs -> gmem, collective store
    static_assert(kWarpTileSeqLenP == 1);
    {
    #pragma unroll
        for (int j = 0; j < kWarpTileHeadDimV; j++) {
            if constexpr(kCanPrefetchQs2r && kNumPrefetchQs2r > 1) {
                R_Q[0][0][0] = R_D[0][j][0];
                R_Q[1][0][0] = R_D[0][j][1];
                R_Q[0][0][1] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 1, 4);
                R_Q[0][0][2] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 2, 4);
                R_Q[0][0][3] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 3, 4);
                R_Q[1][0][1] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 1, 4);
                R_Q[1][0][2] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 2, 4);
                R_Q[1][0][3] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 3, 4);

                if (lane_id % 4 == 0) {
                    int store_warp_regs_O_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenP) + 0 * kMmaAtomM;
                    int store_lane_gmem_O_Br = O_tile_id * Br + store_warp_regs_O_Br + lane_id / 4;
                    int store_warp_regs_O_d = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
                    int store_lane_gmem_O_d = store_warp_regs_O_d;
                    int store_gmem_O_addr_0 =
                        (O_gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim +
                        store_lane_gmem_O_d);
                    int store_gmem_O_addr_1 =
                        (O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim +
                        store_lane_gmem_O_d);
                    LDST128BITS(O[store_gmem_O_addr_0]) = LDST128BITS(R_Q[0][0][0]);
                    LDST128BITS(O[store_gmem_O_addr_1]) = LDST128BITS(R_Q[1][0][0]);
                }
            } else {
                uint32_t R_Z[2][4];
                R_Z[0][0] = R_D[0][j][0];
                R_Z[1][0] = R_D[0][j][1]; // warp_size 4
                R_Z[0][1] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 1, 4);
                R_Z[0][2] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 2, 4);
                R_Z[0][3] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 3, 4);
                R_Z[1][1] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 1, 4);
                R_Z[1][2] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 2, 4);
                R_Z[1][3] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 3, 4);
                if (lane_id % 4 == 0) {
                int store_warp_regs_O_Br =
                    warp_QP * (kMmaAtomM * kWarpTileSeqLenP) + 0 * kMmaAtomM;
                int store_lane_gmem_O_Br =
                    O_tile_id * Br + store_warp_regs_O_Br + lane_id / 4; // 0~7
                int store_warp_regs_O_d =
                    warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
                int store_lane_gmem_O_d = store_warp_regs_O_d;
                int store_gmem_O_addr_0 =
                    (O_gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim +
                    store_lane_gmem_O_d);
                int store_gmem_O_addr_1 =
                    (O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim +
                    store_lane_gmem_O_d);
                LDST128BITS(O[store_gmem_O_addr_0]) = LDST128BITS(R_Z[0][0]);
                LDST128BITS(O[store_gmem_O_addr_1]) = LDST128BITS(R_Z[1][0]);
            }
        }
    }
}
}

template <const int kHeadDim, const int kStage>
void launch_flash_attn_mma_stages_split_q_shared_kv(torch::Tensor Q,
                                                    torch::Tensor K,
                                                    torch::Tensor V,
                                                    torch::Tensor O) {
    constexpr int kMmaAtomM = 16;
    constexpr int kMmaAtomN = 8;
    constexpr int kMmaAtomK = 16;
#ifdef BUILD_FLASH_ATTN_MMA_L20
    constexpr int kMmaTileSeqLenQ = 4;
    constexpr int kMmaTileSeqLenK = 1;
    constexpr int kMmaTileSeqLenP = 4;
    constexpr int kMmaTileHeadDimV = 1;
    constexpr int kWarpTileSeqLenQ = 1;
    constexpr int kWarpTileSeqLenK = (kStage > 1) ? 4 : 8;
    constexpr int kWarpTileSeqLenP = 1;
#else
    constexpr int kMmaTileSeqLenQ = (kHeadDim < 128) ? 8 : 8;
    constexpr int kMmaTileSeqLenK = 1;
    constexpr int kMmaTileSeqLenP = (kHeadDim < 128) ? 8 : 8;
    constexpr int kMmaTileHeadDimV = 1;
    constexpr int kWarpTileSeqLenQ = 1;
    constexpr int kWarpTileSeqLenK = (kHeadDim < 128) ? 8 : 4;
    constexpr int kWarpTileSeqLenP = 1;
#endif
    constexpr int kWarpTileHeadDimV = (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV));
    constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
    constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK; //  8*1*8=64
    constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK; // 32*4*1=128, num threads
    constexpr int kPadQ = 8;
    constexpr int kPadK = 8;
    constexpr int kPadV = 8;
    constexpr int kOStorageAccFloat32 = (kHeadDim < 256) ? 1 : 0;

    const int Q_tile_size = (Br * (kHeadDim + kPadQ));
    const int K_tile_size = (Bc * (kHeadDim + kPadK));
    const int V_tile_size = (Bc * (kHeadDim + kPadV));
    const int smem_max_size = (Q_tile_size + kStage * max(K_tile_size, V_tile_size)) * sizeof(half);
    const int QKV_batch = Q.size(0);
    const int QKV_head = Q.size(1);
    const int QKV_seqlen = Q.size(2);
    assert(QKV_seqlen % max(Br, Bc) == 0);

    dim3 grid(div_ceil(QKV_seqlen, Br), QKV_batch * QKV_head);
    dim3 block(kNumThreads);

    cudaFuncSetAttribute(
      flash_attn_mma_stages_split_q_shared_kv_kernel<
          kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
          kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ,
          kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
          kOStorageAccFloat32, kStage, kPadQ, kPadK, kPadV>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      // kMaxSramPerBlock
      98304);

    flash_attn_mma_stages_split_q_shared_kv_kernel<
        kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
        kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ,
        kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
        kOStorageAccFloat32, kStage, kPadQ, kPadK, kPadV>
        <<<grid, block, smem_max_size>>>(reinterpret_cast<half *>(Q.data_ptr()),
                                        reinterpret_cast<half *>(K.data_ptr()),
                                        reinterpret_cast<half *>(V.data_ptr()),
                                        reinterpret_cast<half *>(O.data_ptr()),
                                        QKV_seqlen, QKV_head);
}

void flash_attn_mma_stages_split_q_shared_kv(torch::Tensor Q, torch::Tensor K,
                    torch::Tensor V, torch::Tensor O, int stages) {
    CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf)
    const int d = Q.size(2);    // B, H, N, d

    if (stages > 1) {
        switch (d) {
        case 32:
            launch_flash_attn_mma_stages_split_q_shared_kv<32, 2>(Q, K, V, O);
            break;
        case 64:
            launch_flash_attn_mma_stages_split_q_shared_kv<64, 2>(Q, K, V, O);
            break;
        case 96:
            launch_flash_attn_mma_stages_split_q_shared_kv<96, 2>(Q, K, V, O);
            break;
        case 128:
            launch_flash_attn_mma_stages_split_q_shared_kv<128, 2>(Q, K, V, O);
            break;
        default:
            throw std::runtime_error("headdim not supported!");
            break;
        }
    } else {
        switch (d) {
        case 32:
            launch_flash_attn_mma_stages_split_q_shared_kv<32, 1>(Q, K, V, O);
            break;
        case 64:
            launch_flash_attn_mma_stages_split_q_shared_kv<64, 1>(Q, K, V, O);
            break;
        case 96:
            launch_flash_attn_mma_stages_split_q_shared_kv<96, 1>(Q, K, V, O);
            break;
        case 128:
            launch_flash_attn_mma_stages_split_q_shared_kv<128, 1>(Q, K, V, O);
            break;
        case 256:
            launch_flash_attn_mma_stages_split_q_shared_kv<256, 1>(Q, K, V, O);
            break;
        default:
        throw std::runtime_error("headdim not support!");
        break;
        }
    }
}