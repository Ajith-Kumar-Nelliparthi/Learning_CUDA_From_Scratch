#include <utils.h>

template <const int kHeaddim,
    const int kMmaAtomM,
    const int kMmaAtomN,
    const int kMmaAtomK,
    const int kMMaTileSeqLenQ,
    const int kMmaTileSeqLenK,
    const int kMmaTileSeqLenP,
    const int kMmaTileHeadDimV,
    const int kWarpTileSeqLenQ,
    const int kWarpTileSeqLenK,
    const int kWarpTileSeqLenP,
    const int kWarpTileHeadDimV,
    const int kOStorageAccFloat32,
    const int kStage,
    const int kPadQ,
    const int kPadK, const int kPadV>
__global__ void __launch_bounds__(WARP_SIZE * kMMaTileSeqLenQ * kMmaTileSeqLenK)
    flash_attn_mma_stages_split_q_shared_qkv_kernel(half *Q, half *K, half *V, half *O, int QKV_seqlen, int QKV_head) {
    static_assert(kMmaAtomM == 16 && kMmaAtomN == 8 && kMmaAtomK == 16);
    static_assert(kMMaTileSeqLenQ <= 8 && kMmaTileSeqLenK == 1);
    static_assert(kMmaTileSeqLenP <= 8 && kMmaTileHeadDimV == 1);
    static_assert(kWarpTileSeqLenQ == 1 && kWarpTileSeqLenK <= 16);
    static_assert(kWarpTileSeqLenP == 1 && kWarpTileHeadDimV == (kHeaddim / (kMmaAtomN * kMmaTileHeadDimV)))
    static_assert(kOStorageAccFloat32 == 0 || kOStorageAccFloat32 == 1);
    static_assert(kStage > 0 && kStage < 3);
    static_assert(kPadQ >= 0 && kPadQ % 8 == 0);
    static_assert(kPadK >= 0 && kPadK % 8 == 0);
    static_assert(kPadV >= 0 && kPadV % 8 == 0);
    constexpr int Br = kMmaAtomM * kMMaTileSeqLenQ * kWarpTileSeqLenQ;
    constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
    static_assert(Br >= Bc);
    constexpr int kNumThreads = WARP_SIZE * kMMaTileSeqLenQ * kMmaTileSeqLenK;
    const int Tc = div_ceil(QKV_seqlen, Bc);
    const float scale = 1.0f / sqrt((float)kHeaddim);

    const int QKV_batch_id = blockIdx.y / QKV_head;
    const int QKV_head_id = blockIdx.y % QKV_head;
    const int Q_tile_id = blockIdx.x;
    const int O_tile_id = Q_tile_id;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_QP = warp_id;
    const int warp_KV = 0;

    const int Q_gmem_offset = ((QKV_batch_id * QKV_head * QKV_seqlen * kHeaddim) + (QKV_head_id * QKV_seqlen * kHeaddim));
    const int K_gmem_offset = ((QKV_batch_id * QKV_head * QKV_seqlen * kHeaddim) + (QKV_head_id * QKV_seqlen * kHeaddim));
    const int V_gmem_offset = Q_gmem_offset;
    const int O_gmem_offset = Q_gmem_offset;

    int load_smem_Q_Br = (tid / (kNumThreads / Br));
    int load_smem_Q_d = (tid % (kNumThreads / Br)) * (kHeaddim / (kNumThreads / Br));
    int load_smem_K_Bc = (tid / (kNumThreads / Bc));
    int load_smem_K_d = (tid % (kNumThreads / Bc)) * (kHeaddim / (kNumThreads / Bc));
    int load_smem_V_Bc = (tid / (kNumThreads / Bc));
    int load_smem_V_d = (tid % (kNumThreads / Bc)) * (kHeaddim / (kNumThreads / Bc));
    int load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br;
    if (load_gmem_Q_Br >= QKV_seqlen) return;

    int load_gmem_K_Bc_offset = 0;
    int load_gmem_V_Bc_offset = 0;

    extern __shared__ half smem[];
    constexpr int Q_tile_size = Br * (kHeaddim + kPadQ);
    constexpr int K_tile_size = Bc * (kHeaddim + kPadK);
    constexpr int V_tile_size = Bc * (kHeaddim + kPadV);
    half *Q_tile_smem = smem;
    half *K_tile_smem = Q_tile_smem;
    half *V_tile_smem = Q_tile_smem;    // QKV share same smem

    uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
    uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
    uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

    float lane_block_row_max_old[kWarpTileSeqLenQ][2];
    float lane_block_row_sum_old[kWarpTileSeqLenQ][2];
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_max_old, -INFINITY);
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_sum_old, 0.0f);

    static_assert(kHeaddim <= 512, "shared_qkv only support head dim <= 512");
    static_assert(kHeaddim >= 32, "shared_qkv only support head dim >= 32");

    constexpr bool kCanPrefetchQs2r = ((kHeaddim / kMmaAtomK) <= 32);
    constexpr bool kCanPrefetchKVg2s = (((Q_tile_size / (K_tile_size > V_tile_size ? K_tile_size : V_tile_size)) >= 2) && (kStage >= 2));
    constexpr int kPrefetchKg2sSmemId = 0;
    constexpr int kPrefetchVg2sSmemId = (kCanPrefetchKVg2s ? 1 : 0);
    constexpr int kNumPrefetchQs2r = (kCanPrefetchQs2r) ? (kHeaddim / kMmaAtomK) : 1;
    uint32_t R_Q[kNumPrefetchQs2r][kWarpTileSeqLenQ][4];
    uint32_t R_K[kWarpTileSeqLenK][2];
    uint32_t R_V[kWarpTileSeqLenK][2];
    uint32_t R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][2];
    uint32_t R_O[kWarpTileSeqLenP][kWarpTileHeadDimV][2];
    uint32_t R_D[kWarpTileSeqLenP][kWarpTileHeadDimV][(kOStorageAccFloat32) ? 4 : 2];
    fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV, (kOStorageAccFloat32) ? 4 : 2>(R_D, 0);

    // load q from gmem to smem
    {
        int load_gmem_Q_d = load_smem_Q_d;
        int load_gmem_Q_addr = Q_gmem_offset + load_gmem_Q_Br * kHeaddim + load_gmem_Q_d;
        uint32_t load_smem_Q_ptr = (smem_Q_base_ptr + (load_smem_Q_Br * (kHeaddim + kPadQ) + load_smem_Q_d) * sizeof(half));
    #pragma unroll
        for (int i = 0; i < (kHeaddim / (kNumThreads / Br)); i+= 8) {
            CP_ASYNC_CG(load_smem_Q_ptr + i * 2, &Q[load_gmem_Q_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
    }

#pragma unroll 1
    for (int tile_K_seqlen = 0; tile_K_seqlen < Tc; tile_K_seqlen++) {
        static_assert(kCanPrefetchQs2r);
        if constexpr(kCanPrefetchQs2r) {
            if (tile_K_seqlen == 0) {
                CP_ASYNC_WAIT_GROUP(0);
                __syncthreads();

            #pragma unroll
                for (int tile_K_d = 0; tile_K_d < (kHeaddim / kMmaAtomK); tile_K_d++) {
                #pragma unroll
                    for (int i = 0; i < kWarpTileSeqLenQ; i++) {
                        int warp_smem_Q_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
                        int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;
                        int lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8;
                        uint32_t lane_smem_Q_ptr = (smem_Q_base_ptr + (lane_smem_Q_Br * (kHeaddim + kPadQ) + lane_smem_Q_d) * sizeof(half));
                        LDMATRIX_X4(R_Q[tile_K_d][i][0], R_Q[tile_K_d][i][1], 
                                    R_Q[tile_K_d][i][2], R_Q[tile_K_d][i][3],
                                    lane_smem_Q_ptr);
                    }
                }
                __syncthreads();
            }
        }

        // load k tile from gmem to smem
        if constexpr(kCanPrefetchKVg2s) {
            if (tile_K_seqlen == 0) {
                load_gmem_K_Bc_offset = tile_K_seqlen * Bc;
                int load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
                int load_gmem_K_d = load_smem_K_d;
                int load_gmem_K_addr = (K_gmem_offset + load_gmem_K_Bc * kHeaddim + load_gmem_K_d);
                uint32_t load_smem_K_ptr = (smem_K_base_ptr + (load_smem_K_Bc * (kHeaddim + kPadK) + load_smem_K_d) * sizeof(half));
            #pragma unroll
                for (int i = 0; i < (kHeaddim / (kNumThreads / Bc)); i += 8) {
                    CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
                }
                CP_ASYNC_COMMIT_GROUP();

                CP_ASYNC_WAIT_GROUP(0);
                __syncthreads();
            }
            // prefetch v tile from gmem to smem
            {
                load_gmem_V_Bc_offset = tile_K_seqlen * Bc;
                int load_gmem_V_Bc = load_gmem_V_Bc_offset + load_smem_V_Bc;
                int load_gmem_V_d = load_smem_V_d;
                int load_gmem_V_addr = (V_gmem_offset + load_gmem_V_Bc * kHeaddim + load_gmem_V_d);
                uint32_t load_smem_V_ptr = (smem_V_base_ptr + (load_smem_V_Bc * (kHeaddim + kPadV) + load_smem_V_d) * sizeof(half));
            #pragma unroll
                for (int i = 0; i < (kHeaddim / (kNumThreads / Bc)); i += 8) {
                    CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
                }
                CP_ASYNC_COMMIT_GROUP();
            }
        } else {
            load_gmem_K_Bc_offset = tile_K_seqlen * Bc;
            int load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
            int load_gmem_K_d = load_smem_K_d;
            int load_gmem_K_addr = (K_gmem_offset + load_gmem_K_Bc * kHeaddim + load_gmem_K_d);
            uint32_t load_smem_K_ptr = (smem_K_base_ptr + (load_smem_K_Bc * (kHeaddim + kPadK) + load_smem_K_d) * sizeof(half));
        #pragma unroll
            for (int i = 0; i < (kHeaddim / (kNumThreads / Bc)); i += 8) {
                CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
            }
            CP_ASYNC_COMMIT_GROUP();

            CP_ASYNC_WAIT_GROUP(0);
            __syncthreads();
        }
        
        fill_3D_regs<uint32_t, kWarpTileSeqLenQ, kWarpTileSeqLenK, 2>(R_S, 0);
    #pragma unroll
        for (int tile_K_d = 0; tile_K_d < (kHeaddim / kMmaAtomK); tile_K_d++) {
        #pragma unroll
            for (int j = 0; j < kWarpTileSeqLenK; j++) {
                int warp_smem_K_Bc = warp_KV * (kMmaAtomK * kWarpTileSeqLenK) + j * kMmaAtomK;
                int lane_smem_K_Bc = warp_smem_K_Bc + lane_id % 8;
                int lane_smem_K_d = tile_K_d * kMmaAtomK + ((lane_id / 8) % 2) * 8;
                uint32_t lane_smem_K_ptr = (smem_K_base_ptr + (lane_smem_K_Bc * (kHeaddim + kPadK) + lane_smem_K_d) * sizeof(half));
                LDMATRIX_X2_T(R_K[j][0], R_K[j][1], lane_smem_K_ptr);
            }

            // mma compute
            static_assert(kWarpTileSeqLenQ == 1);
            {
            #pragma unroll
                for (int j = 0; j < kWarpTileSeqLenK; j++) {
                    HMMA16816(R_S[0][j][0], R_S[0][j][1], R_Q[tile_K_d][0][0],
                            R_Q[tile_K_d][0][1], R_Q[tile_K_d][0][2],
                            R_Q[tile_K_d][0][3], R_K[j][0], R_K[j][1], R_S[0][j][0],
                            R_S[0][j][1]);
                }
            }
        }
        __syncthreads();

        // prefetch V from g2s
        if constexpr(!kCanPrefetchKVg2s) {
            load_gmem_V_Bc_offset = tile_K_seqlen * Bc;
            int load_gmem_V_Bc = load_gmem_V_Bc_offset + load_smem_V_Bc;
            int load_gmem_V_d = load_smem_V_d;
            int load_gmem_V_addr = (V_gmem_offset + load_gmem_V_Bc * kHeaddim + load_gmem_V_d);
            uint32_t load_smem_V_ptr = (smem_V_base_ptr + (load_smem_V_Bc * (kHeaddim + kPadV) + load_smem_V_d) * sizeof(half));
        #pragma unroll
            for (int i = 0; i < (kHeaddim / (kNumThreads / Bc)); i += 8) {
                CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
            }
            CP_ASYNC_COMMIT_GROUP();
        }

        // prefetch next K tile from g2s
        if constexpr(kCanPrefetchKVg2s) {
            if ((tile_K_seqlen + 1) < Tc) {
                load_gmem_K_Bc_offset = (tile_K_seqlen + 1) * Bc;
                int load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
                int load_gmem_K_d = load_smem_K_d;
                int load_gmem_K_addr = (K_gmem_offset + load_gmem_K_Bc * kHeaddim + load_gmem_K_d);
                uint32_t load_smem_K_ptr = (smem_K_base_ptr + (load_smem_K_Bc * (kHeaddim + kPadK) + load_smem_K_d) * sizeof(half));
            #pragma unroll
                for (int i = 0; i < (kHeaddim / (kNumThreads / Bc)); i += 8) {
                    CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
                }
                CP_ASYNC_COMMIT_GROUP();
            }
        }

        // online safe softmax
        float lane_row_max_new[kWarpTileSeqLenQ][2];
        float lane_row_sum_new[kWarpTileSeqLenQ][2];
        fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_max_new, -INFINITY);
        fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_sum_new, 0.0f);

        static_assert(kWarpTileSeqLenQ == 1);
        {
        #pragma unroll
            for (int j = 0; j < kWarpTileSeqLenK; j++) {
                half *t_hptr_S_0_1 = reinterpret_cast<half *>(&(R_S[0][j][0]));
                float top_max_0 = __half2float(__hmax(t_hptr_S_0_1[0], t_hptr_S_0_1[1])) * scale;
                float top_max_1 = __half2float(__hmax(t_hptr_S_0_1[2], t_hptr_S_0_1[3])) * scale;
                lane_row_max_new[0][0] = max(lane_row_max_new[0][0], top_max_0);
                lane_row_max_new[0][1] = max(lane_row_max_new[0][1], top_max_1);
            }
            // warp level reduce max
            lane_row_max_new[0][0] = warp_reduce_max<float, 4>(lane_row_max_new[0][0]);
            lane_row_max_new[0][1] = warp_reduce_max<float, 4>(lane_row_max_new[0][1]);
        }

        static_assert(kWarpTileSeqLenQ == 1);
        {
            float block_row_max_new_0 = lane_row_max_new[0][0];
            float block_row_max_new_1 = lane_row_max_new[0][1];

            float block_row_max_old_0 = lane_block_row_max_old[0][0];
            float block_row_max_old_1 = lane_block_row_max_old[0][1];

            block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
            block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);

        #pragma unroll
            for (int j = 0; j < kWarpTileSeqLenK; j++) {
                float *t_hptr_S_0_1 = reinterpret_cast<float *>(&(R_S[0][j][0]));
                float4 t_reg_S_0_1;
                t_reg_S_0_1.x = __expf(__fmaf_rn(__half2float(t_hptr_S_0_1[0]), scale, -block_row_max_new_0));
                t_reg_S_0_1.y = __expf(__fmaf_rn(__half2float(t_hptr_S_0_1[1]), scale, -block_row_max_new_0));
                t_reg_S_0_1.z = __expf(__fmaf_rn(__half2float(t_hptr_S_0_1[2]), scale, -block_row_max_new_1));
                t_reg_S_0_1.w = __expf(__fmaf_rn(__half2float(t_hptr_S_0_1[3]), scale, -block_row_max_new_1));
                lane_row_sum_new[0][0] += (t_reg_S_0_1.x + t_reg_S_0_1.y);
                lane_row_sum_new[0][1] += (t_reg_S_0_1.z + t_reg_S_0_1.w);

                t_hptr_S_0_1[0] = __float2half_rn(t_reg_S_0_1.x);
                t_hptr_S_0_1[1] = __float2half_rn(t_reg_S_0_1.y);
                t_hptr_S_0_1[2] = __float2half_rn(t_reg_S_0_1.z);
                t_hptr_S_0_1[3] = __float2half_rn(t_reg_S_0_1.w);
            }
            lane_row_sum_new[0][0] = warp_reduce_sum<float, 4>(lane_row_sum_new[0][0]);
            lane_row_sum_new[0][1] = warp_reduce_sum<float, 4>(lane_row_sum_new[0][1]);
        }

        if constexpr(kCanPrefetchKVg2s) {
            if ((tile_K_seqlen + 1) < Tc) {
                CP_ASYNC_WAIT_GROUP(1);
            } else {
                CP_ASYNC_WAIT_GROUP(0);
            } 
        } else {
            CP_ASYNC_WAIT_GROUP(0);
        }
        __syncthreads();

        fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV, 2>(R_O, 0);
    #pragma unroll
        for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); tile_V_Bc++) {
        #pragma unroll
            for (int j = 0; j < kWarpTileHeadDimV; j++) {
                int warp_smem_V_d = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
                int lane_smem_V_Bc = tile_V_Bc * kMmaAtomK + lane_id % 16;
                int lane_smem_V_d = warp_smem_V_d;
                uint32_t lane_smem_V_ptr = (smem_V_base_ptr + (lane_smem_V_Bc * (kHeaddim + kPadV) + lane_smem_V_d) * sizeof(half));
                LDMATRIX_X2_T(R_V[j][0], R_V[j][1], lane_smem_V_ptr);
            }

            int w  = tile_V_Bc * 2;
            static_assert(kWarpTileSeqLenP == 1);
            {
            #pragma unroll
                for (int j = 0; j < kWarpTileHeadDimV; j++) {
                    HMMA16816(R_O[0][j][0], R_O[0][j][1], R_S[0][w][0], R_S[0][w][1],
                            R_S[0][w + 1][0], R_S[0][w + 1][1], R_V[j][0], R_V[j][1],
                            R_O[0][j][0], R_O[0][j][1]);
                }
            }
        }
        __syncthreads();
    }
}