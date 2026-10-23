#include <utils.h>

// Step 1: Define Template Parameters
template <
    const int kHeadDim,
    const int kMmaAtomM,
    const int kMmaAtomN,
    const int kMmaAtomK,
    const int kMmaTileSeqLenQ,
    const int kMmaTileSeqLenK,
    const int kMmaTileSeqLenP,
    const int kMmaTileHeadDimV,
    const int kWarpTileSeqLenQ,
    const int kWarpTileSeqLenK,
    const int kWarpTileSeqLenP,
    const int kWarpTileHeadDimV,
    const int kOStorageAccFloat32,
    const int kStage, const int kPad>
// Step 2: Kernel Launch & Tile Setup
__global__ void __launch_bounds__(WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK)
    flash_attn_mma_stages_split_kv_kernel(half *Q, half *K, half *V, half *O, int QKV_seqlen, int QKV_head) {
    static_assert(kMmaAtomM == 16 && kMmaAtomN == 8 && kMmaAtomK == 16);
    static_assert(kMmaTileSeqLenQ == 2 && kMmaTileSeqLenK == 4);
    static_assert(kMmaTileSeqLenP == 2 && kMmaTileHeadDimV == 4);
    static_assert(kWarpTileSeqLenQ == 2 && kWarpTileSeqLenK == 2);
    static_assert(kWarpTileSeqLenP == 2 && kWarpTileHeadDimV == (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV)));
    static_assert(kOStorageAccFloat32 == 0 || kOStorageAccFloat32 == 1);
    static_assert(kStage > 0 && kStage < 3);
    static_assert(kPad >= 0 && kPad % 8 == 0);
    constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
    constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
    static_assert(Br >= Bc);
    constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
    const int Tc = div_ceil(QKV_seqlen, Bc);
    const float scale = 1.0f / sqrt((float)kHeadDim);

    // Step 3: Grid Mapping & Memory Offsets
    const int QKV_batch_id = blockIdx.y / QKV_head;
    const int QKV_head_id = blockIdx.y % QKV_head;
    const int Q_tile_id = blockIdx.x;
    const int O_tile_id = Q_tile_id;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_QP = warp_id % 2;    // 0, 1
    const int warp_KV = warp_id / 2;    // 0, 1, 2, 3

    const int Q_gmem_offset = ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) + 
                                QKV_head_id * QKV_seqlen * kHeadDim);
    const int K_gmem_offset = ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) +
                                QKV_head_id * QKV_seqlen * kHeadDim);
    const int V_gmem_offset = Q_gmem_offset;
    const int O_gmem_offset = Q_gmem_offset;

    // Step 4: Load & Organize Tiles in Shared Memory
    int load_smem_Q_Br = (tid / (kNumThreads / Br));
    int load_smem_Q_d = (tid % (kNumThreads / Br)) * (kHeadDim / (kNumThreads / Br));
    int load_smem_K_Bc = (tid / (kNumThreads / Bc));
    int load_smem_K_d = (tid % (kNumThreads / Bc)) * (kHeadDim / (kNumThreads / Bc));
    int load_smem_V_Bc = (tid / (kNumThreads / Bc));
    int load_smem_V_d = (tid % (kNumThreads / Bc)) * (kHeadDim / (kNumThreads / Bc));
    
    int load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br;
    if (load_gmem_Q_Br >= QKV_seqlen) return;
    int load_gmem_K_Bc_offset = 0;
    int load_gmem_V_Bc_offset = 0;

    extern __shared__ half smem[];
    constexpr int Q_tile_size = Br * (kHeadDim + kPad);
    constexpr int KV_tile_size = Bc * (kHeadDim + kPad);
    constexpr int S_tile_size = Br * (Bc + kPad);
    half *Q_tile_smem = smem;
    half *K_tile_smem = Q_tile_smem + Q_tile_size;
    half *V_tile_smem = K_tile_smem + kStage * KV_tile_size;
    half *S_tile_smem = V_tile_smem + KV_tile_size;

    uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
    uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
    uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);
    uint32_t smem_S_base_ptr = __cvta_generic_to_shared(S_tile_smem);

    float lane_block_row_max_old[kWarpTileSeqLenQ][2];
    float lane_block_row_sum_old[kWarpTileSeqLenQ][2];
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_max_old, -INFINITY);
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_sum_old, 0.0f);

    __shared__ float block_row_max_new_smem[Br][kWarpTileSeqLenK + 1];
    __shared__ float block_row_sum_new_smem[Br][kWarpTileSeqLenK + 1];

    // Step 5: Register Setup
    uint32_t R_Q[kWarpTileSeqLenQ][4];
    uint32_t R_K[kWarpTileSeqLenK][2];
    uint32_t R_V[kWarpTileHeadDimV][2];
    uint32_t R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][2];
    uint32_t R_O[kWarpTileSeqLenP][kWarpTileHeadDimV][2];
    uint32_t R_D[kWarpTileSeqLenP][kWarpTileHeadDimV][2];
    fill_3D_regs<float, kWarpTileSeqLenQ, kWarpTileSeqLenK, 2>(R_S, 0);
    fill_3D_regs<float, kWarpTileSeqLenP, kWarpTileHeadDimV, 2>(R_O, 0);
    fill_3D_regs<float, kWarpTileSeqLenP, kWarpTileHeadDimV, 2>(R_D, 0);

    // Step 6: Load Q / K into Shared Memory
    {
        int load_gmem_Q_d = load_smem_Q_d;
        int load_gmem_Q_addr = Q_gmem_offset + load_gmem_Q_Br * kHeadDim + load_gmem_Q_d;
        uint32_t load_smem_Q_ptr = (smem_Q_base_ptr + (load_smem_Q_Br * (kHeadDim + kPad) + load_smem_Q_d) * sizeof(half));
    #pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Br)); i += 8) {
            CP_ASYNC_CG(load_smem_Q_ptr + i * 2, &Q[load_gmem_Q_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
    }

    if constexpr(kStage > 1) {
    #pragma unroll
        for (int stage = 0; stage < (kStage - 1); stage++) {
            load_gmem_K_Bc_offset = stage * Bc;
            int load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
            int load_gmem_K_d = load_smem_K_d;
            int load_gmem_K_addr = (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
            uint32_t load_smem_K_ptr = (smem_K_base_ptr + (load_smem_K_Bc * (kHeadDim + kPad) + load_smem_K_d) * sizeof(half));
        #pragma unroll
            for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
                CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
            }
            CP_ASYNC_COMMIT_GROUP();
        }
    }
    if constexpr(kStage > 1) {
        CP_ASYNC_WAIT_GROUP(kStage - 2);
        __syncthreads();
    }

    // Step 7: Iterate Over K,V Tiles
#pragma unroll
    for (int tile_K_seqlen = 0; tile_K_seqlen < Tc; tile_K_seqlen++) {
        int smem_sel = (tile_K_seqlen) % kStage;
        int smem_sel_next = (tile_K_seqlen + (kStage - 1)) % kStage;

        if constexpr(kStage > 1) {
            {
                load_gmem_V_Bc_offset = tile_K_seqlen * Bc;
                int load_gmem_V_Bc = load_gmem_K_Bc_offset + load_smem_V_Bc;
                int load_gmem_V_d = load_smem_V_d;
                int load_gmem_V_addr = (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
                uint32_t load_smem_V_ptr = (smem_V_base_ptr + (load_smem_V_Bc * (kHeadDim + kPad) + load_smem_V_d) * sizeof(half));
            #pragma unroll
                for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
                    CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
                }
                CP_ASYNC_COMMIT_GROUP();
            }
            if ((tile_K_seqlen + 1) < Tc) {
                load_gmem_K_Bc_offset = (tile_K_seqlen + 1) * Bc;
                int load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
                int load_gmem_K_d = load_smem_K_d;
                int load_gmem_K_addr = K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d;
                uint32_t load_smem_K_ptr = (smem_K_base_ptr + (load_smem_K_Bc * (kHeadDim + kPad) + load_smem_K_d) * sizeof(half));
            #pragma unroll
                for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
                    CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
                }
                CP_ASYNC_COMMIT_GROUP();
            }
        } else {
            {
                load_gmem_K_Bc_offset = tile_K_seqlen * Bc;
                int load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
                int load_gmem_K_d = load_smem_K_d;
                int load_gmem_K_addr = K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d;
                uint32_t load_smem_K_ptr = (smem_K_base_ptr + (load_smem_K_Bc * (kHeadDim + kPad) + load_smem_K_d) * sizeof(half));
            #pragma unroll
                for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
                    CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
                }
                CP_ASYNC_COMMIT_GROUP();
            }
        // Then, prefetch curr V tile_K_seqlen [d,Bc] (no stages)
            {
                load_gmem_V_Bc_offset = tile_K_seqlen * Bc;
                int load_gmem_V_Bc = load_gmem_K_Bc_offset + load_smem_V_Bc;
                int load_gmem_V_d = load_smem_V_d;
                int load_gmem_V_addr = (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
                uint32_t load_smem_V_ptr = (smem_V_base_ptr + (load_smem_V_Bc * (kHeadDim + kPad) + load_smem_V_d) * sizeof(half));
            #pragma unroll
                for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
                    CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
                }
                CP_ASYNC_COMMIT_GROUP();
            }
            CP_ASYNC_WAIT_GROUP(1);
            __syncthreads();
        }

        fill_3D_regs<uint32_t, kWarpTileSeqLenQ, kWarpTileSeqLenK, 2>(R_S, 0);
    #pragma unroll
        for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); tile_K_d++) {
            // smem -> reg, load m16n8k16 smem Q/K
        #pragma unroll
            for (int i = 0; i < kWarpTileSeqLenQ; i++) {
                int warp_smem_Q_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
                int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;
                int lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8;
                uint32_t lane_smem_Q_ptr = (smem_Q_base_ptr + (lane_smem_Q_Br * (kHeadDim + kPad) + lane_smem_Q_d) * sizeof(half));
                LDMATRIX_X4(R_Q[i][0], R_Q[i][1], R_Q[i][2], R_Q[i][3], lane_smem_Q_ptr);
            }

        #pragma unroll
            for (int j = 0; j < kWarpTileSeqLenK; j++) {
                int warp_smem_K_Bc = warp_KV * (kMmaAtomK * kWarpTileSeqLenK) + j * kMmaAtomK;
                int lane_smem_K_Bc = warp_smem_K_Bc + lane_id % 16;
                int lane_smem_K_d = tile_K_d * kMmaAtomK + ((lane_id / 8) % 2) * 8;
                uint32_t lane_smem_K_ptr = (smem_K_base_ptr + (lane_smem_K_Bc * (kHeadDim + kPad) + lane_smem_K_d) * sizeof(half));
                LDMATRIX_X2(R_K[j][0], R_K[j][1], lane_smem_K_ptr);
            }

        #pragma unroll
            for (int i = 0; i < kWarpTileSeqLenQ; i++) {
        #pragma unroll
                for (int j = 0; j < kWarpTileSeqLenK; j++) {
                    HMMA16816(R_S[i][j][0], R_S[i][j][1], R_Q[i][0], R_Q[i][1], R_Q[i][2],
                    R_Q[i][3], R_K[j][0], R_K[j][1], R_S[i][j][0],
                    R_S[i][j][1]);
                }
            }
        }
        __syncthreads();

        float lane_row_max_new[kWarpTileSeqLenQ][2];
        float lane_row_sum_new[kWarpTileSeqLenQ][2];
        fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_max_new, -INFINITY);
        fill_2D_regs<float, lane_row_sum_new, 2>(lane_row_sum_new, 0.0f);

    #pragma unroll
        for (int i = 0; i < kWarpTileSeqLenQ; i++) {
    #pragma unroll
            for (int j = 0; j < kWarpTileSeqLenK; j++) {
                float2 t_reg_S_0 = __half22float2(R_S[i][j][0]);    //0-7 {c0, c1}
                float2 t_reg_S_1 = __half22float2(R_S[i][j][1]);    //8-15 {c2, c3}
                float top_max_0 = max(t_reg_S_0.x, t_reg_S_0.y) * scale;
                float top_max_1 = max(t_reg_S_1.x, t_reg_S_1.y) * scale;
                lane_row_max_new[i][0] = max(lane_row_max_new[i][0], top_max_0);
                lane_row_max_new[i][1] = max(lane_row_max_new[i][1], top_max_1);
            }
            // warp level reduce max
            lane_row_max_new[i][0] = warp_reduce_max<float, 4>(lane_row_max_new[i][0]);
            lane_row_max_new[i][1] = warp_reduce_max<float, 4>(lane_row_max_new[i][1]);

            if (lane_id % 4 == 0) {
                block_row_max_new_smem[warp_QP * 32 + i * 16 + 0 * 8 + (lane_id / 4)][warp_KV] = lane_row_max_new[i][0];
                block_row_max_new_smem[warp_QP * 32 + i * 16 + 1 * 8 + (lane_id / 4)][warp_KV] = lane_row_max_new[i][1];
            }
        }
        __syncthreads();

        float wrp_row_max_new = (block_row_max_new_smem[tid / kWarpTileSeqLenK][tid % kWarpTileSeqLenK]); //[0-63][0-4]
        float blk_row_max_new = warp_reduce_max<float, 4>(wrp_row_max_new);
        block_row_max_new_smem[tid / kWarpTileSeqLenK][tid % kWarpTileSeqLenK] = blk_row_max_new;
        __syncthreads();

        // Exp sum and mul scale_factor for [Br,Bc] tile, Thread -> Warp -> Block.
    #pragma unroll
        for (int i = 0; i < kWarpTileSeqLenQ; i++) {
            float block_row_max_new_0 = block_row_max_new_smem[warp_QP * 32 + i * 16 + 0 * 8 + (lane_id / 4)][0];
            float block_row_max_new_1 = block_row_max_new_smem[warp_QP * 32 + i * 16 + 1 * 8 + (lane_id / 4)][0];

            float block_row_max_old_0 = lane_block_row_max_old[i][0];
            float block_row_max_old_1 = lane_block_row_max_old[i][1];

            block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
            block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);

        #pragma unroll
            for (int j = 0; j < kWarpTileSeqLenK; j++) {
                float2 t_reg_S_0 = __half22float2(R_S[i][j][0]);
                float2 t_reg_S_1 = __half22float2(R_S[i][j][1]);

                t_reg_S_0.x = __expf(__fmaf_rn(t_reg_S_0.x, scale, -block_row_max_new_0));
                t_reg_S_0.y = __expf(__fmaf_rn(t_reg_S_0.y, scale, -block_row_max_new_0));
                t_reg_S_1.x = __expf(__fmaf_rn(t_reg_S_1.x, scale, -block_row_max_new_1));
                t_reg_S_1.y = __expf(__fmaf_rn(t_reg_S_1.y, scale, -block_row_max_new_1));

                lane_row_sum_new[i][0] += (t_reg_S_0.x + t_reg_S_0.y);
                lane_row_sum_new[i][1] += (t_reg_S_1.x + t_reg_S_1.y);
                HALF2(R_S[i][j][0]) = __float22half2_rn(t_reg_S_0);
                HALF2(R_S[i][j][1]) = __float22half2_rn(t_reg_S_1);
            }

            lane_row_sum_new[i][0] = warp_reduce_sum<float, 4>(lane_row_sum_new[i][0]);
            lane_row_sum_new[i][1] = warp_reduce_sum<float, 4>(lane_row_sum_new[i][1]);

            if (lane_id % 4 == 0) {
                block_row_sum_new_smem[warp_QP * 32 + i * 16 + 0 * 8 + (lane_id / 4)][warp_KV] = lane_row_sum_new[i][0];
                block_row_max_new_smem[warp_QP * 32 + i * 16 + 1 * 8 + (lane_id / 4)][warp_KV] = lane_row_sum_new[i][1];
            }
        }
        __syncthreads();

        float wrp_row_sum_new = (block_row_sum_new_smem[tid / kWarpTileSeqLenK][tid % kWarpTileSeqLenK]);
        float blk_row_sum_new = warp_reduce_sum<float, 4>(wrp_row_sum_new);
        block_row_sum_new_smem[tid / kWarpTileSeqLenK][tid % kWarpTileSeqLenK] = (blk_row_sum_new);
        __syncthreads();

        // Write R_P(R_S) to P_smem [Br,Bc]
    #pragma unroll
        for (int i = 0; i < kWarpTileSeqLenQ; ++i) {
    #pragma unroll
            for (int j = 0; j < kWarpTileSeqLenK; j++) {
                R_Q[0][0] = R_S[i][j][0];
                R_Q[1][0] = R_S[i][j][1];
                R_Q[0][1] = __shfl_sync((0xffffffff), R_S[i][j][0], lane_id + 1, 4);
                R_Q[0][2] = __shfl_sync((0xffffffff), R_S[i][j][0], lane_id + 2, 4);
                R_Q[0][3] = __shfl_sync((0xffffffff), R_S[i][j][0], lane_id + 3, 4);
                R_Q[1][1] = __shfl_sync((0xffffffff), R_S[i][j][1], lane_id + 1, 4);
                R_Q[1][1] = __shfl_sync((0xffffffff), R_S[i][j][1], lane_id + 2, 4);
                R_Q[1][3] = __shfl_sync((0xffffffff), R_S[i][j][1], lane_id + 3, 4);

                if (lane_id % 4 == 0) {
                    int store_warp_regs_S_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
                    int store_lane_smem_S_Br = store_warp_regs_S_Br + lane_id / 4;
                    int store_warp_regs_S_Bc = warp_KV * (kMmaAtomN + kWarpTileSeqLenK) + j * kMmaAtomN;
                    int store_lane_smem_S_Bc = store_warp_regs_S_Bc;
                    int store_smem_S_addr_0 = ((store_lane_smem_S_Br + 0) * (Bc + kPad) + store_lane_smem_S_Bc);
                    int store_smem_S_addr_1 = ((store_lane_smem_S_Br + 8) * (Bc + kPad) + store_lane_smem_S_Bc);
                    LDST128BITS(S_tile_smem[store_smem_S_addr_0]) = LDST128BITS(R_Q[0][0]);
                    LDST128BITS(S_tile_smem[store_smem_S_addr_1]) = LDST128BITS(R_Q[1][0]);
                }
            }
        }
        __syncthreads();

        if constexpr(kStage > 1) {
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
            for (int i = 0; i < kWarpTileSeqLenP; i++) {
                int warp_smem_S_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenP) + i * kMmaAtomM;
                int lane_smem_S_Br = warp_smem_S_Br + lane_id % 16;
                int lane_smem_S_Bc = tile_V_Bc * kMmaAtomK + (lane_id / 16) * 8;
                uint32_t lane_smem_S_ptr = (smem_S_base_ptr + (lane_smem_S_Br * (Bc + kPad) + lane_smem_S_Bc) * sizeof(half));
                LDMATRIX_X4(R_Q[i][0], R_Q[i][1], R_Q[i][2], R_Q[i][3],
                    lane_smem_S_ptr); 
            }
        #pragma unroll
            for (int j = 0; j < kWarpTileHeadDimV; j++) {
                int warp_smem_V_d = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
                int lane_smem_V_Bc = tile_V_Bc * kMmaAtomK + (lane_id % 16);
                int lane_smem_V_d = warp_smem_V_d;
                uint32_t lane_smem_V_ptr = (smem_V_base_ptr + (lane_smem_V_Bc * (kHeadDim + kPad) + lane_smem_V_d) * sizeof(half));
                LDMATRIX_X2_T(R_V[j][0], R_V[j][1], lane_smem_V_ptr);
            }

        #pragma unroll
            for (int i = 0; i < kWarpTileSeqLenP; i++) {
        #pragma unroll
                for (int j = 0; j < kWarpTileHeadDimV; j++) {
                    HMMA16816(R_O[i][j][0], R_O[i][j][1], R_Q[i][0], R_Q[i][1], R_Q[i][2],
                    R_Q[i][3], R_V[j][0], R_V[j][1], R_O[i][j][0],
                    R_O[i][j][1]);
                }
            }
        }
        __syncthreads();

        // rescale O
    #pragma unroll
        for (int i = 0; i < kWarpTileSeqLenP; i++) {
            float block_row_max_new_0 = block_row_max_new_smem[warp_QP * 32 + i * 16 + 0 * 8 + (lane_id / 4)][0];
            float block_row_max_new_1 = block_row_max_new_smem[warp_QP * 32 + i * 16 + 1 * 8 + (lane_id / 4)][0];
            float block_row_sum_new_0 = block_row_sum_new_smem[warp_QP * 32 + i * 16 + 0 * 8 + (lane_id / 4)][0];
            float block_row_sum_new_1 = block_row_sum_new_smem[warp_QP * 32 + i * 16 + 1 * 8 + (lane_id / 4)][0];

            float block_row_max_old_0 = lane_block_row_max_old[i][0];
            float block_row_max_old_1 = lane_block_row_max_old[i][1];

            block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
            block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);

            block_row_max_old_0 = (tile_K_seqlen > 0 ? block_row_max_old_0 : block_row_max_new_0);
            block_row_max_old_1 = (tile_K_seqlen > 0 ? block_row_max_old_1 : block_row_max_new_1);

            // rescale factor for O and l, exp(m_old - m)
            float rescale_o_factor_0 = __expf(block_row_max_old_0 - block_row_max_new_0);
            float rescale_o_factor_1 = __expf(block_row_max_old_1 - block_row_max_new_1);

        #pragma unroll
            for (int j = 0; j < kWarpTileHeadDimV; j++) {
                float2 t_reg_O_0 = __half22float2(HALF2(R_O[i][j][0]));
                float2 t_reg_O_1 = __half22float2(HALF2(R_O[i][j][1]));
                float2 t_reg_D_0 = __half22float2(HALF2(R_D[i][j][0]));
                float2 t_reg_D_1 = __half22float2(HALF2(R_D[i][j][1]));

                t_reg_D_0.x = __fmaf_rn(rescale_o_factor_0, t_reg_D_0.x, t_reg_O_0.x);
                t_reg_D_0.y = __fmaf_rn(rescale_o_factor_0, t_reg_D_0.y, t_reg_O_0.y);
                t_reg_D_1.x = __fmaf_rn(rescale_o_factor_1, t_reg_D_1.x, t_reg_O_1.x);
                t_reg_D_1.y = __fmaf_rn(rescale_o_factor_1, t_reg_D_1.y, t_reg_O_1.y);
                HALF2(R_D[i][j][0]) = __float22half2_rn(t_reg_D_0);
                HALF2(R_D[i][j][1]) = __float22half2_rn(t_reg_D_1);
            }

            float block_row_sum_old_0 = lane_block_row_sum_old[i][0];
            float block_row_sum_old_1 = lane_block_row_sum_old[i][1];

            lane_block_row_sum_old[i][0] = (__fmaf_rn(rescale_o_factor_0, block_row_sum_old_0, block_row_sum_new_0));
            lane_block_row_sum_old[i][1] = (__fmaf_rn(rescale_o_factor_1, block_row_sum_old_1, block_row_sum_new_1));

            lane_block_row_max_old[i][0] = block_row_max_new_0;
            lane_block_row_max_old[i][1] = block_row_max_new_1;
        }

        if constexpr(kStage > 1) {
            if ((tile_K_seqlen + 1) < Tc) {
                CP_ASYNC_WAIT_GROUP(0);
            }
            __syncthreads();
        }
    }
    __syncthreads();

#pragma unroll
    for (i = 0; i < kWarpTileSeqLenP; ++i) {
        float rescale_factor_0 = __frcp_rn(lane_block_row_sum_old[i][0]);
        float rescale_factor_1 = __frcp_rn(lane_block_row_sum_old[i][1]);
    #pragma unroll
        for (int j = 0; j < kWarpTileHeadDimV; ++j) {
            float2 t_reg_D_0 = __half22float2(HALF2(R_D[i][j][0]));
            float2 t_reg_D_1 = __half22float2(HALF2(R_D[i][j][1]));
            t_reg_D_0.x = rescale_factor_0 * t_reg_D_0.x;
            t_reg_D_0.y = rescale_factor_0 * t_reg_D_0.y;
            t_reg_D_1.x = rescale_factor_1 * t_reg_D_1.x;
            t_reg_D_1.y = rescale_factor_1 * t_reg_D_1.y;
            HALF2(R_D[i][j][0]) = __float22half2_rn(t_reg_D_0);
            HALF2(R_D[i][j][1]) = __float22half2_rn(t_reg_D_1);
        }
    }

    // store O(D)
#pragma unroll
    for (int i = 0; i < kWarpTileSeqLenP; ++i) {
#pragma unroll
        for (int j = 0; j < kWarpTileHeadDimV; ++j) {
            static_assert(kWarpTileSeqLenQ >= 2);
            R_Q[0][0] = R_D[i][j][0];
            R_Q[1][0] = R_D[i][j][1];
            R_Q[0][1] = __shfl_sync((0xffffffff), R_D[i][j][0], lane_id + 1, 4);
            R_Q[0][2] = __shfl_sync((0xffffffff), R_D[i][j][0], lane_id + 2, 4);
            R_Q[0][3] = __shfl_sync((0xffffffff), R_D[i][j][0], lane_id + 3, 4);
            R_Q[1][1] = __shfl_sync((0xffffffff), R_D[i][j][1], lane_id + 1, 4);
            R_Q[1][2] = __shfl_sync((0xffffffff), R_D[i][j][1], lane_id + 2, 4);
            R_Q[1][3] = __shfl_sync((0xffffffff), R_D[i][j][1], lane_id + 3, 4);

            if (lane_id % 4 == 0) {
                int store_warp_regs_O_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenP) + i * kMmaAtomM;
                int store_lane_gmem_O_Br = O_tile_id * Br + store_warp_regs_O_Br + lane_id / 4;
                int store_warp_regs_O_d = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
                int store_lane_gmem_O_d = store_warp_regs_O_d;
                int store_gmem_O_addr_0 = (O_gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim + store_lane_gmem_O_d);
                int store_gmem_O_addr_1 = (O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim + store_lane_gmem_O_d);
                LDST128BITS(O[store_gmem_O_addr_0]) = LDST128BITS(R_Q[0][0]);
                LDST128BITS(O[store_gmem_O_addr_1]) = LDST128BITS(R_Q[1][0]);
            }
        }
    }
}

template <const int kHeadDim, const int kStage>
void launch_flash_attn_mma_stages_split_kv(torch::Tensor Q, torch::Tensor K,
                                           torch::Tensor V, torch::Tensor O) {
  constexpr int kMmaAtomM = 16;
  constexpr int kMmaAtomN = 8;
  constexpr int kMmaAtomK = 16;
  constexpr int kMmaTileSeqLenQ = 2;
  constexpr int kMmaTileSeqLenP = 2;
  constexpr int kMmaTileSeqLenK = 4;
  constexpr int kMmaTileHeadDimV = 4;
  constexpr int kWarpTileSeqLenQ = 2;
  constexpr int kWarpTileSeqLenP = 2;
  constexpr int kWarpTileSeqLenK = 2;
  constexpr int kWarpTileHeadDimV = (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV));
  constexpr int Br =
      kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ; // 16*2*2=64
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK; // 8*4*2=64
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
  constexpr int kPad = 8;

  const int smem_max_size =
      ((Br * (kHeadDim + kPad)) + (kStage * Bc * (kHeadDim + kPad)) +
       (Bc * (kHeadDim + kPad)) + (Br * (Bc + kPad))) *
      sizeof(half);

  const int QKV_batch = Q.size(0);
  const int QKV_head = Q.size(1);
  const int QKV_seqlen = Q.size(2); // QKV_seqlen
  assert(QKV_seqlen % Bc == 0);     // multiple of Bc=64


  dim3 grid(div_ceil(QKV_seqlen, Br), QKV_batch * QKV_head);
  dim3 block(kNumThreads); // 4/8 warps per block

  cudaFuncSetAttribute(
      flash_attn_mma_stages_split_kv_kernel<
          kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
          kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ,
          kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV, kStage, kPad>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      // kMaxSramPerBlock
      98304);

  flash_attn_mma_stages_split_kv_kernel<
      kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
      kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ,
      kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV, kStage, kPad>
      <<<grid, block, smem_max_size>>>(reinterpret_cast<half *>(Q.data_ptr()),
                                       reinterpret_cast<half *>(K.data_ptr()),
                                       reinterpret_cast<half *>(V.data_ptr()),
                                       reinterpret_cast<half *>(O.data_ptr()),
                                       QKV_seqlen, QKV_head);
}

void flash_attn_mma_stages_split_kv(torch::Tensor Q, torch::Tensor K,
                                    torch::Tensor V, torch::Tensor O,
                                    int stages) {
  CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf) // Q [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf) // K [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf) // V [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf) // O [B,H,N,D]
  const int d = Q.size(3);                  // B, H, N, d

  if (stages > 1) {
    switch (d) {
    case 32:
      launch_flash_attn_mma_stages_split_kv<32, 2>(Q, K, V, O);
      break;
    case 64:
      launch_flash_attn_mma_stages_split_kv<64, 2>(Q, K, V, O);
      break;
    case 96:
      launch_flash_attn_mma_stages_split_kv<96, 2>(Q, K, V, O);
      break;
    case 128:
      launch_flash_attn_mma_stages_split_kv<128, 2>(Q, K, V, O);
      break;
    default:
      throw std::runtime_error("headdim not support!");
      break;
    }
  } else {
    switch (d) {
    case 32:
      launch_flash_attn_mma_stages_split_kv<32, 1>(Q, K, V, O);
      break;
    case 64:
      launch_flash_attn_mma_stages_split_kv<64, 1>(Q, K, V, O);
      break;
    case 96:
      launch_flash_attn_mma_stages_split_kv<96, 1>(Q, K, V, O);
      break;
    case 128:
      launch_flash_attn_mma_stages_split_kv<128, 1>(Q, K, V, O);
      break;
    default:
      throw std::runtime_error("headdim not support!");
      break;
    }
  }
}