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
    const int kStage,
    const int kPadQ,
    const int kPadK, const int kPadV>
__global__ void __launch_bounds__(WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK)
    flash_attn_mma_stages_split_q_shared_kv_acc_f32_kernel(half *Q, half *K, half *V, half *O, int QKV_seqlen, int QKV_head) {
    static_assert(kMmaAtomM == 16 && kMmaAtomN == 8 && kMmaAtomK == 16);
    static_assert(kMmaTileSeqLenQ <= 8 && kMmaTileSeqLenK == 1);
    static_assert(kMmaTileSeqLenP <= 8 && kMmaTileHeadDimV == 1);
    static_assert(kWarpTileSeqLenQ == 1 && kWarpTileSeqLenK <= 16);
    static_assert(kWarpTileSeqLenP == 1 && kWarpTileHeadDimV == (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV)));
    static_assert(kOStorageAccFloat32 == 0 || kOStorageAccFloat32 == 1);
    static_assert(kStage < 3 && kStage > 0);
    static_assert(kPadQ >= 0 && kPadQ % 8 == 0);
    static_assert(kPadK >= 0 && kPadK % 8 == 0);
    static_assert(kPadV >= 0 && kPadV % 8 == 0);
    constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
    constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
    static_assert(Br >= Bc);
    constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
    const int Tc = div_ceil(QKV_seqlen, Bc);
    const float scale = 1.0f / sqrt((float)kHeadDim);
}