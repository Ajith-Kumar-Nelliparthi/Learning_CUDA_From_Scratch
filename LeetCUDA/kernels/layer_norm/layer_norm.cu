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
#include <torch/types.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat16 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP 32
// warp reduce sum
template <const int kwarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = kwarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// block reduce sum
template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
    const int tid = threadIdx.x;
    const int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];

    val = warp_reduce_sum_f32<WARP_SIZE>(val);
    if (lane_id == 0) shared[warp_id] = val;
    __syncthreads();

    val = (lane_id < NUM_WARPS) ? shared[lane_id] : 0.0f;
    val = warp_reduce_sum_f32<NUM_WARPS>(val);
    return val;
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f32_kernel(float *x, float *y, float g, float b, int N, int K) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = blockIdx.x* blockDim.x + tid;
    const float epsilon = 1e-5f;

    __shared__ float s_mean;
    __shared__ float s_var;
    float value = (idx < N * K) ? x[idx] : 0.0f;
    float sum = block_reduce_sum_f32<NUM_THREADS>(value);
    if (tid == 0) s_mean = sum / (float)K;
    __syncthreads();

    // variance
    float variance = (value - s_mean) * (value - s_mean);
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    if (tid == 0) s_var = rsqrtf(variance / (float)K + epsilon);
    __syncthreads();

    if (idx < N * K) {
        y[idx] = ((value - s_mean) * s_var) * g + b;
    }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f32x4_kernel(float *x, float *y, float g, float b, int N, int K) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = (bid * blockDim.x + tid) * 4;
    const float epsilon = 1e-5f;

    __shared__ float s_mean;
    __shared__ float s_var;
    float4 reg_x = FLOAT4(x[idx]);
    float value = (idx < N * K) ? (reg_x.x + reg_x.y + reg_x.w + reg_x.z) : 0.0f;
    float sum = block_reduce_sum_f32<NUM_THREADS>(value);
    if (tid == 0) s_mean = sum / (float)K;
    __syncthreads();

    float4 reg_var;
    reg_var.x = reg_x.x - s_mean;
    reg_var.y = reg_x.y - s_mean;
    reg_var.w = reg_x.w - s_mean;
    reg_var.z = reg_x.z - s_mean;
    float variance = reg_var.x * reg_var.x + reg_var.y * reg_var.y + 
                    reg_var.w * reg_var.w + reg_var.z * reg_var.z;
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    if (tid == 0) s_var = rsqrtf(variance / (float)K + epsilon);
    __syncthreads();

    float4 reg_y;
    reg_y.x = reg_var.x * s_var * g + b;
    reg_y.y = reg_var.y * s_var * g + b;
    reg_y.w = reg_var.w * s_var * g + b;
    reg_y.z = reg_var.z * s_var * g + b;
    if (idx < N * K) {
        float4(y[idx]) = reg_y;
    }
}

// FP 16
// warp reduce sum : half
template <const int kwarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
#pragma unroll
    for (int mask = kwarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <const int kwarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {
    float val_f32 = __half2float(val);
#pragma unroll
    for (int mask = kwarpSize >> 1; mask >= 1; mask >>= 1) {
        val_f32 += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val_f32;
}

template <const int NUM_THREADS = 256>
__device__ half block_reduce_sum_f16_f16(half val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ half shared[NUM_WARPS];
    val = warp_reduce_sum_f16_f16<WARP_SIZE>(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();

    val = (lane < NUM_WARPS) ? shared[lane] : __float2half(0.0f);
    val = warp_reduce_sum_f16_f16<NUM_WARPS>(val);
    return val;
}

template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f16_f32(half val) {
    constexpr NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];
    float val_f32 = warp_reduce_sum_f16_f32<WARP_SIZE>(val);
    if (lane == 0) shared[warp] = val_f32;
    __syncthreads();

    val_f32 = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
    val_f32 = warp_reduce_sum_f32<NUM_WARPS>(val_f32);
    return val_f32;
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16_f16_kernel(half *x, half *y, float g, float b, int N, int K) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    const half epsilon = __float2half(1e-5f);
    const half g_ = __float2half(g);
    const half b_ = __float2half(b);
    const half k_ = __int2half_rn(K);
    
    // const int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ half s_mean;
    __shared__ half s_var;

    half value = (idx < N * K) ? x[idx] : __float2half(0.0f);
    half sum = block_reduce_sum_f16_f16<NUM_THREADS>(value);
    if (tid == 0) s_mean = sum / k_;
    __syncthreads();

    half variance = (value - s_mean) * (value - s_mean);
    variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
    if (tid == 0) s_var = hrsqrt(variance / k_ + epsilon);
    __syncthreads();

    if (idx < N * k) {
        y[idx] = __hfma((value - s_mean) * s_var, g_, b_);
    }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x2_f16_kernel(half *x, half *y, float g, float b, int N, int K) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = (bid * blockDim.x + tid) * 2;
    const half g_ = __float2half(g);
    const half b_ = __float2half(b);
    const half k_ = __float2half_rn(K);
    const half epsilon = __float2half(1e-5f);
    __shared__ half s_mean;
    __shared__ half s_var;
    half2 reg_x = HALF2(x[idx]);

    half value = (idx < N * K) ? (reg_x.x + reg_x.y) : __float2half(0.0f);
    half sum =block_reduce_sum_f16_f16<NUM_THREADS>(value);
    if (tid == 0) s_mean = sum / k_;
    __syncthreads();

    half2 reg_x_h;
    reg_x_h.x = (reg_x.x - s_mean);
    reg_x_h.y = reg_x.y - s_mean;
    half variance = reg_x_h.x * reg_x_h.x + reg_x_h.y * reg_x_h.y;
    variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
    if (tid == 0) s_var = hrsqrt(variance / k_ + epsilon);
    __syncthreads();

    if (idx < N * K) {
        half2 reg_y;
        reg_y.x = __hfma(reg_x_h.x * s_var, g_,b_);
        reg_y.y = __hfma(reg_x_h.y * s_var, g_, b_);
        HALF2(y[idx]) = reg_y;
    }
}

#define HALF2_SUM(reg, i)   \
    (((idx + (i)) < N * K) ? ((reg).x + (reg).y) : __float2half(0.0f))

#define HALF2_sUB(reg_x, reg_y) \
    (reg_y).x = (reg_x).x - s_mean; \
    (reg_y).y = (reg_x).y - s_mean;

#define HALF2_VARIANCE(reg, i) \
    (((idx + (i)) < N * K) / ((reg).x * (reg).x + (reg).y * (reg).y : __float2half(0.0f)));

#define HALF2_LAYER_NORM(reg_y, reg_x, g_, b_)                                 \
  (reg_y).x = __hfma((reg_x).x * s_variance, g_, b_);                          \
  (reg_y).y = __hfma((reg_x).y * s_variance, g_, b_);


template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x8_f16_kernel(half *x, half *y, float g, float b, int N, int K) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = (bid * blockDim.x + tid) * 8;
    const half epsilon = __float2half(1e-5f);
    const half g_ = __float2half(g);
    const half b_ = __float2half(b);
    const half k_ = __float2half_rn(K);

    __shared__ half s_mean;
    __shared__ half s_var;
    half2 reg_x_0 = HALF2(x[idx + 0]);
    half2 reg_x_1 = HALF2(x[idx + 2]);
    half2 reg_x_2 = HALF2(x[idx + 4]);
    half2 reg_x_3 = HALF2(x[idx + 6]);

    half value = HALF2_SUM(reg_x_0, 0);
    value += HALF2_SUM(reg_x_1, 2);
    value += HALF2_SUM(reg_x_2, 4);
    value += HALF2_SUM(reg_x_3, 6);

    half sum = block_reduce_sum_f16_f16<NUM_THREADS>(value);
    if (tid == 0) s_mean = sum / k_;
    __syncthreads();

    half2 reg_x_hat_0, reg_x_hat_1, reg_x_hat_2, reg_x_hat_3;
    HALF2_sUB(reg_x_hat_0, reg_x_0);
    HALF2_sUB(reg_x_hat_1, reg_x_1);
    HALF2_sUB(reg_x_hat_2, reg_x_2);
    HALF2_sUB(reg_x_hat_3, reg_x_3);

    half varince = HALF2_VARIANCE(reg_x_hat_0, 0);
    varince += HALF2_VARIANCE(reg_x_hat_1, 2);
    varince += HALF2_VARIANCE(reg_x_hat_2, 4);
    varince += HALF2_VARIANCE(reg_x_hat_3, 6);

    varince = block_reduce_sum_f16_f16<NUM_THREADS>(varince);
    if (tid == 0) s_var = hrsqrt(varince / k_ + epsilon);
    __syncthreads();

    half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
    HALF2_LAYER_NORM(reg_y_0, reg_x_hat_0, g_, b_);
    HALF2_LAYER_NORM(reg_y_1, reg_x_hat_1, g_, b_);
    HALF2_LAYER_NORM(reg_y_2, reg_x_hat_2, g_, b_);
    HALF2_LAYER_NORM(reg_y_3, reg_x_hat_3, g_, b_);

    if ((idx + 0) < N * K) {
        HALF2(y[idx + 0]) = reg_y_0;
    }
    if ((idx + 2) < N * K) {
        HALF2(y[idx + 2]) = reg_y_1;
    }
    if ((idx + 4) < N * K) {
        HALF2(y[idx + 4]) = reg_y_2;
    }
    if ((idx + 6) < N * K) {
        HALF2(y[idx + 6]) = reg_y_3;
    }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16_f32_kernel(half *x, half *y, float g, float b, int N, int K) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    const float epsilon = 1e-5f;

    __shared__ float s_mean;
    __shared__ float s_var;
    float value = (idx < N * K) ? __half2float(x[idx]) : 0.0f;
    float sum = block_reduce_sum_f32<NUM_THREADS>(value);
    if (tid == 0) s_mean = sum / (float)K;
    __syncthreads();

    float variance = (value - s_mean) * (value - s_mean);
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    if (tid == 0) s_var = rsqrtf(variance / (float)K + epsilon);
    __syncthreads();

    if (idx < N * K) {
        y[idx] = __float2half(__fmaf_rn(((value - s_mean) * s_var), g, b));
  }
}