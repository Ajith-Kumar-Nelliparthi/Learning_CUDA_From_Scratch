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

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x8_pack_f16_kernel(half *x, half *y, float g, float b, int N, int K) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = (bid * blockDim.x + tid) * 8;
    const half epsilon = __float2half(1e-5f);
    const half g_ = __float2half(g);
    const half b_ = __float2half(b);
    const half k_ = __float2half2_rn(K);
    const half z_ = __float2half(0.0f);

    __shared__ half s_mean;
    __shared__ half s_var;

    half pack_x[8], pack_y[8]; // 8x16 bits = 128 bits
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits

    half value = z_;
#pragma unroll
    for (int i = 0; i<8; ++i) {
        value += ((idx + i) < N * K ? pack_x[i] : z_);
    }
    half sum = block_reduce_sum_f16_f16<NUM_THREADS>(value);
    if (tid == 0) s_mean = sum / k_;
    __syncthreads();

    half var = z_;
#pragma unroll
    for (int i=0; i<8; i++) {
        half v_hat = pack_x[i] - s_mean;
        var += ((idx + i) < N * k ? v_hat * v_hat : z_);
    }
    var = block_reduce_sum_f16_f16<NUM_THREADS>(var);
    if (tid == 0) s_var = rsqrtf(var / k_ + epsilon);
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 8; i++) {
        pack_y[i] = __hfma((pack_x[i] - s_mean) * s_var, g_, b_);
    }
    if ((idx + 7) < N * K) {
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x8_pack_f32_kernel(half *x, half *y, float g, float b, int N, int K) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = (bid * blockDim.x + tid) * 8;
    const half epsilon = 1e-5f;

    __shared__ float s_mean;
    __shared__ float s_var;
    half pack_x[8], pack_y[8];
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

    float value = 0.0f;
#pragma unroll
    for (int i=0; i<8; ++i) {
        value += ((idx + i) < N * K ? __half2float(pack_x[i]) : 0.0f);
    }
    float sum = block_reduce_sum_f32<NUM_THREADS>(value);
    if (tid == 0) s_mean = sum / (float)K;
    __syncthreads();

    float var = 0.0f;
#pragma unroll
    for (int i=0; i<8; ++i) {
        float v_hat = __half2float(pack_x[i]) - s_mean;
        var += ((idx + i) < N * K ? v_hat * v_hat : 0.0f);
    }
    var = block_reduce_sum_f32<NUM_THREADS>(var);
    if (tid == 0) s_var = rsqrtf(var / (float)K + epsilon);
    __syncthreads();

#pragma unroll
    for (int i=0; i<8; ++i) {
        pack_y[i] = __float2half(__fmaf_rn(((__half2float(pack_x[i]) - s_mean) * s_var), g, b));
    }
    if ((idx + 7) < N * K) {
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                                       \
  assert((T1).dim() == (T2).dim());                                            \
  for (int i = 0; i < (T1).dim(); ++i) {                                       \
    if ((T2).size(i) != (T1).size(i)) {                                        \
      throw std::runtime_error("Tensor size mismatch!");                       \
    }                                                                          \
  }

// fp32
#define LANUCH_LAYER_NORM_F32_KERNEL(K)                                        \
  layer_norm_f32_kernel<(K)><<<grid, block>>>(                                 \
      reinterpret_cast<float *>(x.data_ptr()),                                 \
      reinterpret_cast<float *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F32_KERNEL(N, K)                                   \
  dim3 block((K));                                                             \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F32_KERNEL(64)                                           \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F32_KERNEL(128)                                          \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F32_KERNEL(256)                                          \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F32_KERNEL(512)                                          \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F32_KERNEL(1024)                                         \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/256/512/1024");           \
    break;                                                                     \
  }

#define LANUCH_LAYER_NORM_F32x4_KERNEL(K)                                      \
  layer_norm_f32x4_kernel<(K) / 4><<<grid, block>>>(                           \
      reinterpret_cast<float *>(x.data_ptr()),                                 \
      reinterpret_cast<float *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F32x4_KERNEL(N, K)                                 \
  dim3 block((K) / 4);                                                         \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F32x4_KERNEL(64) break;                                  \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F32x4_KERNEL(128) break;                                 \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F32x4_KERNEL(256) break;                                 \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F32x4_KERNEL(512) break;                                 \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F32x4_KERNEL(1024) break;                                \
  case 2048:                                                                   \
    LANUCH_LAYER_NORM_F32x4_KERNEL(2048) break;                                \
  case 4096:                                                                   \
    LANUCH_LAYER_NORM_F32x4_KERNEL(4096) break;                                \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/.../1024*4");             \
    break;                                                                     \
  }

// fp16
#define LANUCH_LAYER_NORM_F16F16_KERNEL(K)                                     \
  layer_norm_f16_f16_kernel<(K)>                                               \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F16F16_KERNEL(N, K)                                \
  dim3 block((K));                                                             \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16F16_KERNEL(64)                                        \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16F16_KERNEL(128)                                       \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16F16_KERNEL(256)                                       \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16F16_KERNEL(512)                                       \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16F16_KERNEL(1024)                                      \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/256/512/1024");           \
    break;                                                                     \
  }

#define LANUCH_LAYER_NORM_F16F32_KERNEL(K)                                     \
  layer_norm_f16_f32_kernel<(K)>                                               \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F16F32_KERNEL(N, K)                                \
  dim3 block((K));                                                             \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16F32_KERNEL(64)                                        \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16F32_KERNEL(128)                                       \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16F32_KERNEL(256)                                       \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16F32_KERNEL(512)                                       \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16F32_KERNEL(1024)                                      \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/256/512/1024");           \
    break;                                                                     \
  }

#define LANUCH_LAYER_NORM_F16x2F16_KERNEL(K)                                   \
  layer_norm_f16x2_f16_kernel<(K) / 2>                                         \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F16x2F16_KERNEL(N, K)                              \
  dim3 block((K) / 2);                                                         \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(64) break;                               \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(128) break;                              \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(256) break;                              \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(512) break;                              \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(1024) break;                             \
  case 2048:                                                                   \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(2048) break;                             \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/.../1024*2");             \
    break;                                                                     \
  }

#define LANUCH_LAYER_NORM_F16x8F16_KERNEL(K)                                   \
  layer_norm_f16x8_f16_kernel<(K) / 8>                                         \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F16x8F16_KERNEL(N, K)                              \
  dim3 block((K) / 8);                                                         \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(64) break;                               \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(128) break;                              \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(256) break;                              \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(512) break;                              \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(1024) break;                             \
  case 2048:                                                                   \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(2048) break;                             \
  case 4096:                                                                   \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(4096) break;                             \
  case 8192:                                                                   \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(8192) break;                             \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/.../1024*8");             \
    break;                                                                     \
  }

#define LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(K)                             \
  layer_norm_f16x8_pack_f16_kernel<(K) / 8>                                    \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(N, K)                        \
  dim3 block((K) / 8);                                                         \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(64) break;                         \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(128) break;                        \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(256) break;                        \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(512) break;                        \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(1024) break;                       \
  case 2048:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(2048) break;                       \
  case 4096:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(4096) break;                       \
  case 8192:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(8192) break;                       \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/.../1024*8");             \
    break;                                                                     \
  }

#define LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(K)                             \
  layer_norm_f16x8_pack_f32_kernel<(K) / 8>                                    \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(N, K)                        \
  dim3 block((K) / 8);                                                         \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(64) break;                         \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(128) break;                        \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(256) break;                        \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(512) break;                        \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(1024) break;                       \
  case 2048:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(2048) break;                       \
  case 4096:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(4096) break;                       \
  case 8192:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(8192) break;                       \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/.../1024*8");             \
    break;                                                                     \
  }

void layer_norm_f32(torch::Tensor x, torch::Tensor y, float g, float b) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_LAYER_NORM_F32_KERNEL(N, K)
}

void layer_norm_f32x4(torch::Tensor x, torch::Tensor y, float g, float b) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_LAYER_NORM_F32x4_KERNEL(N, K)
}

void layer_norm_f16_f16(torch::Tensor x, torch::Tensor y, float g, float b) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_LAYER_NORM_F16F16_KERNEL(N, K)
}

void layer_norm_f16x2_f16(torch::Tensor x, torch::Tensor y, float g, float b) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_LAYER_NORM_F16x2F16_KERNEL(N, K)
}

void layer_norm_f16x8_f16(torch::Tensor x, torch::Tensor y, float g, float b) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_LAYER_NORM_F16x8F16_KERNEL(N, K)
}

void layer_norm_f16x8_pack_f16(torch::Tensor x, torch::Tensor y, float g,
                               float b) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(N, K)
}

void layer_norm_f16x8_pack_f32(torch::Tensor x, torch::Tensor y, float g,
                               float b) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(N, K)
}

void layer_norm_f16_f32(torch::Tensor x, torch::Tensor y, float g, float b) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_LAYER_NORM_F16F32_KERNEL(N, K)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f32)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16_f16)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16_f32)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16x2_f16)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16x8_f16)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16x8_pack_f16)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16x8_pack_f32)
}