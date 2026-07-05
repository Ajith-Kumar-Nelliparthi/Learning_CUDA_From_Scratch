#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <float.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <torch/extension.h>
#include <torch/types.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// fp 32
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int offset = kWarpSize >> 1; offset >= 1; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

template <const int NUM_THREADS = 256>
__global__ void dot_product_f32_f32_kernel(float *a, float *b, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];

    // keep the data in registers as much as possible
    float prod = [idx < N] ? a[idx] * b[idx] : 0.0f;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    // warp reduce
    prod = warp_reduce_sum_f32<WARP_SIZE>(prod);
    // write to shared memory
    if (lane_id == 0) {
        smem[warp_id] = prod;
    }
    __syncthreads();

    // reduce the shared memory
     prod = (lane_id < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp_id == 0)
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    if (tid == 0)
        atomicAdd(y, prod);
}

// dot product + vec4
template <const int NUM_THREADS = 256 / 4>
__global__ void dot_product_f32x4_f32_kernel(float *a, float *b, float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * blockDim.x + tid) * 4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];

    // change a and b to float4 to reduce the number of loads
    float4 reg_x = FLOAT4[a[idx]];
    float4 reg_y = FLOAT4[b[idx]];
    float prod = [idx < N] ? (reg_x.x * reg_y.x + reg_x.y * reg_y.y + reg_x.z * reg_y.z + reg_x.w * reg_y.w) : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    // warp reduce
    prod = warp_reduce_sum_f32<WARP_SIZE>(prod);
    // write to shared memory
    if (lane == 0) {
        smem[warp] = prod;
    }
    __syncthreads();

    // reduce the shared memory
    prod = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
    if (warp == 0)
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    if (tid == 0)
        atomicAdd(y, prod);
}

// fp 16
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16(half val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// fp 16 + fp 32
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {
    float val_f32 = __half2float(val);
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
    }
    return val_f32;
}

// fp 16 x fp 32
template <const int NUM_THREADS = 256>
__global__ void dot_product_f16_f32_kernel(half *a, half *b, float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];

    // keep the data in registers as much as possible
    half prod_f16 = [idx < N] ? __hmul(a[idx], b[idx]) : __float2half(0.0f);
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    // warp reduce
    float prod = warp_reduce_sum_f16_f32<WARP_SIZE>(prod_f16);
    // write to shared memory
    if (lane_id == 0) {
        smem[warp_id] = prod;
    }
    __syncthreads();

    // reduce the shared memory
    prod = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;
    if (warp_id == 0)
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    if (tid == 0)
        atomicAdd(y, prod);
}

// fp16x2
template <const int NUM_THREADS = 256 / 2>
__global__ void dot_product_f16x2_f32_kernel(half *a, half *b, float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * blockDim.x + tid) * 2;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];

    half2 reg_x = HALF2[a[idx]];
    half2 reg_y = HALF2[b[idx]];
    half prod_f16 = [idx < N] ? __hmul(reg_x.x, reg_y.x) + __hmul(reg_x.y, reg_y.y) : __float2half(0.0f);

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    // warp reduce
    float prod = warp_reduce_sum_f16_f32<WARP_SIZE>(prod_f16);
    // write to shared memory
    if (lane_id == 0) {
        smem[warp_id] = prod;
    }
    __syncthreads();

    // reduce the shared memory
    prod = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;
    if (warp_id == 0)
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    if (tid == 0)
        atomicAdd(y, prod);
}

// fp 16 x 8
template <const int NUM_THREADS = 256 / 8>
__global__ void dot_product_f16x8_pack_f32_kernel(half *a, half *b, float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * blockDim.x + tid) * 8;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];
    // temprorary register to hold the 8 half values
    half pack_a[8], pack_b[8];                         // 8 x 16 bits = 128 bits
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);      // load 128 bits from global memory to register
    LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);      // load 128 bits from global memory to register
    const half z = __float2half(0.0f);

    half prod_f16 = z;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        half2 v = __hmul2(HALF2(pack_a[i]), HALF2(pack_b[i]));
        prod_f16 += (((idx + i) < N) ? (v.x + v.y) : z);
    }

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    float prod = warp_reduce_sum_f16_f32<WARP_SIZE>(prod_f16);
    // write to shared memory
    if (lane_id == 0) {
        smem[warp_id] = prod;
    }
    __syncthreads();

    // reduce the shared memory
    prod = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;
    if (warp_id == 0)
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    if (tid == 0)
        atomicAdd(y, prod);
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define LANUCH_DOT_PROD_KERNEL(NT, packed_type, acc_type, element_type)        \
  dot_prod_##packed_type##_##acc_type##_kernel<(NT)>                           \
      <<<grid, block>>>(reinterpret_cast<element_type *>(a.data_ptr()),        \
                        reinterpret_cast<element_type *>(b.data_ptr()),        \
                        prod.data_ptr<float>(), N);

#define DISPATCH_DOT_PROD_KERNEL(K, packed_type, acc_type, element_type,       \
                                 n_elements)                                   \
  const int NT = (K) / (n_elements);                                           \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (NT) {                                                                \
  case 32:                                                                     \
    LANUCH_DOT_PROD_KERNEL(32, packed_type, acc_type, element_type)            \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_DOT_PROD_KERNEL(64, packed_type, acc_type, element_type)            \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_DOT_PROD_KERNEL(128, packed_type, acc_type, element_type)           \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_DOT_PROD_KERNEL(256, packed_type, acc_type, element_type)           \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_DOT_PROD_KERNEL(512, packed_type, acc_type, element_type)           \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_DOT_PROD_KERNEL(1024, packed_type, acc_type, element_type)          \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error(                                                  \
        "only support (K)/(n_elements): 32/64/128/256/512/1024");              \
    break;                                                                     \
  }

#define TORCH_BINDING_DOT_PROD(packed_type, acc_type, th_type, element_type,   \
                               n_elements)                                     \
  torch::Tensor dot_prod_##packed_type##_##acc_type(torch::Tensor a,           \
                                                    torch::Tensor b) {         \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                     \
    auto options =                                                             \
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0); \
    auto prod = torch::zeros({1}, options);                                    \
    const int ndim = a.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= a.size(i);                                                        \
      }                                                                        \
      dim3 block(256);                                                         \
      dim3 grid(((N + 256 - 1) / 256) / (n_elements));                         \
      dot_prod_##packed_type##_##acc_type##_kernel<256>                        \
          <<<grid, block>>>(reinterpret_cast<element_type *>(a.data_ptr()),    \
                            reinterpret_cast<element_type *>(b.data_ptr()),    \
                            prod.data_ptr<float>(), N);                        \
    } else {                                                                   \
      const int S = a.size(0);                                                 \
      const int K = a.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        DISPATCH_DOT_PROD_KERNEL(K, packed_type, acc_type, element_type,       \
                                 n_elements)                                   \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= a.size(i);                                                      \
        }                                                                      \
        dim3 block(256);                                                       \
        dim3 grid(((N + 256 - 1) / 256) / (n_elements));                       \
        dot_prod_##packed_type##_##acc_type##_kernel<256>                      \
            <<<grid, block>>>(reinterpret_cast<element_type *>(a.data_ptr()),  \
                              reinterpret_cast<element_type *>(b.data_ptr()),  \
                              prod.data_ptr<float>(), N);                      \
      }                                                                        \
    }                                                                          \
    return prod;                                                               \
  }

// packed_type, acc_type, th_type, element_type, n_elements_per_pack
TORCH_BINDING_DOT_PROD(f32, f32, torch::kFloat32, float, 1)
TORCH_BINDING_DOT_PROD(f32x4, f32, torch::kFloat32, float, 4)
TORCH_BINDING_DOT_PROD(f16, f32, torch::kHalf, half, 1)
TORCH_BINDING_DOT_PROD(f16x2, f32, torch::kHalf, half, 2)
TORCH_BINDING_DOT_PROD(f16x8_pack, f32, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(dot_product_f32_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_product_f32x4_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_product_f16_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_product_f16x2_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_product_f16x8_pack_f32)
}