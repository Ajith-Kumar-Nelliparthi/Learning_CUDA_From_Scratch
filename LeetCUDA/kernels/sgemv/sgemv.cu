#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <torch/extension.h>
#include <torch/types.h>


#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define WARP_SIZE 32

// FP 32
// warp reduce sum
template <const int kwarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int mask = kwarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// Y = α*(A[MK] * X[K]) + β.y
__global__ void sgemv_k32_f32_kernel(float *a, float *x, float *y, int M, int K) {
    int tx = threadIdx.x;       // 0 - 31
    int ty = threadIdx.y;       // 0 - 4
    int bid = blockIdx.x;        // 0 - M/4
    int m = bid * blockDim.y + threadIdx.y;    // (0-M/4) * 4 + (0-3)
    int lane = tx % WARP_SIZE;   // 0-31
    if (m < M) {
        float sum = 0.0f;
        int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
#pragma unroll
        for (int w=0; w<NUM_WARPS; w++) { 
            int k = w * WARP_SIZE + lane;              // 0-32, 32-64, 64-96, 96-128
            sum += a[m * K + k] * x[k];             // m=rows(0-4) K=32, k=(32,64,96,128) 
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        if (lane == 0) {
            y[m] = sum;
        }
    }
}

__global__ void sgemv_k128_f32x4_kernel(float *a, float *x, float *y, int M, int K) {
    int tx = threadIdx.x; // 0-31
    int ty = threadIdx.y ;// 0-4
    int bx = blockIdx.x;  // 0-M/4
    int lane = tx % WARP_SIZE; // 0-31
    int m = bx * blockDim.y + threadIdx.y; // (0-M/4) * 4 (No of rows in y) + (0-3)
    if (m < M) {
        float sum = 0.0f;
        int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
#pragma unroll
        for (int w=0; w<NUM_WARPS; w++) {
            int k = (w * WARP_SIZE + lane) * 4;
            float4 reg_x = FLOAT4(x[k]);
            float4 reg_a = FLOAT4(a[m * K + k]);
            sum += (reg_a.x * reg_x.x + reg_a.y * reg_x.y + reg_a.w * reg_x.w + reg_a.z * reg_x.z);
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        if (lane == 0) y[m] = sum;
    } 
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)  \
    if (((T).options().dtype() != (th_type))) { \
        std::cout << "tensor info:" << (T).options() << std::endl; \
        throw std::runtime_error("values must be " #th_type); \
}

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

#define ASSERT_K_IS_MULTIBLE_OF(V)                                             \
  if (K % (V) != 0) {                                                          \
    throw std::runtime_error("K must be multiple of " #V);                     \
  }

#define ASSERT_K_IS_EQUAL_OF(V)                                                \
  if (K != (V)) {                                                              \
    throw std::runtime_error("K must be " #V);                                 \
}

void sgemv_k32_f32(torch::Tensor a, torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(x, K, 1)
  CHECK_TORCH_TENSOR_SHAPE(y, M, 1)
  ASSERT_K_IS_MULTIBLE_OF(32)

  dim3 block(32, 4);
  dim3 grid((M + 4 - 1) / 4);

  sgemv_k32_f32_kernel<<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                                        reinterpret_cast<float *>(x.data_ptr()),
                                        reinterpret_cast<float *>(y.data_ptr()),
                                        M, K);
}

void sgemv_k128_f32x4(torch::Tensor a, torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(x, K, 1)
  CHECK_TORCH_TENSOR_SHAPE(y, M, 1)
  ASSERT_K_IS_MULTIBLE_OF(128)

  dim3 block(32, 4);
  dim3 grid((M + 4 - 1) / 4);

  sgemv_k128_f32x4_kernel<<<grid, block>>>(
      reinterpret_cast<float *>(a.data_ptr()),
      reinterpret_cast<float *>(x.data_ptr()),
      reinterpret_cast<float *>(y.data_ptr()), M, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(sgemv_k32_f32)
  TORCH_BINDING_COMMON_EXTENSION(sgemv_k128_f32x4)
}