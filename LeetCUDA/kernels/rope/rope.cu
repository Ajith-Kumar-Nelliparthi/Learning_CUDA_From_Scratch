#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <float.h>
#include <math.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <algorithm>

#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define BLOCK_SIZE 256
#define theta 10000.0f

__global__ void rope_f32_kernel(float *x, float *out, int seq_len, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 2d pair
    float x1 = x[idx * 2]; // corresponds to 2i
    float x2 = x[idx * 2 + 1]; // corresponds to 2i + 1
    // compute the position and dimension index
    int token_pos = idx / N;  // refers to m in the paper
    int token_idx = idx % N;  // refers to i in the paper
    // calculate frequency theta (0)
    float exp_v = 1.0f / powf(theta, 2 * token_idx / (N * 2.0f));
    // calculate the rotation angle
    float sin_v = sinf(token_pos * exp_v);
    float cos_v = cosf(token_pos * exp_v);
    // apply the rotation
    float out1 = x1 * cos_v - x2 * sin_v;
    float out2 = x1 * sin_v + x2 * cos_v;
    out[idx * 2] = out1;
    out[idx * 2 + 1] = out2;
}

__global__ void rope_f32x4_pack_kernel(float *x, float *out, int seq_len, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float4 x_v = FLOAT4(x[idx * 4]);
    int token_pos = idx / N;
    int token_idx = idx % N;
    float exp_f_v = 1.0f / powf(theta, 2 * token_idx / (N * 4.0f)); // multiply by 4 because we are processing 4 dimensions at once
    float exp_s_v = 1.0f / powf(theta, 2 * (token_idx + 1) / (N * 4.0f)); // for the second pair of dimensions
    float sin_f_v = sinf(token_pos * exp_f_v);
    float cos_f_v = cosf(token_pos * exp_f_v);
    float sin_s_v = sinf(token_pos * exp_s_v);
    float cos_s_v = cosf(token_pos * exp_s_v);
    float4 out_v;
    out_v.x = x_v.x * cos_f_v - x_v.y * sin_f_v;
    out_v.y = x_v.x * sin_f_v + x_v.y * cos_f_v;
    out_v.z = x_v.z * cos_s_v - x_v.w * sin_s_v;
    out_v.w = x_v.z * sin_s_v + x_v.w * cos_s_v;
    FLOAT4(out[idx * 4]) = out_v;
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
    if (((T).options().dtype() != (th_type))) { \
        std::cout << "Tenosr type mismatch: " << (T).options().dtype() << " vs " << (th_type) << std::endl; \
        throw std::runtime_error("Tenosr type mismatch"); \
}

void rope_f32(torch::Tensor x, torch::Tensor out) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32);
    CHECK_TORCH_TENSOR_DTYPE(out, torch::kFloat32);
    int seq_len = x.size(0);
    int hidden_size = x.size(1);
    int N = (int)(hidden_size / 2);
    dim3 grid((seq_len * N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE);
    rope_f32_kernel<<<grid, blockDim>>>(x.data_ptr<float>(), out.data_ptr<float>(), seq_len, N);
}

void rope_f32x4_pack(torch::Tensor x, torch::Tensor out) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32);
    CHECK_TORCH_TENSOR_DTYPE(out, torch::kFloat32);
    int seq_len = x.size(0);
    int hidden_size = x.size(1);
    int N = (int)(hidden_size / 4);
    dim3 grid((seq_len * N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE);
    rope_f32x4_pack_kernel<<<grid, blockDim>>>(x.data_ptr<float>(), out.data_ptr<float>(), seq_len, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(rope_f32);
    TORCH_BINDING_COMMON_EXTENSION(rope_f32x4_pack);
}