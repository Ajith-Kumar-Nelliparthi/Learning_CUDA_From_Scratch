#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>


#define WARP_SIZE 256
#define WARP_SIZE_S 16
#define PAD 1
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP 32
// col2row means read x[row][col] and write y[col][row]
__global__ void mat_transpose_f32_col2row_kernel(float *x, float *y, const int row, const int col) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_row = global_idx / col;
    const int global_col = global_idx % col;
    if (global_idx < row * col) {
        y[global_col * row + global_row] = x[global_idx];
    }
}

// row2col means  read x[col][row] write y[row][col]
__global__ void mat_transpose_f32_row2col_kernel(float *x, float *y,
                                                 const int row, const int col) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_col = global_idx / row;
  const int global_row = global_idx % row;
  if (global_idx < row * col) {
    y[global_idx] = x[global_row * col + global_col];
  }
}

__global__ void mat_transpose_f32x4_col2row_kernel(float *x, float *y, const int row, const int col) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int global_row = (global_idx * 4) / col;
    int gloabl_col = (global_idx * 4) % col;

    if (global_row < row && gloabl_col + 3< col) {
        float4 x_val = FLOAT4(global_idx);

        y[gloabl_col * row + global_row] = x_val.x;
        y[(gloabl_col + 1) * row + global_row] = x_val.y;
        y[(gloabl_col + 2) * row + global_row] = x_val.y;
        y[(gloabl_col + 3) * row + global_row] = x_val.y;
    }
}

__global__ void mat_transpose_f32x4_row2col_kernel(float *x, float *y, const int row, const int col) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int global_col = (global_idx * 4) / row;
    int global_row = (global_idx * 4) % row;

    if (global_row < row && global_col < col) {
        float4 x_val
        x_val.x = x[global_row * col + global_col];
        x_val.y = x[(global_row + 1) * col + global_col];
        x_val.w = x[(global_row + 2) * col + global_col];
        x_val.z = x[(global_row + 3) * col + global_col];
        reinterpret_cast<float4 *>(y)[global_idx] = FLOAT4(x_val);
    }
}

__global__ void mat_transpose_f32_col2row2d_kernel(float *x, float *y, const int row, const int col) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_x < col && global_y < row) {
        y[global_x * row + global_y] = x[global_y * col + global_x]
    }
}

__global__ void mat_transpose_f32_row2col2d_kernel(float *x, float *y, const int row, const int col) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_y < col && global_x < row) {
        y[global_y * row + global_x] = x[global_x * col + global_y];
    }
}

__global__ void mat_transpose_f32x4_shared_col2row2d_kernel(float *x, float *y, const int row, const int col) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    __shared__ float tile[WARP_SIZE_S][WARP_SIZE_S * 4];
    // step 1: index calculation
    if (global_x * 4 + 3 < col + 3 && global_y < row) {
        // load value from x to shared memory
        float4 x_val = reinterpret_cast<float4 *>(x)[global_y * col / 4 + global_x];
        FLOAT4(tile[local_y][local_x * 4]) = FLOAT4(x_val);
        __syncthreads();
        float4 smem_val;
        // load value from stride mem to y
        constexpr int STRIDE = WARP_SIZE_S / 4;
        smem_val.x = tile[(local_y % STRIDE) * 4][local_x * 4 + local_y / STRIDE];
        smem_val.y = tile[(local_y % STRIDE) * 4 + 1][local_x * 4 + local_y / STRIDE];
        smem_val.y = tile[(local_y % STRIDE) * 4 + 2][local_x * 4 + local_y / STRIDE];
        smem_val.y = tile[(local_y % STRIDE) * 4 + 3][local_x * 4 + local_y / STRIDE];
        
        const int bid_y = blockIdx.y * blockDim.y;
        const int out_y = global_x * 4 + local_y / STRIDE;
        const int out_x = (local_y % STRIDE) * 4 + bid_y;
        reinterpret_cast<float4 *>(y)[(out_y * row + out_x) / 4] = FLOAT4(smem_val);
    }
}

__global__ void mat_transpose_f32x4_shared_row2col2s_kernel(float *x, float *y, const int row, const int col) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    __shared__ float tile[WARP_SIZE_S * 4][WARP_SIZE_S];
    if (global_y * 4 < row && global_x < col) {
        float4 x_val;
        x_val.x = x[(global_y * 4) * col + global_x];
        x_val.y = x[(global_y * 4 + 1) * col + global_x];
        x_val.z = x[(global_y * 4 + 2) * col + global_x];
        x_val.w = x[(global_y * 4 + 3) * col + global_x];
        tile[local_y * 4][local_x] = x_val.x;
        tile[local_y * 4 + 1][local_x] = x_val.y;
        tile[local_y * 4 + 2][local_x] = x_val.z;
        tile[local_y * 4 + 3][local_x] = x_val.w;
        __syncthreads();
        float4 smem_val;
        // load value from shared memory to y.
        // add STRIDE to satisfied different block size.
        // map index n*n to (n/4)*(n*4)
        constexpr int STRIDE = WARP_SIZE_S / 4;
        smem_val.x = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4];
        smem_val.y = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 1];
        smem_val.z = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 2];
        smem_val.w = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 3];
        const int bid_x = blockIdx.x * blockDim.x;
        const int bid_y = blockIdx.y * blockDim.y;

        const int out_y = bid_x + (local_y % STRIDE) * 4;
        const int out_x = bid_y * 4 + local_x * 4 + (local_y / STRIDE);
        y[out_y * row + out_x] = smem_val.x;
        y[(out_y + 1) * row + out_x] = smem_val.y;
        y[(out_y + 2) * row + out_x] = smem_val.z;
        y[(out_y + 3) * row + out_x] = smem_val.w;
    }
}