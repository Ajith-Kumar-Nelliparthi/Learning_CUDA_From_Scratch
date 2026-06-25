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