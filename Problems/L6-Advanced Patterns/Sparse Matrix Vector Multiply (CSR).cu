#include <stdio.h>
#include <cuda_runtime.h>

__global__ void spmv_kernel(int num_rows,
                        const int* row_ptr,
                        const int* col_indices,
                        float* values,
                        const float* x,
                        float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        for (int i = row_start; i < row_end; i++) {
            int col = col_indices[i];
            sum += values[i] * x[col];
        }
        y[row] = sum;
    }
}