#include <stdio.h>
#include <cuda_runtime.h>

// Kernel to perform sparse matrix-vector multiplication
__global__ void spmv_kernel(int nnz,
                        const int* row_indices,
                        const int* col_indices,
                        const float* values,
                        const float* x,
                        float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        int row = row_indices[idx];
        int col = col_indices[idx];
        float val = values[idx];
        atomicAdd(&y[row], val * x[col]);
    }
}

// Function to perform sparse matrix-vector multiplication
void spmv(int nnz,
          const int* row_indices,
          const int* col_indices,
          const float* values,
          const float* x,
          float* y) {
    // Allocate device memory
    int *d_row_indices, *d_col_indices;
    float *d_values, *d_x, *d_y;
    cudaMalloc(&d_row_indices, nnz * sizeof(int));
    cudaMalloc(&d_col_indices, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(float));
    cudaMalloc(&d_x, nnz * sizeof(float));
    cudaMalloc(&d_y, nnz * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_row_indices, row_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, nnz * sizeof(float)); // Initialize output vector to zero

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (nnz + blockSize - 1) / blockSize;
    spmv_kernel<<<numBlocks, blockSize>>>(nnz, d_row_indices, d_col_indices, d_values, d_x, d_y);

    // Copy result back to host
    cudaMemcpy(y, d_y, nnz * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_row_indices);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}

// Example usage
int main() {
    // Example sparse matrix in COO format
    int nnz = 5;
    int row_indices[] = {0, 0, 1, 2, 2};
    int col_indices[] = {0, 2, 2, 0, 1};
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float x[] = {1.0f, 2.0f, 3.0f}; // Input vector
    float y[3] = {0.0f}; // Output vector

    spmv(nnz, row_indices, col_indices, values, x, y);

    // Print result
    printf("Result of sparse matrix-vector multiplication:\n");
    for (int i = 0; i < 3; i++) {
        printf("y[%d] = %f\n", i, y[i]);
    }

    return 0;
}