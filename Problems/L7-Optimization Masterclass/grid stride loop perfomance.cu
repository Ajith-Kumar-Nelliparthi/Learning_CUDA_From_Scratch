#include <stdio.h>
#include <cuda_runtime.h>

__global__ void normal(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        data[idx] = data[idx] * 2.0f;
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void grid_stride(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < N; idx += blockDim.x * gridDim.x) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    const int N = 1 << 20; // 1 million elements
    float *data;
    cudaMalloc(&data, N * sizeof(float));
    cudaMemset(data, 1, N * sizeof(float)); // Initialize data to 1

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    normal<<<gridSize, blockSize>>>(data, N);
    cudaDeviceSynchronize();

    grid_stride<<<gridSize, blockSize>>>(data, N);
    cudaDeviceSynchronize();

    cudaFree(data);
    return 0;
}