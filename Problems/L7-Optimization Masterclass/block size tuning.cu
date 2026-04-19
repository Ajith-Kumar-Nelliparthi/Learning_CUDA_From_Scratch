#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dummy(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f; // simple operation to keep the kernel non-empty
    }
}

void benchmark(int N) {
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    for (int blockSize = 32; blockSize <= 1024; blockSize *= 2) {
        int gridSize = (N + blockSize - 1) / blockSize; // calculate grid size

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        dummy<<<gridSize, blockSize>>>(d_data, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Block Size: %d, Time: %.3f ms\n", blockSize, ms);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    cudaFree(d_data);
}

int main() {
    int N = 1 << 20; // 1 million elements
    benchmark(N);
    return 0;
}