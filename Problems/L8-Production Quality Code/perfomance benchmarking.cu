#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// element-wise addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

// Benchmark function
float benchmarkKernel(int N, int blockSize) {
    // Allocate host memory
    std::vector<float> h_A(N, 1.0f), h_B(N, 2.0f), h_C(N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid
    int gridSize = (N + blockSize - 1) / blockSize;

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // Cleanup
    cudaMemcpy(h_C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return ms;
}

int main() {
    int N = 1 << 20; // 1M elements
    std::vector<int> blockSizes = {64, 128, 256, 512};

    for (int b : blockSizes) {
        float time_ms = benchmarkKernel(N, b);
        std::cout << "Block size " << b << " → " << time_ms << " ms\n";
    }
    return 0;
}
