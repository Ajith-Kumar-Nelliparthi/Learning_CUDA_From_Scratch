#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void unified(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    // Explicitly memory allocation on the host
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float* d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    auto start1 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    unified<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "Explicit transfers time: "
              << std::chrono::duration<double, std::milli>(end1 - start1).count()
              << " ms\n";

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;

    // Unified memory allocation
    float *a, *b, *c;
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    auto start2 = std::chrono::high_resolution_clock::now();

    unified<<<(N+255)/256, 256>>>(a, b, c, N);
    cudaDeviceSynchronize();

    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "Unified memory time: "
              << std::chrono::duration<double, std::milli>(end2 - start2).count()
              << " ms\n";

    cudaFree(a); cudaFree(b); cudaFree(c);

    return 0;

}