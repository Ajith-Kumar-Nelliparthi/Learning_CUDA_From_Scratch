#include <cuda_runtime.h>
#include <iostream>

__global__ void fillKernel(float* data, int N, float val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i] = val;
}

int main() {
    const int N = 1 << 20;
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    int dev0 = 0, dev1 = 1;

    // Check peer access capability
    int canAccess01, canAccess10;
    cudaDeviceCanAccessPeer(&canAccess01, dev0, dev1);
    cudaDeviceCanAccessPeer(&canAccess10, dev1, dev0);

    if (!canAccess01 || !canAccess10) {
        std::cerr << "Peer access not supported between devices.\n";
        return 1;
    }

    // Allocate memory on both GPUs
    float* d0_data;
    float* d1_data;

    cudaSetDevice(dev0);
    cudaMalloc(&d0_data, N * sizeof(float));
    cudaDeviceEnablePeerAccess(dev1, 0);

    cudaSetDevice(dev1);
    cudaMalloc(&d1_data, N * sizeof(float));
    cudaDeviceEnablePeerAccess(dev0, 0);

    // Fill data on GPU0
    cudaSetDevice(dev0);
    fillKernel<<<gridSize, blockSize>>>(d0_data, N, 3.14f);

    // Copy directly GPU0 → GPU1
    cudaMemcpyPeer(d1_data, dev1, d0_data, dev0, N * sizeof(float));

    // Verify by copying back from GPU1
    float* h_data = new float[N];
    cudaMemcpy(h_data, d1_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "First element on GPU1 after P2P copy: " << h_data[0] << "\n";

    // Cleanup
    delete[] h_data;
    cudaFree(d0_data);
    cudaFree(d1_data);

    return 0;
}
