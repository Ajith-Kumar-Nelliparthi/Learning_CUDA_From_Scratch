#include <cuda_runtime.h>
#include <iostream>

__global__ void computeKernel(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i] = data[i] * 2.0f;
}

void CUDART_CB hostCallback(void* userData) {
    int* id = (int*)userData;
    std::cout << "Stream finished, callback triggered for chunk " << *id << "\n";
    delete id;
}

int main() {
    const int N = 1 << 16;
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    computeKernel<<<gridSize, blockSize, 0, stream>>>(d_data, N);
    int* chunkId = new int(1);
    cudaLaunchHostFunc(stream, hostCallback, chunkId);

    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    cudaFree(d_data);

    return 0;
}
