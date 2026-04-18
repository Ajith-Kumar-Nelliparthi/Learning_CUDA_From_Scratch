#include <stdio.h>
#include <cuda_runtime.h>

// example kernel
__global__ void myKernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx % 2 == 0) {
        // do something for even threads
    }
}

int main() {
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    int blockSize = 256; // number of threads per block
    size_t dynamicSMem = 8 * 1024;

    int minGridSize = 0;
    int optimalBlockSize = 0;

    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &optimalBlockSize,
        myKernel,
        dynamicSMem,
        0
    );

    int numSMs = prop.multiProcessorCount;
    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    int maxWarpsPerSM = maxThreadsPerSM / prop.warpSize;

    int activeBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM,
        myKernel,
        blockSize,
        dynamicSMem
    );

    int activeWarpsPerSM = activeBlocksPerSM * (blockSize / prop.warpSize);
    float occupancy = (float)activeWarpsPerSM / (float)maxWarpsPerSM;

    printf("Device: %s\n", prop.name);
    printf("Optimal block size: %d\n", optimalBlockSize);
    printf("Active blocks per SM: %d\n", activeBlocksPerSM);
    printf("Active warps per SM: %d\n", activeWarpsPerSM);
    printf("Max warps per SM: %d\n", maxWarpsPerSM);
    printf("Theoretical occupancy: %.2f%%\n", occupancy * 100.0f);

    return 0;
}