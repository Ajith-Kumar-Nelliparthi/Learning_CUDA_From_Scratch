#include <stdio.h>
#include <cuda_runtime.h>

__global__ void persistent_kernel(int *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int workIndex;
    if (threadIdx.x == 0) workIndex = 0;
    __syncthreads();

    while (true) {
        int idx = atomicAdd(&workIndex, 1);

        if (idx >= n) return;

        data[idx] = data[idx] * 2;
    }
}