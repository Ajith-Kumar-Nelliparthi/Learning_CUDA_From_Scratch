#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

__global__ void zero_copy(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 1;
    }
}

int main() {
    const int N = 1 << 20;
    size_t size = N * sizeof(int);
    int *host_ptr;
    int *device_ptr;

    cudaHostAlloc(&host_ptr, size, cudaHostAllocMapped);
    cudaHostGetDevicePointer((void **)&device_ptr, host_ptr, 0);

    for (int i = 0; i < N; ++i) {
        host_ptr[i] = i;
    }

    zero_copy<<<1, N>>>(device_ptr, N);
    cudaDeviceSynchronize();

    std::cout << "Results after GPU increment:" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << host_ptr[i] << " ";
    }
    std::cout << std::endl;

    // Free pinned memory
    cudaFreeHost(host_ptr);

    return 0;

}