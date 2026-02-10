#include <stdio.h>
#include <cuda_runtime.h>

// global atomic
__global__ void global_hist(int *data, int *hist, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        atomicAdd(&hist[data[idx]], 1);
    }
}

// shared atomic
__global__ void shared_hist(int *data, int *hist, int n){
    __shared__ unsigned int temp[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    temp[threadIdx.x] = 0;
    __syncthreads();

    atomicAdd(&temp[data[idx]], 1);
    __syncthreads();

    atomicAdd(&hist[threadIdx.x], temp[threadIdx.x]);
}

int main(){
    int n = 1 << 20;
    int *data, *hist;
    cudaMallocManaged(&data, n * sizeof(int));
    cudaMallocManaged(&hist, 256 * sizeof(int));

    for (int i = 0; i < n; i++){
        data[i] = rand() % 256;
    }
    for (int i = 0; i < 256; i++){
        hist[i] = 0;
    }

    global_hist<<<(n + 255) / 256, 256>>>(data, hist, n);
    cudaDeviceSynchronize();

    printf("Global Histogram:\n");
    for (int i = 0; i < 256; i++){
        printf("%d: %d\n", i, hist[i]);
    }

    for (int i = 0; i < 256; i++){
        hist[i] = 0;
    }

    shared_hist<<<(n + 255) / 256, 256>>>(data, hist, n);
    cudaDeviceSynchronize();

    printf("Shared Histogram:\n");
    for (int i = 0; i < 256; i++){
        printf("%d: %d\n", i, hist[i]);
    }

    cudaFree(data);
    cudaFree(hist);
    return 0;
}