#include <stdio.h>
#include <cuda_runtime.h>

__global__ void atomic_add_comparison(int *data, int *atomic_result, int *non_atomic_result, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // atomic addition
    atomicAdd(&atomic_result[idx], data[0]);

    // non-atomic addition (for comparison)
    non_atomic_result[idx] += data[0];
}

// min atomic operation
__global__ void atomic_min_comp(int *data, int *atomic_result, int *non_atomic_result, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // atomic min
    atomicMin(&atomic_result[idx], data[0]);

    // non-atomic min (for comparison)
    if (non_atomic_result[idx] > data[0]) {
        non_atomic_result[idx] = data[0];
    }
}

// max atomic operation
__global__ void atmoic_max_comp(int *data, int *atomic_result, int *non_atomic_result, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // atomic max
    atomicMax(&atomic_result[idx], data[0]);

    // non-atomic max (for comparison)
    if (non_atomic_result[idx] < data[0]) {
        non_atomic_result[idx] = data[0];
    }
}

// cas atomic operation
__global__ void atomic_cas_comp(int *data, int *atomic_result, int *non_atomic_result, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // atomic compare and swap
    int old = atomicCAS(&atomic_result[idx], data[0], data[1]);

    // non-atomic compare and swap (for comparison)
    if (non_atomic_result[idx] == data[0]) {
        non_atomic_result[idx] = data[1];
    }
}

// host function to initialize data and launch kernels
void compare_atomic_operations(int n){
    int *data, *atomic_result, *non_atomic_result;
    int *d_data, *d_atomic_result, *d_non_atomic_result;

    // allocate host memory
    data = (int*)malloc(2 * sizeof(int));
    atomic_result = (int*)malloc(n * sizeof(int));
    non_atomic_result = (int*)malloc(n * sizeof(int));

    // initialize data
    data[0] = 1; // value to add/min/max/cas
    data[1] = 2; // value for CAS

    for (int i = 0; i < n; i++) {
        atomic_result[i] = 0; // initialize atomic result
        non_atomic_result[i] = 0; // initialize non-atomic result
    }

    // allocate device memory
    cudaMalloc(&d_data, 2 * sizeof(int));
    cudaMalloc(&d_atomic_result, n * sizeof(int));
    cudaMalloc(&d_non_atomic_result, n * sizeof(int));

    // copy data to device
    cudaMemcpy(d_data, data, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_atomic_result, atomic_result, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_non_atomic_result, non_atomic_result, n * sizeof(int), cudaMemcpyHostToDevice);

    // launch kernels
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    atomic_add_comparison<<<numBlocks, blockSize>>>(d_data, d_atomic_result, d_non_atomic_result, n);
    atomic_min_comp<<<numBlocks, blockSize>>>(d_data, d_atomic_result, d_non_atomic_result, n);
    atmoic_max_comp<<<numBlocks, blockSize>>>(d_data, d_atomic_result, d_non_atomic_result, n);
    atomic_cas_comp<<<numBlocks, blockSize>>>(d_data, d_atomic_result, d_non_atomic_result, n);

    // copy results back to host
    cudaMemcpy(atomic_result, d_atomic_result, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(non_atomic_result, d_non_atomic_result, n * sizeof(int), cudaMemcpyDeviceToHost);

    // print results for comparison
    printf("Atomic Result vs Non-Atomic Result:\n");
    for (int i = 0; i < n; i++) {
        printf("Index %d: Atomic = %d, Non-Atomic = %d\n", i, atomic_result[i], non_atomic_result[i]);
    }

    // free device memory
    cudaFree(d_data);
    cudaFree(d_atomic_result);
    cudaFree(d_non_atomic_result);

    // free host memory
    free(data);
    free(atomic_result);
    free(non_atomic_result);
}

int main() {
    int n = 10; // number of threads
    compare_atomic_operations(n);
    return 0;
}