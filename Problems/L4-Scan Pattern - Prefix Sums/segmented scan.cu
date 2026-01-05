#include <stdio.h>
#include <cuda_runtime.h>

__global__ void segmented_scan(const int* __restrict__ A,
                               const int* __restrict__ flags,
                               int* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int sum = 0;
    for (int i = 0; i <= idx; i++) {
        if (flags[i] == 1) {
            sum = A[i]; 
        } else {
            sum += A[i];
        }
    }
    out[idx] = sum;
}

int main() {
    const int n = 1000;
    size_t size = n * sizeof(int);
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    int *h_i = (int*)malloc(size);
    int *h_flags = (int*)malloc(size);
    int *h_o = (int*)malloc(size);

    // initialization
    for (int i = 0; i < n; i++) {
        h_i[i] = rand() % 10;     
        h_flags[i] = rand() % 2; 
    }

    int *d_i, *d_flags, *d_o;
    cudaMalloc((void**)&d_i, size);
    cudaMalloc((void**)&d_flags, size);
    cudaMalloc((void**)&d_o, size);

    cudaMemcpy(d_i, h_i, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_flags, h_flags, size, cudaMemcpyHostToDevice);

    segmented_scan<<<blocks, threadsPerBlock>>>(d_i, d_flags, d_o, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_o, d_o, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 20; i++) {
        printf("%d ", h_o[i]);
    }
    printf("\n");

    cudaFree(d_i);
    cudaFree(d_flags);
    cudaFree(d_o);
    free(h_i);
    free(h_flags);
    free(h_o);

    return 0;
}