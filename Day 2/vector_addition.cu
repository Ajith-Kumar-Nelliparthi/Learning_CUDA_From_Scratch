#include <stdio.h>
#include <cuda_runtime.h>

// Initalize kernel
__global__ void add(const float *a, const float *b, float *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(; idx < size; idx += blockDim.x * gridDim.x) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {

    // Define problem size
    const int N = 1024;
    size_t size = N * sizeof(float);

    // Allocate Memory on Host(CPU)
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    // Intialize the input vectors
    for(int i=0; i<N; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // Allocate memory on Device (GPU)
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data from Host to Device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Decide hou many threads & blocks to kernel
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel on the GPU
    add<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Check for Launch Errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to Finish
    cudaDeviceSynchronize();

    // Copy results back GPU -> CPU
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print Results
    for(int i=0; i< 5; i++) {
        printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);
    }

    // Free all Memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}