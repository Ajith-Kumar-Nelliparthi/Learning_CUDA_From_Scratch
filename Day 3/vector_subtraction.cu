#include <stdio.h>
#include <cuda_runtime.h>

// Initalize kernel
__global__ void sub(const float *a, const float *b, float *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < size; idx+= blockDim.x * gridDim.x) {
        c[idx] = a[idx] - b[idx];
    }
}

int main() {
    // problem size
    const int N = 1024;
    size_t size = N * sizeof(float);

    // allocate memory on host
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size); 
    float *h_c = (float *)malloc(size);

    // initlaize the vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // allocate memory on host
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // initalize threads and blocks in a kernel
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // launch the kernel
    sub<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // check for launch errors
    cudaError err_t = cudaGetLastError();
    if (err_t != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(err_t));
        return 1;
    }

    // wait for gpu to finish
    cudaDeviceSynchronize();

    // copy results back from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // print results
    for (int i = 0; i<5; i++){
        printf("%f - %f = %f\n", h_a[i], h_b[i], h_c[i]);
    }

    // free all memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}