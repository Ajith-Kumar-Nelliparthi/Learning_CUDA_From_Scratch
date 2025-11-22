#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void add(const float *a, float *c, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(; idx<size; idx += blockDim.x * gridDim.x){
        c[idx] = a[idx] + 2.0f; // add 2.0 to each element
    }
}

int main() {
    const int N = 10000;
    size_t size = N * sizeof(float);

    // allocate memory on the host
    float *h_a = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    if (!h_a || !h_c) {
        printf("Host memory allocation failed!\n");
        return 1;
    }
    
    srand((unsigned int)time(NULL));   // seed for random number generation
    // initialize input data
    for(int i=0; i<N; i++){
        h_a[i] = rand() % 100; // random values between 0 and 99
    }

    // allocate memory on the device
    float *d_a, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_c, size);

    // copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    int threadsperblock = 256;
    int blocks = (N + threadsperblock - 1) / threadsperblock;
    // kernel
    add<<<blocks, threadsperblock>>>(d_a,d_c,N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceSynchronize();

    // cpy results back from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i=0; i<5; i++){
        printf("%f + 2.0 = %f\n", h_a[i], h_c[i]);
    }

    // Free all Memory
    cudaFree(d_a);
    cudaFree(d_c);
    free(h_a);
    free(h_c);

    return 0;
}