#include <stdio.h>
#include <cuda_runtime.h>

__global__ void arrayMulOptimized(const float *a, const float *b, float *c, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        c[idx] = a[idx] * b[idx];
    }
}

int main(){
    const int N = 10000;
    size_t size = N * sizeof(float);

    const int blockSizes[] = {64, 128, 256};
    const int numtests = 3;

    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    for (int i=0; i< N; i++){
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    float *d_a, *d_b ,*d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    for (int t=0; t< numtests; t++){
        int threadsPerBlock = blockSizes[t];
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

        printf("\nTesting block size: %d (blocks: %d)\n", threadsPerBlock, blocks);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        arrayMulOptimized<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
            return -1;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Time taken: %.2f ms\n", milliseconds);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
}