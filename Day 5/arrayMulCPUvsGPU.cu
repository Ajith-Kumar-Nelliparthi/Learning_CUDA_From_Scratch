#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>

// Kernel
__global__ void arrayMul(const float *a, const float *b, float *c, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(; idx < size; idx+= blockDim.x * gridDim.x){
        c[idx] = a[idx] * b[idx];
    }
}

void arrayMulCPU(const float *a, const float *b, float *c, int size){
    for(int i=0; i<size; i++){
        c[i] = a[i] * b[i];
    }
}

int main(){
    const int N = 10000;
    size_t size = N * sizeof(float);

    const int blockSizes[] = {64, 128, 256};
    const int numTests = 3;

    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);
    float *h_c_cpu = (float *)malloc(size);

    for(int i=0; i<N; i++){
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    arrayMulCPU(h_a, h_b, h_c_cpu, N); 
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    printf("CPU execution time : %.3f ms\n", cpu_time);

    for (int i=0; i< numTests; i++){
        int threadsPerBlock = blockSizes[i];
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

        printf("\nTesting block size: %d (blocks: %d)\n", threadsPerBlock, blocks);

        dim3 dimBlock(threadsPerBlock);
        dim3 dimGrid(blocks);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        arrayMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }

        float gpu_time = 0;
        cudaEventElapsedTime(&gpu_time, start, stop);
        printf("GPU Time: %f ms\n", gpu_time);

        cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

        bool correct = true;
        for(int j=0; j<N; j++){
            if (fabs(h_c[j] - h_c_cpu[j]) > 1e-5) {
                printf("verification failed at index %d: GPU = %.2f, CPU = %.2f\n", j, h_c[j], h_c_cpu[j]);
                correct = false;
                break;
            }
        }
        
        if (correct){
            printf("Result verification: SUCCESS\n");
            if(gpu_time > 0.0)
                printf("speedup: %.2fX\n", cpu_time / gpu_time);
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_cpu);

    return 0;
}