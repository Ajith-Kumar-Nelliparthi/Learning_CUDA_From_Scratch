// Naive interleaved reduction kernel
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduce_v1(float *input, float * output, int n){
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // load two elements per thread
    float sum = 0.0f;
    if (idx < n){
        sum = input[idx];
        if (idx + blockDim.x < n){
            sum += input[idx + blockDim.x];
        }
    }

    sdata[tid] = sum;
    __syncthreads();

    // interleaved reduction . Stride halves every interation
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if (tid < stride){
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0){
        output[blockIdx.x] = sdata[0];
    }
}

int main(){
    int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    const int blockSizes[] = {128, 256, 512};
    const int numTests = 3;
    
    float *h_input = (float *)malloc(size);

    for (int i=0; i< N; i++){
        h_input[i] = 1.0f; // Initialize all elements to 1.0f
    }

    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, size);

    for (int t=0; t< numTests; t++){
        int threadsPerBlock = blockSizes[t];
        int blocks = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

        cudaMalloc((void **)&d_output, blocks * sizeof(float));
        cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

        printf("\nTesting block size: %d (blocks: %d)\n", threadsPerBlock, blocks);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        reduce_v1<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_output, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
            return -1;
        }

        float gpu_time = 0;
        cudaEventElapsedTime(&gpu_time, start, stop);
        printf("Time taken for v1: %.2f ms\n", gpu_time);

        float *h_output = (float *)malloc(blocks * sizeof(float));
        cudaMemcpy(h_output, d_output, blocks * sizeof(float), cudaMemcpyDeviceToHost);

        float final_sum = 0.0f;
        for (int i = 0; i < blocks; i++) {
            final_sum += h_output[i];
        }

        printf("Final Sum = %.0f (expected %d)\n", final_sum, N);
        free(h_output);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    return 0;

}