#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduce_v2(float *i, float *o, int n){
    extern __shared__ float std[];

    int tid = threadIdx.x; // thread index within the block
    int idx  = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // load data into shared memory
    float sum = 0.0f;
    if (idx < n){
        sum = i[idx];
        if (idx + blockDim.x < n){
            sum += i[idx + blockDim.x];
        }
    }

    // store the partial sum in shared memory
    std[tid] = sum;
    __syncthreads();

    // perform reduction in shared memory
    for (int stride=1; stride < blockDim.x; stride <<=1){
        int index = 2 * stride * tid;
        if (index < blockDim.x){
            std[index] += std[index + stride];
        }
        __syncthreads();
    }

    if (tid == 0){
        o[blockIdx.x] = std[0];
    }
}

int main(){
    int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    const int blockSizes[] = {128, 256, 512};
    const int numTests = 3;

    // Allocate host memory
    float *h_i = (float *)malloc(size);
    // Initialize input data
    for (int i=0; i<N; i++){
        h_i[i] = 1.0f;
    }

    float *d_a, *d_b;
    cudaMalloc((void **)&d_a, size);

    for(int t=0; t<numTests; t++){
        int threadsPerBlock = blockSizes[t];
        int blocks = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

        cudaMalloc(&d_b, blocks * sizeof(float));
        cudaMemcpy(d_a, h_i, size, cudaMemcpyHostToDevice);

        printf("\nTesting block size: %d (blocks: %d)\n", threadsPerBlock, blocks);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        reduce_v2<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_a, d_b, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess){
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            return -1;
        }

        float gpu_time = 0.0f;
        cudaEventElapsedTime(&gpu_time, start, stop);
        printf("Time taken for v2: %.4f ms\n", gpu_time);

        float *h_o = (float *)malloc(blocks * sizeof(float));
        cudaMemcpy(h_o, d_b, blocks * sizeof(float), cudaMemcpyDeviceToHost);

        float final_sum = 0;
        for(int i=0; i<blocks; i++){
            final_sum += h_o[i];
        }
        printf("Final Sum = %.0f (expected %d)\n", final_sum, N);

        free(h_o);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_b);
    }
    cudaFree(d_a);
    free(h_i);
    return 0;
}