#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduction_kernel1(const float *i, float *o, int N){
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // load data into shared memory
    float sum = 0.0f;
    if (idx < N){
        sum = i[idx];
        if (idx + blockDim.x < N){
            sum += i[idx + blockDim.x];
        }
    }
    sdata[tid] = sum;

    for (int s=blockDim.x/2; s>0; s >>= 1){
        __syncthreads();
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
    }
    if (tid == 0) o[blockIdx.x] = sdata[0];
}

int main(){
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    const int blockSizes[] = {128, 256, 512};
    const int numTests = 3;

    float *h_a = (float *)malloc(size);
    for (int i=0; i<N; i++){
        h_a[i] = 1.0f;
    }

    float *d_a, *d_b;
    cudaMalloc((void **)&d_a, size);

    for (int t=0; t<numTests; t++){
        int threadsPerBlock = blockSizes[t];
        int blocks = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

        cudaMalloc(&d_b, blocks * sizeof(float));
        cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

        printf("\nTesting block size: %d (blocks: %d)\n", threadsPerBlock, blocks);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        reduction_kernel1<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_a, d_b, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess){
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            return -1;
        }

        float gpu_time = 0;
        cudaEventElapsedTime(&gpu_time, start, stop);
        printf("Time taken for v4: %.4f ms\n", gpu_time);

        float *h_b = (float *)malloc(blocks * sizeof(float));
        cudaMemcpy(h_b, d_b, blocks * sizeof(float), cudaMemcpyDeviceToHost);

        float final_sum = 0;
        for(int i=0; i<blocks; i++){
            final_sum += h_b[i];
        }
        printf("Final Sum = %.0f (expected %d)\n", final_sum, N);

        free(h_b);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_b);
    }

    cudaFree(d_a);
    free(h_a);
    return 0;
}