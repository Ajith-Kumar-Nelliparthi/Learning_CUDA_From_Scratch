#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduce_v5(float *i, float *o, int n){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 8) + threadIdx.x;

    float sum = 0.0f;
    // load 8 elements per thread
    if (idx + 7 * blockDim.x < n){
        sum += i[idx];
        sum += i[idx + blockDim.x];
        sum += i[idx + 2 * blockDim.x];
        sum += i[idx + 3 * blockDim.x];
        sum += i[idx + 4 * blockDim.x];
        sum += i[idx + 5 * blockDim.x];
        sum += i[idx + 6 * blockDim.x];
        sum += i[idx + 7 * blockDim.x];
    }
    sdata[tid] = sum;
    __syncthreads();

    if (blockDim.x >= 1024){
        if (tid < 512) sdata[tid] += sdata[tid + 512];
        __syncthreads();
    }
    if (blockDim.x >= 512){
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockDim.x >= 256){
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockDim.x >= 128){
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }

    if (tid < 32){
        volatile float *vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    if (tid == 0){
        o[blockIdx.x] = sdata[0];
    }
}

int main(){
    const int N = 1 << 20;
    size_t size = N * sizeof(float);
    const int blockSizes[] = {128, 256, 512};
    int numTests = 3;

    float *h_a = (float *)malloc(size);
    for (int i=0; i<N; i++){
        h_a[i] = 1.0f;
    }

    float *d_a, *d_b;
    cudaMalloc((void **)&d_a, size);

    for (int t=0; t<numTests; t++){
        int threadsPerBlock = blockSizes[t];
        int blocks = (N + threadsPerBlock * 8 - 1) / (threadsPerBlock * 8);

        cudaMalloc(&d_b, blocks * sizeof(float));
        cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
        printf("\nTesting block size: %d (blocks: %d)\n", threadsPerBlock, blocks);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        reduce_v5<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_a, d_b, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess){
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            return -1;
        }

        float gpu_time = 0;
        cudaEventElapsedTime(&gpu_time, start, stop);
        printf("Time taken for v6: %.4f ms\n", gpu_time);

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