#include <stdio.h>
#include <cuda_runtime.h>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_v7(float *i, float *o, int n) {
    extern __shared__ float sdata[];

    int tid  = threadIdx.x;
    int idx  = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    int grid = blockDim.x * 8 * gridDim.x;

    float sum = 0.0f;

    // grid-stride loop
    while (idx < n) {
        sum += i[idx];
        if (idx + blockDim.x < n)
            sum += i[idx + blockDim.x];
        idx += grid;
    }

    // Store in shared memory
    sdata[tid] = sum;
    __syncthreads();

    // block reduction
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // warp reduction
    float val = sdata[tid];
    if (tid < 32) {
        val = warpReduceSum(val);
    }

    if (tid == 0) {
        o[blockIdx.x] = val;
    }
}

int main() {
    const int N = 1 << 20;
    size_t size = N * sizeof(float);

    const int blockSizes[] = {128, 256, 512};
    int numTests = 3;

    float *h_a = (float *)malloc(size);
    for (int i = 0; i < N; i++) h_a[i] = 1.0f;

    float *d_a, *d_b;
    cudaMalloc(&d_a, size);

    for (int t = 0; t < numTests; t++) {
        int threadsPerBlock = blockSizes[t];
        int blocks = (N + threadsPerBlock * 8 - 1) / (threadsPerBlock * 8);

        cudaMalloc(&d_b, blocks * sizeof(float));
        cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

        printf("\nTesting block size: %d (blocks: %d)\n",
               threadsPerBlock, blocks);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        reduce_v7<<<blocks, threadsPerBlock,
                    threadsPerBlock * sizeof(float)>>>(d_a, d_b, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_time;
        cudaEventElapsedTime(&gpu_time, start, stop);
        printf("Time taken for v7: %.4f ms\n", gpu_time);

        float *h_b = (float *)malloc(blocks * sizeof(float));
        cudaMemcpy(h_b, d_b, blocks * sizeof(float), cudaMemcpyDeviceToHost);

        float final_sum = 0;
        for (int i = 0; i < blocks; i++) final_sum += h_b[i];

        printf("Final Sum = %.0f (expected %d)\n", final_sum, N);

        free(h_b);
        cudaFree(d_b);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(d_a);
    free(h_a);
    return 0;
}