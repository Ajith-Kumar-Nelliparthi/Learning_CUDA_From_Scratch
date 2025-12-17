#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA CHECK ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


__inline__ __device__ float warpReuce(float val){
    for (int offset=16; offset >0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_v2(const float *i, float *o, int N){
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int grid = blockDim.x * gridDim.x;

    float sum = 0.0f;
    while (idx < N){
        sum += i[idx];
        if (idx + blockDim.x < N){
            sum += i[idx + blockDim.x];
        }
        idx += grid;
    }
    sdata[tid] = sum;
    __syncthreads();

    // block reduction
    for (int s=blockDim.x/2; s >= 32; s >>= 1){
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // warp reduction
    float val = sdata[tid];
    if (tid < 32){
        val = warpReuce(val);
    }

    if (tid == 0){
        o[blockIdx.x] = val;
    }
}

int main(){
    printf("=== CUDA Reduction Perfomance Test ===\n\n");

    int N = 1 << 20;
    size_t size = N * sizeof(float);

    const int blockSizes[] = {128, 256, 512};
    const int numTests = 3;

    float *h_a = (float *)malloc(size);
    for (int i=0; i<N; i++){
        h_a[i] = 1.0f;
    }

    float *d_a, *d_b;
    printf("Allocating GPU Memory\n");
    CUDA_CHECK(cudaMalloc((void **)&d_a, size));

    int maxBlocks = (N + 128 * 2 - 1) / (128 * 2);
    CUDA_CHECK(cudaMalloc((void **)&d_b, maxBlocks * sizeof(float)));
    printf(" d_a: %.2f MB\n", size / (1024.0*1024.0));
    printf(" d_b: %.2f MB (max %d blocks)\n\n", (maxBlocks * sizeof(float)) / (1024.0*1024.0), maxBlocks);

    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    printf(" Transferred: %.2f MB \n\n", size / (1024.0*1024.0));

    // warmup run
    reduce_v2<<<512, 256, 256 * sizeof(float)>>>(d_a, d_b, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Running Tests...\n");
    // overall timing
    cudaEvent_t overall_start, overall_stop;
    CUDA_CHECK(cudaEventCreate(&overall_start));
    CUDA_CHECK(cudaEventCreate(&overall_stop));
    CUDA_CHECK(cudaEventRecord(overall_start));

    for (int t=0; t<numTests; t++){
        int threadsPerBlock = blockSizes[t];
        int blocks = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        reduce_v2<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_a, d_b, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float kernel_time = 0;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_time, start, stop));

        // copy results device to host
        float *h_b = (float *)malloc(blocks * sizeof(float));
        cudaEvent_t copy_start, copy_stop;
        CUDA_CHECK(cudaEventCreate(&copy_start));
        CUDA_CHECK(cudaEventCreate(&copy_stop));
        CUDA_CHECK(cudaEventRecord(copy_start));
        CUDA_CHECK(cudaMemcpy(h_b, d_b, blocks * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(copy_stop));
        CUDA_CHECK(cudaEventSynchronize(copy_stop));

        float copy_time = 0;
        CUDA_CHECK(cudaEventElapsedTime(&copy_time, copy_start, copy_stop));

        // total sum
        float final_sum = 0;
        for (int i=0; i<blocks; i++){
            final_sum += h_b[i];
        }
        printf("%-15d %-10d %-15.4f %-15.4f %.0f\n",
            threadsPerBlock, blocks, kernel_time, kernel_time + copy_time, final_sum);
        free(h_b);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaEventDestroy(copy_start));
        CUDA_CHECK(cudaEventDestroy(copy_stop));
    }
    CUDA_CHECK(cudaEventRecord(overall_stop));
    CUDA_CHECK(cudaEventSynchronize(overall_stop));

    float total_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_time, overall_start, overall_stop));

    printf("\nTotal time for all 3 tests: %.4f ms\n", total_time);
    printf("\n=== Performance Summary ===\n");
    printf("Data size: %d elements (%.2f MB)\n", N, size / (1024.0*1024.0));
    printf("Average time per test: %.4f ms\n", total_time / numTests);

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    free(h_a);
    CUDA_CHECK(cudaEventDestroy(overall_start));
    CUDA_CHECK(cudaEventDestroy(overall_stop));

    return 0;
}