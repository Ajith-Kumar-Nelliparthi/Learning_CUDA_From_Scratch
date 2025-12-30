#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

// Error checking macro
#define CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void warmup() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("Warmup complete. Beginning benchmarks...\n");
}

__global__ void strided_Access_analysis(const float* __restrict__ A, float* B, int N, int stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid * stride;

    if (i < N) {
        B[i] = A[i];
    }
}

int main() {
    int N = 1 << 24; 
    size_t full_size = N * sizeof(float);

    // Run warmup to wake up the GPU clock
    warmup<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());

    // Allocate memory once outside the loop for efficiency
    float *h_i = (float *)malloc(full_size);
    for (int i = 0; i < N; i++) h_i[i] = static_cast<float>(i);

    float *d_i, *d_o;
    CHECK(cudaMalloc((void **)&d_i, full_size));
    CHECK(cudaMalloc((void **)&d_o, full_size));
    CHECK(cudaMemcpy(d_i, h_i, full_size, cudaMemcpyHostToDevice));

    int strides[] = {1, 2, 4, 8, 16, 32};

    printf("%-10s | %-12s | %-12s\n", "Stride", "Time (ms)", "Throughput (GB/s)");
    printf("----------------------------------------------------------\n");

    for (int s = 0; s < 6; s++) {
        int stride = strides[s];

        int active_threads = N / stride;
        int threadsPerBlock = 256;
        int blocks = (active_threads + threadsPerBlock - 1) / threadsPerBlock;

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));

        CHECK(cudaEventRecord(start));
        strided_Access_analysis<<<blocks, threadsPerBlock>>>(d_i, d_o, N, stride);
        CHECK(cudaEventRecord(stop));
        
        CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, start, stop));

        double bytes_processed = (double)(N / stride) * 2.0 * sizeof(float);
        double gbps = (bytes_processed / (ms / 1000.0)) / 1e9;

        printf("Stride %2d  | %12.4f | %12.2f\n", stride, ms, gbps);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    // Cleanup
    CHECK(cudaFree(d_i));
    CHECK(cudaFree(d_o));
    free(h_i);

    return 0;
}