#include <stdio.h>
#include <cuda_runtime.h>

__global__ void fma_kernel(float* out, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 1.0f, b = 2.2f, c = 3.3f;
    float result = 0.0f;

    for (int i = 0; i < iterations; i++) {
        result = fmaf(a, b, result);
        result = fmaf(result, c, a);
    }
    out[idx] = result;
}

int main() {
    int N = 1 << 20;
    int iterations = 1 << 12;

    float *d_out;
    cudaMalloc(&d_out, N * sizeof(float));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    fma_kernel<<<grid, block>>>(d_out, iterations);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Total operations = iterations * FMAs per loop * threads
    double total_ops = (double)iterations * 2 * N;
    double gflops = total_ops / (ms * 1e6);

    printf("Time: %.3f ms, Throughput: %.2f GFLOP/s\n", ms, gflops);

    cudaFree(d_out);
    return 0;
}