#include <stdio.h>
#include <cuda_runtime.h>

__global__ void streamKernel(float *data, float factor, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= factor;
    }
}

int main() {
    const int N = 1 << 16;
    size_t size = N * sizeof(float);

    float *h_a = new float[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;
    }

    float *d_a, *d_b;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_a, size, cudaMemcpyHostToDevice);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    streamKernel<<<blocks, threadsPerBlock, 0, stream1>>>(d_a, 2.0f, N);
    streamKernel<<<blocks, threadsPerBlock, 0, stream2>>>(d_b, 3.0f, N);

    // Copy results back asynchronously
    cudaMemcpyAsync(h_a, d_a, size, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(h_a, d_b, size, cudaMemcpyDeviceToHost, stream2);

    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_a);
    cudaFree(d_b);
    delete[] h_a;

    return 0;
}