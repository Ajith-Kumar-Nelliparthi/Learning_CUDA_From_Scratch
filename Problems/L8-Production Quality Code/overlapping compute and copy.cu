#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void scaleKernel(float *data, float factor, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= factor;
    }
}

int main() {
    const int N = 1 << 20;
    const int chunksize = 1 << 18;
    const int numstreams = N / chunksize;
    const int bytesperchunk = chunksize * sizeof(float);

    float *h_data;
    cudaMallocHost(&h_data, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }

    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    cudaStream_t streams[numstreams];
    for (int i = 0; i < numstreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int threadsPerBlock = 256;
    int bloksize = (chunksize + threadsPerBlock - 1) / threadsPerBlock;

    for (int i = 0; i < numstreams; i++) {
        int offset = i * chunksize;
        cudaMemcpyAsync(d_data + offset, h_data + offset,
                        bytesperchunk, cudaMemcpyHostToDevice, streams[i]);

        scaleKernel<<<bloksize, threadsPerBlock, 0, streams[i]>>>(d_data + offset, 2.0f, chunksize);

        cudaMemcpyAsync(h_data + offset, d_data + offset,
                        bytesperchunk, cudaMemcpyDeviceToHost, streams[i]);
    }
    for (int i = 0; i < numstreams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Verify
    for (int i = 0; i < N; i++) {
        if (h_data[i] != 2.0f) {
            std::cerr << "Mismatch at " << i << "\n";
            break;
        }
    }
    cudaFree(d_data);
    cudaFreeHost(h_data);
    return 0;
}