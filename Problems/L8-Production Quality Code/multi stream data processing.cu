#include <stdio.h>
#include <cuda_runtime.h>

__global__ void processData(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f; // Example processing
    }
}

int main() {
    const int N = 1 << 20;
    const int chunkSize = 1 << 16;
    const int numChunks = N / chunkSize;
    const int threadsPerBlock = 256;
    const int blockSize = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;

    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;

    float* d_buffers[numChunks];
    cudaStream_t streams[numChunks];

    for (int c = 0; c < numChunks; c++) {
        cudaMalloc(&d_buffers[c], chunkSize * sizeof(float));
        cudaStreamCreate(&streams[c]);
    }

    for (int c = 0; c < numChunks; c++) {
        float* h_chunk = h_data + c * chunkSize;

        // Async copy host → device
        cudaMemcpyAsync(d_buffers[c], h_chunk,
                        chunkSize * sizeof(float),
                        cudaMemcpyHostToDevice, streams[c]);

        // Kernel launch
        processData<<<blockSize, threadsPerBlock, 0, streams[c]>>>(d_buffers[c], chunkSize);

        // Async copy device → host
        cudaMemcpyAsync(h_chunk, d_buffers[c],
                        chunkSize * sizeof(float),
                        cudaMemcpyDeviceToHost, streams[c]);
    }

    for (int c = 0; c < numChunks; c++) {
        cudaStreamSynchronize(streams[c]);
        cudaStreamDestroy(streams[c]);
        cudaFree(d_buffers[c]);
    }

    delete[] h_data;
    return 0;
}