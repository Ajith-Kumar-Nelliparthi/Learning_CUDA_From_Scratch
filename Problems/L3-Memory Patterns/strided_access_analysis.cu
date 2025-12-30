#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("Warmup complete.\n");
}

#define CHECK(call) \
do{ \
    cudaError_t err = call; \
    if (err != cudaSuccess){ \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void strided_Access_analysis(const float* __restrict__ A,
                                       float* B, int N, int stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid * stride;

    if (i < N) {
        B[i] = A[i];
    }
}

int main(){
    // warmup
    warmup<<<1,1>>>();
    cudaDeviceSynchronize();

    int N = 1 <<24;
    size_t size = N * sizeof(float);

    float *h_i = (float *)malloc(size);
    float *h_o = (float *)malloc(size);
    for (int i=0; i<N; i++) h_i[i] = static_cast<float>(i);

    float *d_i, *d_o;
    cudaMalloc((void **)&d_i, size);
    cudaMalloc((void **)&d_o, size);
    cudaMemcpy(d_i, h_i, size, cudaMemcpyHostToDevice);

    int threadsPerblock = 256;
    int strides[] = {1, 2, 4, 8, 16, 32};

    for (int s=0; s<6; s++){
        int stride = strides[s];
        int blocks = (N / stride + threadsPerblock - 1) / threadsPerblock;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        strided_Access_analysis<<<blocks, threadsPerblock>>>(d_i, d_o, N, stride);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        double bytes = N * sizeof(float);
        double gbps = (bytes / (ms / 1000.0)) / 1e9;

        std::cout << "Stride " << stride << ": " << ms << " ms, "
                  << gbps << " GB/s" << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    cudaMemcpy(h_o, d_o, size, cudaMemcpyDeviceToHost);
    cudaFree(d_i); cudaFree(d_o);
    free(h_i); free(h_o);
    return 0;
}