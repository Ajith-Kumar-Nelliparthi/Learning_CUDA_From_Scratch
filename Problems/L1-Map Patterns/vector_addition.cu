#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addition(const float *A, const float *B, float *C, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        C[idx] = A[idx] + B[idx];
    }
}
int main(){
    const int N = 1<<20; // 1M elements
    size_t size = N * sizeof(float);
    size_t halfSize = size / 2;
    int half_N = N / 2;

    int threadsPerBlock = 256;
    int blocksPerStream = (half_N + threadsPerBlock - 1) / threadsPerBlock;

    float *h_a, *h_b, *h_c;
    cudaHostAlloc((void **)&h_a, size, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_b, size, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_c, size, cudaHostAllocDefault);

    for (int i=0; i<N; i++){
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // stream1 handles for first half
    cudaMemcpyAsync(d_a, h_a, halfSize, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_b, h_b, halfSize, cudaMemcpyHostToDevice, stream1);
    addition<<<blocksPerStream, threadsPerBlock, 0, stream1>>>(d_a, d_b, d_c, half_N);
    cudaMemcpyAsync(h_c, d_c, halfSize, cudaMemcpyDeviceToHost, stream1);

    // stream2 handles for second half
    cudaMemcpyAsync(d_a + half_N, h_a + half_N, halfSize, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_b + half_N, h_b + half_N, halfSize, cudaMemcpyHostToDevice, stream2);
    addition<<<blocksPerStream, threadsPerBlock, 0, stream2>>>(d_a + half_N, d_b + half_N, d_c + half_N, half_N);
    cudaMemcpyAsync(h_c + half_N, d_c + half_N, halfSize, cudaMemcpyDeviceToHost, stream2);
    
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    for (int i=0; i<5; i++){
        printf("C[i] = A[i] + B[i]\n %f = %f + %f\n", h_c[i], h_a[i], h_b[i]);
    }

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);
    return 0;
}