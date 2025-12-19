#include <stdio.h>
#include <cuda_runtime.h>

__global__ void thread(int *i, int N){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int blockId = blockIdx.x;

    if (idx < N){
        printf("Global Idx: %d, Value: %d\n", idx, i[idx]);
        printf("Thread Id: %d, Block Id: %d\n", tid, blockId);
    }
}
int main(){
    int N = 32;
    size_t size = N * sizeof(int);
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    int *h_i = (int *)malloc(size);
    for (int i=0; i<N; i++){
        h_i[i] = 1;
    }

    int *d_o;
    cudaMalloc((void **)&d_o, size);
    cudaMemcpy(d_o, h_i, size, cudaMemcpyHostToDevice);
    thread<<<blocks, threadsPerBlock>>>(d_o, N);
    cudaDeviceSynchronize();

    cudaFree(d_o);
    free(h_i);
    return 0;
}