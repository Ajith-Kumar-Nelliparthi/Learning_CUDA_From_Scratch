#include <stdio.h>
#include <cuda_runtime.h>

__global__ void Hello(const char *i, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        printf("Thread %d prints: %c/n", idx, i[idx]);
    }
}
int main(){
    // define size
    const int N = 15;
    size_t size = (N + 1) * sizeof(char);
    // initalize host pointers
    char *h_i = (char *)malloc(size);
    // copy string from allocated memory
    strcpy(h_i, "HELLO FROM GPU");
    // initalize device pointers
    char *d_i;
    cudaMalloc((void **)&d_i, size);

    int threadsPerBLock = 256;
    int blocks = (N + threadsPerBLock - 1) / threadsPerBLock;

    cudaMemcpy(d_i, h_i, size, cudaMemcpyHostToDevice);
    Hello<<<blocks, threadsPerBLock>>>(d_i, N);
    cudaDeviceSynchronize();

    cudaFree(d_i);
    free(h_i);
    return 0;
}