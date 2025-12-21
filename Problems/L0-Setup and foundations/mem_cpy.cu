#include <stdio.h>
#include <cuda_runtime.h>

__global__ void memCpy(const float *i, float *o, int N){
    int idx = blockIdx.x * gridDim.x + threadIdx.x;
    if (idx < N) o[idx] = i[idx] * 10.0f;
}
int main(){
    int N = 206;
    size_t size = N * sizeof(float);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *h_i = (float *)malloc(size);
    float *h_o = (float *)malloc(size);
    for (int i=0; i<N; i++){
        h_i[i] = 1.0f;
    }

    float *d_i, *d_o;
    cudaMalloc((void **)&d_i, size);
    cudaMalloc((void **)&d_o, size);

    cudaMemcpy(d_i, h_i, size, cudaMemcpyHostToDevice);
    memCpy<<<blocks, threadsPerBlock>>>(d_i, d_o, N);
    cudaMemcpy(h_o, d_o, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int t=0; t<10; t++){
        printf("%f\n", h_o[t]);
    }

    cudaFree(d_i);
    cudaFree(d_o);
    free(h_i);
    free(h_o);
    return 0;
}