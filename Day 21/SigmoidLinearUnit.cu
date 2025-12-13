#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>

__global__ void Sigmoid(float *I, float *O, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        float x = I[idx];
        O[idx] = 1.0f / (1.0f + __expf(-x));
    }
}

int main(){
    const int N = 1024;
    size_t bytes = N * sizeof(float);
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *h_i = (float *)malloc(bytes);
    float *h_o = (float *)malloc(bytes);

    for (int i=0; i<N; i++){
        h_i[i] = (float)i / N;
    }

    float *d_i, *d_o;
    cudaMalloc((void **)&d_i, bytes);
    cudaMalloc((void **)&d_o, bytes);

    cudaMemcpy(d_i, h_i, bytes, cudaMemcpyHostToDevice);
    Sigmoid<<<blocks, threadsPerBlock>>>(d_i, d_o, N);
    cudaMemcpy(h_o, d_o, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) {
        printf("sigmoid(%f) = %f\n", h_i[i], h_o[i]);
    }

    cudaFree(d_i);
    cudaFree(d_o);
    free(h_i);
    free(h_o);

    return 0;

}