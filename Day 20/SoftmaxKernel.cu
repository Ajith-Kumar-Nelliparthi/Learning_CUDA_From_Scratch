#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>

__global__ void Softmax(float* input, float* output, int N){
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    sdata[tid] = (tid < N) ? input[tid] : -FLT_MAX;

    for (int s=blockDim.x/2; s>0; s >>= 1){
        if (tid < s){
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float maxval = sdata[0];
    __syncthreads();

    float expval = (tid < N) ? expf(input[tid] - maxval) : 0.0f;
    sdata[tid] = expval;
    __syncthreads();

    for (int s=blockDim.x/2; s>0; s >>= 1){
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float expsum = sdata[0];
    __syncthreads();

    if (tid < N) output[tid] = expval / expsum;
}

int main(){
    const int N = 512;
    size_t bytes = N * sizeof(float);
    int threadsPerBLock = 512;
    int blocks = 1;

    float *h_i = (float*)malloc(bytes);
    float *h_o = (float*)malloc(bytes);

    srand(42);
    for(int i=0; i<N; i++){
        h_i[i] = (rand() / (float)RAND_MAX) * 10.0f - 5.0f;
    }

    float *d_i, *d_o;
    cudaMalloc((void **)&d_i, bytes);
    cudaMalloc((void **)&d_o, bytes);

    cudaMemcpy(d_i, h_i, bytes, cudaMemcpyHostToDevice);
    Softmax<<<blocks, threadsPerBLock>>>(d_i, d_o, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_o, d_o, bytes, cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for(int i=0; i<N; i++){
        sum+= h_o[i];
    }
    printf("Softmax Probability: %f\n", sum);

    cudaFree(d_i);
    cudaFree(d_o);
    free(h_i);
    free(h_o);
    return 0;
}