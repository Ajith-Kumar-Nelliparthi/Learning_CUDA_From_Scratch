# include <stdio.h>
#include <cuda_runtime.h>

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("Warmup complete!\n");
}

__global__ void log(const float *A, float *B, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i=idx; i<N; i+=stride){
        B[i] = log(A[i]);
    }
}

__global__ void log1(const float *A, float *B, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int warpId = idx / 32;
    int laneId = idx % 32;
    for (int i=32*warpId+laneId; i<N; i+=stride){
        B[i] = log(A[i]);
    }
}

__global__ void log2(const float *A, float *B, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int vecN = N /4;
    for (int i=idx; i<vecN; i+=stride){
        float4 a = ((float4*)A)[i];
        float4 b;
        b.x = log(a.x);
        b.y = log(a.y);
        b.z = log(a.z);
        b.w = log(a.w);
        ((float4*)B)[i] = b;
    }
    for (int i=vecN*4; i<N; i+=stride){
        B[i] = log(A[i]);
    }
}