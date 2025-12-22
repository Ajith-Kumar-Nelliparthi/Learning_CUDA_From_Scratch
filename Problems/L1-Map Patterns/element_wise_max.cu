#include <stdio.h>
#include <cuda_runtime.h>

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("Warmup complete!\n");
}

__global__ void elementwiseMax(const float *A, const float *B, float *C, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i=idx; i<N; i+=stride){
        C[i] = fmaxf(A[i], B[i]);
    }
}

__global__ void elementwiseMaxWarp(const float *A, const float *B, float *C, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = idx / 32;
    int laneId = idx % 32;
    for (int i=warpId*32 + laneId; i<N; i+= (blockDim.x * gridDim.x)){
        C[i] = fmaxf(A[i], B[i]);
    }
}

__global__ void elementwiseMax3(const float *A, const float *B, float *C, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int vecN = N / 4;
    for (int i=idx; i<vecN; i+=stride){
        float4 a = ((float4*)(A))[i];
        float4 b = ((float4*)(B))[i];
        float4 c;
        c.x = fmaxf(a.x, b.x);
        c.y = fmaxf(a.y, b.y);
        c.z = fmaxf(a.z, b.z);
        c.w = fmaxf(a.w, b.w);
        ((float4*)(C))[i] = c;
    }
    // handle tail elements
    for (int i=vecN*4 + idx; i<N; i+=stride){
        C[i] = fmaxf(A[i], B[i]);
    }
}