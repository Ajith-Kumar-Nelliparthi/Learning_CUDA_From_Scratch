#include <stdio.h>
#include <cuda_runtime.h>

// warmup kernel to initialize GPU
__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("Warmup complete!\n");
}

__global__ void sigmoid(const float *A, float *B, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i=idx; i<N; i+=stride){
        B[i] = 1.0f / (1.0f + expf(-A[i]));
    }
}

__global__ void sigmoid1(const float *A, float *B, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = idx / 32;
    int laneId = idx % 32;
    for (int i = warpId*32+laneId; i<N; i+= (gridDim.x * blockDim.x)){
        B[i] = 1.0f / (1.0f + expf(-A[i]));
    }
}

__global__ void sigmoid2(const float *A, float *B, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int vecN = N / 4;
    for (int i=idx; i<vecN; i+=stride){
        float4 a = ((float4*)(A))[i];
        float4 b;
        b.x = 1.0f / (1.0f + expf(-a.x));
        b.y = 1.0f / (1.0f + expf(-a.y));
        b.z = 1.0f / (1.0f + expf(-a.z));
        b.w = 1.0f / (1.0f + expf(-a.w));
        ((float4*)(B))[i] = b;
    }
    // handle remaining elements
    for (int i=vecN*4 + idx; i<N; i+=stride){
        B[i] = 1.0f / (1.0f + expf(-A[i]));
    }
}