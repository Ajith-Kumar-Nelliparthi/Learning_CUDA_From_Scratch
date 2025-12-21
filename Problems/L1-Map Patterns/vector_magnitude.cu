# include <stdio.h>
#include <cuda_runtime.h>

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == -1) {
        printf("This is a warmup kernel.\n");
    }
}

__global__ void vector_magnitude(const float *A, float x, float y, float *B, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i=idx; i<N; i+=stride){
        B[i] = sqrtf(A[i]*A[i] + x*x + y*y);
    }
}

__global__ void magnitude2(const float *A, float x, float y, float *B, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int vecN = N / 4;
    for (int i=idx; i<vecN; i+=stride){
        float4 a = ((float4*)A)[i];
        float4 res;
        res.x = sqrtf(a.x*a.x + x*x + y*y);
        res.y = sqrtf(a.y*a.y + x*x + y*y);
        res.z = sqrtf(a.z*a.z + x*x + y*y);
        res.w = sqrtf(a.w*a.w + x*x + y*y);
        ((float4*)B)[i] = res;
    }
    // Handle remaining elements
    for (int i=vecN + idx; i<N; i+=stride){
        B[i] = sqrtf(A[i]*A[i] + x*x + y*y);
    }
}

__global__ void magnitude3(const float *A, float x, float y, float *B, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpid = idx / 32;
    int laneid = idx % 32;
    for (int i=warpid*32+laneid; i<N; i+=gridDim.x * blockDim.x){
        B[i] = sqrtf(A[i]*A[i] + x*x + y*y);
    }
}