#include <stdio.h>
#include <cuda_runtime.h>

__global__ void DotProduct(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N){
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float temp = 0.0f;
    if(idx < N){
        temp = A[idx] * B[idx];
    }
    sdata[tid] = temp;
    __syncthreads();

    for (int s=blockDim.x/2; s>0; s >>= 1){
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0){
        atomicAdd(C, sdata[0]);
    }
}