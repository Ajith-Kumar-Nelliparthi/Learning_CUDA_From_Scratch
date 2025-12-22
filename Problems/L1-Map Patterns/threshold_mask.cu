#include <stdio.h>
#include <cuda_runtime.h>

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("Warming up GPU...\n");
}

#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess){ \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void threshold1(const float *A, float *B, float threshold, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i=idx; i<N; i+=stride){
        B[i] = (A[i] > threshold) ? 1.0f : 0.0f;
    }
}

__global__ void threshold2(const float *A, float *B, float threshold, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = idx / 32;
    int laneId = idx % 32;
    for (int i= warpId*32+laneId; i<N; i+= (blockDim.x * gridDim.x)){
        float val = A[i];
        B[i] = (val > threshold) ? 1.0f : 0.0f;
    }
}

__global__ void threshold3(const float *A, float *B, float threshold, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int vecN = N / 4;
    for (int i=idx; i<vecN; i+=stride){
        float4 val = ((float4*)(A))[i];
        float4 result;
        result.x = (val.x > threshold) ? 1.0f : 0.0f;
        result.y = (val.y > threshold) ? 1.0f : 0.0f;
        result.z = (val.z > threshold) ? 1.0f : 0.0f;
        result.w = (val.w > threshold) ? 1.0f : 0.0f;
        ((float4*)(B))[i] = result;
    }
    // handle tail elements
    for (int i=vecN*4 + idx; i<N; i+=stride){
        B[i] = (A[i] > threshold) ? 1.0f : 0.0f;
    }
}