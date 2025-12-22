#include <stdio.h>
#include <cuda_runtime.h>

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("Warmup complete!\n");
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

__global__ void clampValues1(const float *A, float *B, float minVal, float maxVal, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride){
        B[i] = fmaxf(minVal, fminf(A[i], maxVal));
    }
}

__global__ void clampValues2(const float *A, float *B, float minVal, float maxVal, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int warpId = idx / 32;
    int laneId = idx % 32;
    int totalWarps = stride / 32;
    for (int i = warpId; i < (N + 31) / 32; i += totalWarps){
        #pragma unroll
        for (int j = 0; j < 32; j++){
            int index = i * 32 + j;
            if (index < N){
                float val = A[index];
                val = fmaxf(minVal, fminf(val, maxVal));
                B[index] = val;
            }
        }
    }
}

__global__ void clampValues3(const float *A, float *B, float minVal, float maxVal, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int vecN = N / 4;
    for (int i=idx; i<vecN; i+=stride){
        float4 val = ((float4*)(A))[i];
        val.x = fmaxf(minVal, fminf(val.x, maxVal));
        val.y = fmaxf(minVal, fminf(val.y, maxVal));
        val.z = fmaxf(minVal, fminf(val.z, maxVal));
        val.w = fmaxf(minVal, fminf(val.w, maxVal));
        ((float4*)(B))[i] = val;
    }
    // handle remaining elements
    for (int i = vecN * 4 + idx; i < N; i += stride){
        B[i] = fmaxf(minVal, fminf(A[i], maxVal));
    }
}