#include <stdio.h>
#include <cuda_runtime.h>

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("warmup complete\n");
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

__global__ void reduce_by_key(const int* __restrict__ A, const int* __restrict__ keys, int *out,int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int sum0 = 0;
    int sum1 = 0;

    for (int i=idx; i<N; i+=stride){
        if (keys[i] == 0){
            sum0 += A[i];
        }
        else if(keys[i] == 1){
            sum1 += A[i];
        }
    }
    atomicAdd(&out[0], sum0);
    atomicAdd(&out[1], sum1);
}