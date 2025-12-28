#include <stdio.h>
#include <cuda_runtime.h>

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("warmup complete.\n");
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

__global__ void histogram(const int* __restrict__ A, int *hist, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        atomicAdd(&hist[A[idx]], 1);
    }
}

// method -2:
__global__ void Histogram(const int* __restrict__ A, int* __restrict__ hist, int N, int num_bins){
    extern __shared__ int s_hist[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // step-1: each thread of shared memory initalizes to zero
    for (int i=tid; i<num_bins; i+=blockDim.x){
        s_hist[i] = 0;
    }
    __syncthreads();

    // step-2: each thread process elements and updates private histogram.
    if (idx < N){
        int val = A[idx];
        if (val > 0 && val < num_bins){
            atomicAdd(&s_hist[val], 1);
        }
    }
    __syncthreads();

    // step-3: each thread update its shared histogram to global memory
    for (int i=tid; i<num_bins; i+=blockDim.x){
        atomicAdd(&hist[i], s_hist[i]);
    }
}