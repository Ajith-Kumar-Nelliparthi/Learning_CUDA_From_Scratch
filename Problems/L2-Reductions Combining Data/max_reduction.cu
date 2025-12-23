#include <stdio.h>
#include <cuda_runtime.h>

__global__ void warmup(){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx == 0) printf("Hello from thread %d\n", idx);
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


// shared kernel for maximum reduction
__global__ void max_reduction(const int *A, int *B, int N){
    extern __shared__ int sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N) ? A[idx] : INT_MIN;
    __syncthreads();

    for (int s=blockDim.x/2; s>0; s >>=1){
        if (tid < s){
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0){
        atomicMax(B, sdata[0]);
    }
}

// unroll last warp for maximum reduction
__device__ void warpMax(volatile int *sdata, unsigned int tid){
    sdata[tid] = max(sdata[tid], sdata[tid + 32]);
    sdata[tid] = max(sdata[tid], sdata[tid + 16]);
    sdata[tid] = max(sdata[tid], sdata[tid + 8]);
    sdata[tid] = max(sdata[tid], sdata[tid + 4]);
    sdata[tid] = max(sdata[tid], sdata[tid + 2]);
    sdata[tid] = max(sdata[tid], sdata[tid + 1]);
}

__global__ void max_reduction_unroll(const int *A, int *B, int N){
    extern __shared__ int sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    sdata[tid] = (idx < N) ? A[idx] : INT_MAX;
    if (idx + blockDim.x < N){
        sdata[tid] = max(sdata[tid], A[idx + blockDim.x]);
    }
    __syncthreads();

    for (int s=blockDim.x / 2; s>0; s>>=1){
        if (tid < s){
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // unroll last warp
    if (tid < 32){
        warpMax(sdata, tid);
    }
    if (tid == 0){
        atomicMax(B, sdata[0]);
    }
}

// complete unrolling
template <unsigned int blockSize>
__device__ void warpMaxUnroll(volatile int *sdata, int tid){
    if (blockSize >= 64) sdata[tid] = max(sdata[tid], sdata[tid + 32]);
    if (blockSize >= 32) sdata[tid] = max(sdata[tid], sdata[tid + 16]);
    if (blockSize >= 16) sdata[tid] = max(sdata[tid], sdata[tid + 8]);
    if (blockSize >= 8) sdata[tid] = max(sdata[tid], sdata[tid + 4]);
    if (blockSize >= 4) sdata[tid] = max(sdata[tid], sdata[tid + 2]);
    if (blockSize >= 2) sdata[tid] = max(sdata[tid], sdata[tid + 1]);
}

template <unsigned int blockSize>
__global__ void max_reduction_complete_unroll(const int *A, int *B, int N){
    extern __shared__ int sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    sdata[tid] = (idx < N) ? A[idx] : INT_MAX;
    if (idx + blockDim.x < N){
        sdata[tid] = max(sdata[tid], A[idx + blockDim.x]);
    }
    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = max(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = max(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = max(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = max(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
    if (tid < 32){
        warpMaxUnroll<blockSize>(sdata, tid);
    }
    if (tid == 0){
        atomicMax(B, sdata[0]);
    }
}

