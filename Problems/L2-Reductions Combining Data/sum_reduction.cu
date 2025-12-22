#include <stdio.h>
#include <cuda_runtime.h>

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("Warmup kernel executed.\n");
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

// Naive sum reduction kernel using atomicAdd
__global__ void sum_reduction_naive(const float *A, float *B, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        atomicAdd(B, A[idx]);
    }
}

// Kernel-2: Sequential Addressing (reduces bank conflicts)
__global__ void sum_reduction2(const float *A, float *B, int N){
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    sdata[tid] = (idx < N) ? A[idx] : 0.0f;
    __syncthreads();

    for (int s=blockDim.x/2; s>0; s>>=1){
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0){
        atomicAdd(B, sdata[0]);
    }
}

// kernel-3: First add during load
__global__ void sum_reduction3(const float *A, float *B, int N){
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    sdata[tid] = 0.0f;
    if (idx < N) (sdata[tid] = A[idx]);
    if (idx + blockDim.x < N) sdata[tid] += A[idx + blockDim.x];
    __syncthreads();

    for (int s=blockDim.x / 2; s>0; s>>= 1){
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0){atomicAdd(B, sdata[0]);}
}

// kernel-4: unroll last warp
__device__ void warpReduce(volatile float *sdata, int tid){
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void sum_reduction_unroll_last_warp(const float *A, float *B, int N){
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    sdata[tid] = 0.0f;
    if (idx < N) sdata[tid] = A[idx];
    if (idx + blockDim.x < N) sdata[tid] += A[idx + blockDim.x];
    __syncthreads();

    for (int s=blockDim.x/2; s>0; s>>=1){
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // reduce last warp
    if (tid < 32){
        warpReduce(sdata, tid);
    }

    if (tid == 0){
        atomicAdd(B, sdata[0]);
    }
}

// kernel-5: complete unrolling
template <unsigned int blockSize>
__device__ void warpReduceTemplate(volatile float *sdata, int tid){
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void sum_reduction_complete_unroll(const float *A, float *B, int N){
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    sdata[tid] = 0.0f;
    if (idx < N) sdata[tid] = A[idx];
    if (idx + blockDim.x < N) sdata[tid] += A[idx + blockDim.x];
    __syncthreads();

    if (blockSize >= 1024) (if (tid < 512) {sdata[tid] += sdata[tid + 512]; } __syncthreads();)
    if (blockSize >= 512) (if (tid < 256) {sdata[tid] += sdata[tid + 256]; } __syncthreads();)
    if (blockSize >= 256) (if (tid < 128) {sdata[tid] += sdata[tid + 128]; } __syncthreads();)
    if (blockSize >= 128) (if (tid < 64) {sdata[tid] += sdata[tid + 64]; } __syncthreads();)

    if (tid < 32) warpReduceTemplate<blockSize>(sdata, tid);
    if (tid == 0){
        atomicAdd(B, sdata[0]);
    }
}

// kernel-6: multiple elements per thread
template <unsigned int blockSize>
__global__ void sum_reduction_multiple_elements(const float *A, float *B, int N){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i=idx; i<N; i+=stride){
        sum += A[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    if (blockSize >= 1024) (if (tid < 512) {sdata[tid] += sdata[tid + 512]; } __syncthreads();)
    if (blockSize >= 512) (if (tid < 256) {sdata[tid] += sdata[tid + 256]; } __syncthreads();)
    if (blockSize >= 256) (if (tid < 128) {sdata[tid] += sdata[tid + 128]; } __syncthreads();)
    if (blockSize >= 128) (if (tid < 64) {sdata[tid] += sdata[tid + 64]; } __syncthreads();)

    if (tid < 32) warpReduceTemplate<blockSize>(sdata, tid);
    if (tid == 0){
        atomicAdd(B, sdata[0]);
    }
}
