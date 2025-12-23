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


// shared kernel for minimum reduction
__global__ void min_reduction(const int *A, int *B, int N){
    extern __shared__ int sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N) ? A[idx] : INT_MIN;
    __syncthreads();

    for (int s=blockDim.x/2; s>0; s >>=1){
        if (tid < s){
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0){
        atomicMin(B, sdata[0]);
    }
}

// unroll last warp for minimum reduction
__device__ void warpmin(volatile int *sdata, unsigned int tid){
    sdata[tid] = min(sdata[tid], sdata[tid + 32]);
    sdata[tid] = min(sdata[tid], sdata[tid + 16]);
    sdata[tid] = min(sdata[tid], sdata[tid + 8]);
    sdata[tid] = min(sdata[tid], sdata[tid + 4]);
    sdata[tid] = min(sdata[tid], sdata[tid + 2]);
    sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

__global__ void min_reduction_unroll(const int *A, int *B, int N){
    extern __shared__ int sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    sdata[tid] = (idx < N) ? A[idx] : INT_MAX;
    if (idx + blockDim.x < N){
        sdata[tid] = min(sdata[tid], A[idx + blockDim.x]);
    }
    __syncthreads();

    for (int s=blockDim.x / 2; s>0; s>>=1){
        if (tid < s){
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // unroll last warp
    if (tid < 32){
        warpmin(sdata, tid);
    }
    if (tid == 0){
        atomicMin(B, sdata[0]);
    }
}

// complete unrolling
template <unsigned int blockSize>
__device__ void warpminUnroll(volatile int *sdata, int tid){
    if (blockSize >= 64) sdata[tid] = min(sdata[tid], sdata[tid + 32]);
    if (blockSize >= 32) sdata[tid] = min(sdata[tid], sdata[tid + 16]);
    if (blockSize >= 16) sdata[tid] = min(sdata[tid], sdata[tid + 8]);
    if (blockSize >= 8) sdata[tid] = min(sdata[tid], sdata[tid + 4]);
    if (blockSize >= 4) sdata[tid] = min(sdata[tid], sdata[tid + 2]);
    if (blockSize >= 2) sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

template <unsigned int blockSize>
__global__ void min_reduction_complete_unroll(const int *A, int *B, int N){
    extern __shared__ int sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    sdata[tid] = (idx < N) ? A[idx] : INT_MAX;
    if (idx + blockDim.x < N){
        sdata[tid] = min(sdata[tid], A[idx + blockDim.x]);
    }
    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = min(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = min(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
    if (tid < 32){
        warpminUnroll<blockSize>(sdata, tid);
    }
    if (tid == 0){
        atomicMin(B, sdata[0]);
    }
}

// warp level reduction
__inline__ __device__ int warpReducemin(int val){
    for (int offset = 16; offset >0; offset >>=1){
        val = min(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void min_reduction_warp(const int *A, int *B, int N){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int val = 0;
    for (int i=idx; i<N; i+=stride){
        val = min(val, A[i]);
    }
    // warp reduction
    val = warpReducemin(val);
    int laneId = tid % 32;
    int warpId = tid / 32;

    // write reduced value to shared memory
    __shared__ int sdata[32]; // min 1024 threads -> 32 warps
    if (laneId == 0) sdata[warpId] = val;
    __syncthreads();

    // block reduction
    if (warpId == 0){
        val = (tid < blockDim.x / 32) ? sdata[laneId] : INT_MIN;
        val = warpReducemin(val);
        if (tid == 0){
            atomicMin(B, val);
        }
    }
}

// vectorized load
__global__ void min_reduction_vectorized(const int *A, int *B, int N){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int vecN = N / 4;
    int val = INT_MIN;
    for (int i=idx; i<vecN; i+=stride){
        int4 data = reinterpret_cast<const int4*>(A)[i];
        val = min(val, min(data.x, min(data.y, min(data.z, data.w))));
    }
    // handle tail
    for (int i=vecN *4 + idx; i<N; i+=stride){
        val = min(val, A[i]);
    }

    // warp reduction
    val = warpReducemin(val);
    int laneId = tid % 32;
    int warpId = tid / 32;

    // write result into shared memory
    __shared__ int sdata[32];
    if (laneId == 0) sdata[warpId] = val;
    __syncthreads();

    // block reduction
    if (warpId == 0){
        val = (tid < (blockDim.x / 32)) ? sdata[laneId] : INT_MIN;
        val = warpReducemin(val);
        if (tid == 0){
            atomicMin(B, val);
        }
    }
}