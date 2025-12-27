#include <stdio.h>
#include <cuda_runtime.h>

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("warmup complete.\n");
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

__global__ void dot_product(const float* __restrict__ A, const float* __restrict__ B, float *result, int N){
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i=idx; i<N; i+=stride){
        sum += A[i] * B[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s=blockDim.x/2; s>0; s>>=1){
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0){
        atomicAdd(result, sdata[0]);
    }
}

// unroll last warp
__device__ void warpReduce(volatile float *sdata, int tid){
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}
__global__ void dot_product_unroll(const float* __restrict__ A, const float* __restrict__ B, float* result, int N){
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // load into shared memory
    float temp = 0.0f;
    for (int i=idx; i<N; i+=stride){
        temp += A[i] * B[i];
    }
    sdata[tid] = temp;
    __syncthreads();

    // block reduction
    for (int s=blockDim.x/2; s >0; s >>=1){
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // unroll last warp
    if (tid < 32) warpReduce(sdata, tid);

    // final reduction
    if (tid == 0) atomicAdd(result, sdata[0]);
}

// complete unroll
template <unsigned int blockSize>
__device__ void warpReduceTemplate(volatile float* sdata, int tid){
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}
template <unsigned int blockSize>
__global__ void dot_product_complete_unroll(const float* __restrict__ A, const float* __restrict__ B, float *result, int N){
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float temp = 0.0f;
    for (int i=idx; i<N; i+=stride){
        temp += A[i] * B[i];
    }
    sdata[tid] = temp;
    __syncthreads();

    if (blockSize >= 1024) (if (tid < 512) {sdata[tid] += sdata[tid + 512]; } __syncthreads();)
    if (blockSize >= 512) (if (tid < 256) (sdata[tid] += sdata[tid + 256];) __syncthreads();)
    if (blockSize >= 256) (if (tid < 128) (sdata[tid] += sdata[tid + 128];) __syncthreads();)
    if (blockSize >= 128) (if (tid < 64) (sdata[tid] += sdata[tid + 64];) __syncthreads();)

    if (tid < 32) warpReduceTemplate<blockSize>(sdata, tid);
    if (tid == 0) atomicAdd(result, sdata[0]);
}

// warp shuffle
__inline__ __device__ float warpReduce(float val){
    for (int offset=16; offset >0; offset >>=1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
__global__ void dot_product_warp(const float* __restrict__ A, const float* __restrict__ B, float* result, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float temp = 0.0f;
    for (int i=idx; i<N; i+=stride){
        temp += A[i] * B[i];
    }

    // warp-level reduction
    temp = warpReduce(temp);
    int laneId = tid % 32;
    int warpId = tid / 32;

    // write reduced value to shared memory
    __shared__ float sdata[32]; // max 1024 threas -> 32 warps
    if (laneId == 0) sdata[warpId] = temp;
    __syncthreads();

    // block level reduction
    if (warpId == 0){
        temp = (tid < (blockDim.x / 32)) ? sdata[laneId] : 0.0f;
        temp = warpReduce(temp);
        if (tid == 0){
            atomicAdd(result, temp);
        }
    }
}

// vectorised loads with shuffle
__global__ void dot_product_vectorised(const float* __restrict__ A, const float* __restrict__ B, float* result, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    int vecN = N / 4;
    const float4 *a = reinterpret_cast<const float4*>(A);
    const float4 *b = reinterpret_cast<const float4*>(B);
    for (int i=idx; i<vecN; i+=stride){
        float4 va = a[i];
        float4 vb = b[i];
        sum += va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
    }
    // handle tail elements
    for (int i=vecN*4 + idx; i<N; i+=stride){
        sum += A[i] * B[i];
    }
    // warp - level reduction
    sum = warpReduce(sum);
    int laneId = tid % 32;
    int warpId = tid / 32;

    // write reduced val to shared memory
    __shared__ float sdata[32];
    if (laneId == 0) sdata[warpId] = sum;
    __syncthreads();

    // block reduction
    if (warpId == 0){
        sum = (tid < (blockDim.x / 32)) ? sdata[laneId] : 0.0f;
        sum = warpReduce(sum);
        if (tid == 0) atomicAdd(result, sum);
    }
}