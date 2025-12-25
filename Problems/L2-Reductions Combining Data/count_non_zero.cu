#include <stdio.h>
#include <cuda_runtime.h>

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("Warmup Complete.\n");
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

__global__ void count_non_Zero_elements(const int* __restrict__ A, int *B, int N){
    extern __shared__ int sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    sdata[tid] = (idx < N && A[idx] != 0) ? 1 : 0;
    __syncthreads();

    for (int s=blockDim.x/2; s>0; s>>=1){
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0){
        B[blockIdx.x] = sdata[0];
    }
}

// unroll last warp
__device__ void warpReduce(volatile int *sdata, int tid){
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void count_non_zero_elemets2(const int* __restrict__ A, int *B, int N){
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N && A[idx] != 0) ? 1 : 0;
    __syncthreads();

    for (int s=blockDim.x/2; s>0; s >>=1){
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    // unroll last warp
    if (tid < 32)  warpReduce(sdata, tid);

    if (tid == 0){
        B[blockIdx.x] = sdata[0];
    }
}

// complete unrolling
template <unsigned int blockSize>
__device__ __forceinline__ void warpReduce(volatile int* sdata, int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void non_zero_elements(const int* __restrict__ A, int *B, int N){
    extern __shared__ int sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    sdata[tid] = (idx < N && A[idx] != 0) ? 1 : 0;
    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) sdata[tid] += sdata[tid + 512]; __syncthreads(); }
    if (blockSize >= 512)  { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256)  { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128)  { if (tid < 64)  sdata[tid] += sdata[tid + 64];  __syncthreads(); }

    if (tid < 32) {
        volatile int* vsmem = sdata;
        warpReduce<blockSize>(vsmem, tid);
    }
    if (tid == 0){
        B[blockIdx.x] = sdata[0];
    }
}

// warp shuffle
__inline__ __device__ int warpReduce(int val){
    for (int offset=16; offset>0; offset >>=1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void sum_reduction_warp_intrinsics(const float *A, float *B, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int sum = 0;
    for (int i=idx; i<N; i+=stride){
        sum += A[i];
    }

    // warp-lelvel reduction
    sum = warpReduce(sum);
    int laneId = tid % 32;
    int warpId = tid / 32;

    // write reduced value to shared memory
    __shared__ float sdata[32]; // max 1024 threads -> 32 warps
    if (laneId == 0) sdata[warpId] = sum;
    __syncthreads();

    // block-level reduction
    if (warpId == 0){
        sum = (tid < (blockDim.x / 32)) ? sdata[laneId] : 0.0f;
        sum = warpReduce(sum);
        if (tid == 0){
            B[blockIdx.x] = sum;
        }
    }
}

// vectorised with warp shuffle
__global__ void vectorized(const int* __restrict__ A, int *B, int N){
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int vecN = N / 4;
    int sum = 0;
    const int4 *a = reinterpret_cast<const int4*>(A);
    for (int i=idx; i<vecN; i+=stride){
        int4 v = a[i];
        sum += v.x + v.y + v.z + v.w;
    }
    // taile elements
    for (int i = vecN * 4 * stride + idx; i < N; i += stride) {
    sum += A[i];
    }
    // warp reduce
    sum = warpReduce(sum);
    int laneId = tid % 32;
    int warpId = tid / 32;

    // write reduced value to shared memory
    __shared__ int sdata[32];
    if (laneId == 0) sdata[warpId] = sum;
    __syncthreads();

    // block reduction
    if (warpId == 0){
        sum = (tid < (blockDim.x / 32)) ? sdata[laneId] : 0.0f;
        sum = warpReduce(sum);
        if (tid == 0){
            B[blockIdx.x] = sum;
        }
    }
}