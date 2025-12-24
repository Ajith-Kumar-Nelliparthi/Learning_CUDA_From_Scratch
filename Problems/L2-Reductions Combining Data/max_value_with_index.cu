#include <stdio.h>
#include <cuda_runtime.h>
#include <limits.h>

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("warmup complete\n");
}

#define CHECK(call) \
    do{ \
        cudaError_t err call; \
        if (err != cudaSuccess){ \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

struct MaxPair{
    int val;
    int idx;
};

__global__ void max_value_with_index(const int *A, int N, MaxPair *out){
    extern __shared__ MaxPair sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < N){
        sdata[tid].val = A[idx];
        sdata[tid].idx = idx;
    }
    else{
        sdata[tid].val = INT_MIN;
        sdata[tid].val = -1;
    }
    __syncthreads();

    for (int s=blockDim.x / 2; s > 0; s >>=1){
        if (tid < s){
            if (sdata[tid + s].val > sdata[tid].val) {sdata[tid] = sdata[tid + s];}
            __syncthreads();
        }
    }
    if (tid == 0){
        out[blockIdx.x] = sdata[0];
    }
}

// 2nd kernel
__device__ __forceinline__ MaxPair max_pair(MaxPair a, MaxPair b){
    return (b.val > a.val) ? b : a;
}

// unroll last warp
__device__ __forceinline__ void warpMax(MaxPair* sdata, unsigned int tid) {
    sdata[tid] = max_pair(sdata[tid], sdata[tid + 32]);
    sdata[tid] = max_pair(sdata[tid], sdata[tid + 16]);
    sdata[tid] = max_pair(sdata[tid], sdata[tid + 8]);
    sdata[tid] = max_pair(sdata[tid], sdata[tid + 4]);
    sdata[tid] = max_pair(sdata[tid], sdata[tid + 2]);
    sdata[tid] = max_pair(sdata[tid], sdata[tid + 1]);
}

__global__ void max_num_unroll(const int* __restrict__ A, int N, MaxPair* __restrict__ out){
    extern __shared__ MaxPair sdata[];
    int tid  = threadIdx.x;
    int gidx = blockIdx.x * blockDim.x + tid;
    int step = blockDim.x * gridDim.x;

    // Initialize local best
    MaxPair best = {INT_MIN, -1};

    // Grid-stride loop
    for (int i = gidx; i < N; i += step) {
        MaxPair cur = {A[i], i};
        best = max_pair(best, cur);

    }

    // Write local best to shared
    sdata[tid] = best;
    __syncthreads();

    // Block reduction in shared memory
    for (int s = blockDim.x >> 1; s >= 64; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max_pair(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Unroll last warp
    if (tid < 32) {
        warpMax(sdata, tid);
    }

    // Block’s final result
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}