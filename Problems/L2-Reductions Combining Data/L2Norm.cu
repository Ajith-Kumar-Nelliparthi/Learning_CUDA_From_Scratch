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

// warp shuffle
__inline__ __device__ float warpReduce(float val){
    for (int offset=16; offset >0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void L2Norm(const float* __restrict__ A, float *result, float *output, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    int vecN = N / 4;
    const float4 *a = reinterpret_cast<const float4*>(A);
    for (int i=idx; i<vecN; i+=stride){
        float4 v = a[i];
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    // handle tail elements
    for (int i=vecN * 4 + idx; i<N; i+=stride){
        sum += A[i] * A[i];
    }

    // warp reduce
    sum = warpReduce(sum);
    int laneId = tid % 32;
    int warpId = tid / 32;

    // write reduced val to shared memory
    extern __shared__ float sdata[];
    if (laneId == 0) sdata[warpId] = sum;
    __syncthreads();

    // block reduction
    if (warpId == 0){
        sum = (tid < (blockDim.x / 32)) ? sdata[laneId] : 0.0f;
        sum = warpReduce(sum);
        if (tid == 0){
            atomicAdd(result, sum);
        }
        __syncthreads();
    }
}