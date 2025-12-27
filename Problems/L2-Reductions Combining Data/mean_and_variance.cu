#include <cuda_runtime.h>
#include <stdio.h>

// Warp-level reduction (sum)
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel 1: Compute total sum (for mean)
__global__ void compute_sum(const float* __restrict__ A, float* global_sum, int N) {
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;

    // Vectorized load (4 floats at a time)
    int vecN = N / 4;
    const float4* A4 = reinterpret_cast<const float4*>(A);
    for (int i = idx; i < vecN; i += stride) {
        float4 v = A4[i];
        sum += v.x + v.y + v.z + v.w;
    }

    // Tail
    for (int i = vecN * 4 + idx; i < N; i += stride) {
        sum += A[i];
    }

    // Warp reduce
    sum = warpReduceSum(sum);

    // One thread per warp writes to shared memory
    if (lane == 0) sdata[warp] = sum;
    __syncthreads();

    // Block reduce
    if (warp == 0) {
        sum = (tid < (blockDim.x / 32)) ? sdata[lane] : 0.0f;
        sum = warpReduceSum(sum);
        if (tid == 0) {
            atomicAdd(global_sum, sum);
        }
    }
}

// Kernel 2: Compute sum of squared differences using known mean
__global__ void compute_variance(const float* __restrict__ A,
                                 float mean,
                                 float* global_var_sum,
                                 int N) {
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    int stride = blockDim.x * gridDim.x;

    float sum_sq_diff = 0.0f;

    // Vectorized load
    int vecN = N / 4;
    const float4* A4 = reinterpret_cast<const float4*>(A);
    for (int i = idx; i < vecN; i += stride) {
        float4 v = A4[i];
        float dx1 = v.x - mean;
        float dx2 = v.y - mean;
        float dx3 = v.z - mean;
        float dx4 = v.w - mean;
        sum_sq_diff += dx1*dx1 + dx2*dx2 + dx3*dx3 + dx4*dx4;
    }

    // Tail
    for (int i = vecN * 4 + idx; i < N; i += stride) {
        float dx = A[i] - mean;
        sum_sq_diff += dx * dx;
    }

    // Same reduction as in sum kernel
    sum_sq_diff = warpReduceSum(sum_sq_diff);

    if (lane == 0) sdata[warp] = sum_sq_diff;
    __syncthreads();

    if (warp == 0) {
        sum_sq_diff = (tid < (blockDim.x / 32)) ? sdata[lane] : 0.0f;
        sum_sq_diff = warpReduceSum(sum_sq_diff);
        if (tid == 0) {
            atomicAdd(global_var_sum, sum_sq_diff);
        }
    }
}