#include <stdio.h>
#include <cuda_runtime.h>

// Naive Kernel
__global__ void naiveMaxRed(const float *input, float *output, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        float max_val = -INFINITY;
        for (int i=idx; i<N; i++){
            max_val = max(max_val, input[i]);
        }
        output[idx] = max_val;
    }
}

// Interleaved Addressing Kernel
__global__ void MaxRed1(float *input, float *output, int N){
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // load into shared memory
    if (idx < N){
        sdata[tid] = input[idx];
    } else {
        sdata[tid] = -INFINITY;
    }
    __syncthreads();

    // do max reduction in shared memory
    for (int stride=1; stride < blockDim.x; stride *= 2){
        if (tid % (2 * stride) == 0){
            sdata[tid] = max(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0){
        output[blockIdx.x] = sdata[0];
    }
}

// Interleaved Addressing Kernel 2
__global__ void MaxRed2(float *input, float *output, int N){
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // load into shared mem
    if (idx < N) sdata[tid] = input[idx];
    else sdata[tid] = -INFINITY;
    __syncthreads();

    // max reduction in shared mem
    for (int stride=1; stride < blockDim.x ; stride *= 2){
        int index = 2 * stride * tid;
        if (index < blockDim.x && index + stride < blockDim.x) sdata[index] = max(sdata[index], sdata[index + stride]);
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Sequential Addressing Kernel
__global__ void MaxRed3(float *input, float *output, int N){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // load into shared mem
    if (idx < N) sdata[tid] = input[idx];
    else sdata[tid] = -INFINITY;
    __syncthreads();

    // max reduction 
    for (int s = blockDim.x / 2; s > 0; s >>= 1){
        if (tid < s){
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Thread-Level Addressing
__global__ void MaxRed4(float *input, float *output, int N){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float thread_max = -INFINITY; 

    // Grid-Stride Loop
    for (int i = idx; i < N; i += blockDim.x * gridDim.x){
        thread_max = fmaxf(thread_max, input[i]);
    }
    sdata[tid] = thread_max;
    __syncthreads();

    // Block-level reduction (Sequential)
    for (int s=blockDim.x / 2; s > 0; s >>= 1){
        if (tid < s){
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// vectorised memory access , warp reduction
__inline__ __device__ float warpReduce(float val){
    for (int offset=16; offset >0; offset /= 2){
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// block reduction
__inline__ __device__ float blockReduce(float val, float* shared){
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // 1. warp-level reduction
    val = warpReduce(val);

    // 2. write reduced val to shared mem
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // 3. read back warp results and reduce the last remaining values
    if (threadIdx.x < (blockDim.x / 32.0f)) val = shared[lane];
    else val = -INFINITY;

    if (wid == 0) {
        val = warpReduce(val);
    }
    return val;
}

__global__ void MaxRed5(float *input, float *output, int N){
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float localMax = -INFINITY;

    // 1.vectorized grid
    int vecN = N / 4;
    int stride = blockDim.x * gridDim.x;
    // reinterpret cast
    float4* intptr = (float4*)input;
    for (int i=idx; i<vecN; i+= stride){
        float4 v = intptr[i];
        localMax = fmaxf(localMax, v.x);
        localMax = fmaxf(localMax, v.y);
        localMax = fmaxf(localMax, v.z);
        localMax = fmaxf(localMax, v.w);
    }
    // handle tail elements
    for (int i = vecN * 4 + idx; i < N; i += stride){
        if (i < N) localMax = fmaxf(localMax, input[i]);
    }

    // 2. Block-wide reduction using warp shuffle
    float blockMax = blockReduce(localMax, sdata);

    // 3. write result for this block
    if (tid == 0) output[blockIdx.x] = blockMax;
}