#include <stdio.h>
#include <cuda_runtime.h>

// warp reduction
__inline__ __device__ float warpreduce(float val){
    for (int offset=warpSize/2; offset>0; offset /= 2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Dot Product Kernel
__global__ void dotProduct(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ result, int N){
    // 1. Thread level reduction
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // process 4 elements at once
    int vec = N/4;
    for (int i=idx; i < vec; i += stride){
        float4 a4 = reinterpret_cast<const float4*>(A)[i];
        float4 b4 = reinterpret_cast<const float4*>(B)[i];

        sum += (a4.x * b4.x);
        sum += (a4.y * b4.y);
        sum += (a4.z * b4.z);
        sum += (a4.w * b4.w);
    }

    // if n is not divisible by 4 - handle remaining elements
    int remainder = vec * 4;
    for (int i= remainder + idx; i<N; i+=stride){
        sum += A[i] * B[i];
    }

    // 2. warp-level reduction
    sum = warpreduce(sum);

    // 3. Block level reduction
    static __shared__ float sharedwarpsum[32]; // max 1024 threads / 32 warps = 32
    int laneId = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    if (laneId == 0){
        sharedwarpsum[warpId] = sum;
    }
    __syncthreads();

    // 4. Final Reduction
    int numWarps = blockDim.x / warpSize;
    if (warpId == 0){
      float blockSum = (laneId < numWarps) ? sharedwarpsum[laneId] : 0.0f;
      blockSum = warpreduce(blockSum);

      if (laneId == 0){
        atomicAdd(result, blockSum);
      }
    }
}

// Host Code
int main(){
    const int N = 1 << 24;
    size_t bytes = N * sizeof(float);
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *h_a = new float[N];
    float *h_b = new float[N];
    float h_result;

    for (int i=0; i<N; i++){
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    dotProduct<<<blocks, threadsPerBlock>>>(d_a, d_b, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU execution time: %.3f ms\n", gpu_time);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Dot Product = %f\n", h_result);

    // CPU verify (double)
    double cpu_sum = 0.0;
    for (int i = 0; i < N; ++i) cpu_sum += double(h_a[i]) * double(h_b[i]);
    printf("Dot Product (CPU verify) = %.6f\n", cpu_sum);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    delete[] h_a;
    delete[] h_b;
    return 0;
}