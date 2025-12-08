#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA CHECK ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// warp reduction
static inline __device__ float warpReduceSum(float val){
    for (int offset = warpSize/2; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel-1:- Sum of Squares using vectorized loads (float4) + register accumulation
__global__ void L2SquaredSumKernel(const float * __restrict__ input, float *globalsum, int N){
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int warpIdInBlock = tid / warpSize;
    const int blockWarpCount = (blockDim.x + warpSize - 1) / warpSize;

    // 1. Thread level reduction
    const int V = 4;
    int idx = blockIdx.x * blockDim.x + tid;
    int totalThreads = blockDim.x * gridDim.x;

    int vecN = N / V;
    const float4 *vinput = reinterpret_cast<const float4*>(input);
    float acc = 0.0f;

    // process 4 elements at once
    for (int i=idx; i<vecN; i += totalThreads){
        float4 x = vinput[i];
        acc += x.x *x.x + x.y*x.y + x.z*x.z + x.w*x.w;
    }
    // Handle remaining elements
    int tail = vecN * V + idx;
    for (int i=tail; i<N; i+= totalThreads){
        float val = input[i];
        acc += val * val;
    }

    // 2. warp-level reduction
    float warpSum = warpReduceSum(acc);
    // 3. Block level reduction
    if (lane == 0){
        sdata[warpIdInBlock] = warpSum;
    }
    __syncthreads();

    // 4. Final Reduction
    if (warpIdInBlock == 0){
        float blockSum = (lane < blockWarpCount) ? sdata[lane] : 0.0f;
        blockSum = warpReduceSum(blockSum);
        if (lane == 0){
            atomicAdd(globalsum, blockSum);
        }
    }
}

// Kernel-2:- Normalized using Pre-computed L2 Norm(scalar), vectorized stores
__global__ void NormalizeKernel(const float * __restrict__ input, float * __restrict__ output, float invNorm, int N){
    const int V = 4;
    int vecN = N / V;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    const float4 *vinput = reinterpret_cast<const float4*>(input);
    float4 *voutput = reinterpret_cast<float4*>(output);

    for (int i=idx; i < vecN; i += totalThreads){
        float4 x = vinput[i];
        x.x *= invNorm; x.y *= invNorm; x.z *= invNorm; x.w *= invNorm;
        voutput[i] = x;
    }
    // tail elements
    int tail = vecN * V + idx;
    for (int i=tail; i<N; i+= totalThreads){
        output[i] = input[i] * invNorm;
    }
}

void solve(const float* d_input, float* d_output, int N){
    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    int maxBlocks = 1024;
    blocks = min(maxBlocks, blocks);

    float *d_globalsum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_globalsum, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_globalsum, 0, sizeof(float)));

    // launch reduction kernel
    int warpsPerBlock = (threadsPerBlock + 31) / 32;
    size_t bytes = warpsPerBlock * sizeof(float);

    L2SquaredSumKernel<<<blocks, threadsPerBlock, bytes>>>(d_input, d_globalsum, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy sum back
    float h_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_globalsum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_globalsum));

    float L2 = sqrtf(h_sum);
    float invNorm = (L2 > 0.0f) ? 1.0f / L2 : 0.0f;

    // normalize kernel
    NormalizeKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, invNorm, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main() {
    int N = 1 << 20; // 1048576
    size_t bytes = (size_t)N * sizeof(float);

    float* h_input = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);

    if (!h_input || !h_output) { fprintf(stderr, "host alloc failed\n"); return 1; }

    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    solve(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU execution time: %.3f ms\n", gpu_time);

    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    printf("First 5 normalized values:\n");
    for (int i = 0; i < 5 && i < N; ++i) {
        printf("output[%d] = %f (input[%d] = %f)\n", i, h_output[i], i, h_input[i]);
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return 0;
}