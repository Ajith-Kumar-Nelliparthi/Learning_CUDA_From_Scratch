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

// warp reduction using shuffle
static inline __device__ float warpReduceSum(float val){
    for (int offset = 16; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block Reduction using shared Memory
__device__ __forceinline__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Warp-level reduction
    val = warpReduceSum(val);

    // Write reduced value to shared memory if first lane in warp
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // First warp reduces per-warp sums
    if (wid == 0) {
        val = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        val = warpReduceSum(val);
    }
    return val;
}


// Kernel-1:- Sum of Squares using vectorized loads (float4) + register accumulation
__global__ void L2SquaredSumKernel(const float * __restrict__ input, float *globalsum, int N){
    extern __shared__ float sdata[];

    float acc = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Vectorized load (float4 = 4 floats)
    int vecN = N / 4;
    const float4 *vinput = reinterpret_cast<const float4*>(input);

    // process 4 elements at once
    for (int i = idx; i < vecN; i += stride) {
        float4 v = vinput[i];
        acc += v.x * v.x;
        acc += v.y * v.y;
        acc += v.z * v.z;
        acc += v.w * v.w;
    }

    // Handle remaining elements
    for (int i = vecN * 4 + idx; i < N; i += stride) {
        float x = input[i];
        acc += x * x;
    }
    // Reduce within block
    acc = blockReduceSum(acc, sdata);

    if (threadIdx.x == 0){
        atomicAdd(globalsum, acc);
    }
}

// Kernel-2:- Normalized using Pre-computed L2 Norm(scalar), vectorized stores
__global__ void NormalizeKernel(const float * __restrict__ input, float * __restrict__ output, float invNorm, int N){
    int vecN = N / 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float4 *vinput = reinterpret_cast<const float4*>(input);
    float4 *voutput = reinterpret_cast<float4*>(output);

    for (int i = idx; i < vecN; i += stride) {
        float4 v = vinput[i];
        v.x *= invNorm;
        v.y *= invNorm;
        v.z *= invNorm;
        v.w *= invNorm;
        voutput[i] = v;
    }
    // tail elements
    for (int i = vecN * 4 + idx; i < N; i += stride) {
        output[i] = input[i] * invNorm;
    }
}

void l2_normalize_cuda(const float* d_input, float* d_output, int N) {
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);

    float *d_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

    // Shared memory: one float per warp
    int warpsPerBlock = (threadsPerBlock + 31) / 32;
    size_t smemBytes = warpsPerBlock * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // Step 1: Compute sum of squares
    L2SquaredSumKernel<<<blocks, threadsPerBlock, smemBytes>>>(d_input, d_sum, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Compute norm and inverse
    float h_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_sum));

    float L2_norm = sqrtf(h_sum);
    float invNorm = (L2_norm > 1e-12f) ? 1.0f / L2_norm : 0.0f;

    // Step 3: Normalize
    NormalizeKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, invNorm, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    printf("L2 Normalization GPU time: %.3f ms (N = %d, %.2f M elements/sec)\n",
           elapsed_ms, N, N / (elapsed_ms * 1e3f));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    int N = 1 << 20; // 1048576
    size_t bytes = N * sizeof(float);

    float* h_input = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);

    if (!h_input || !h_output) { fprintf(stderr, "host alloc failed\n"); return 1; }

    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    l2_normalize_cuda(d_input, d_output, N);

    // Copy back result
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    // Verify correctness: ||output|| should be ~1.0
    double sum_sq = 0.0;
    for (int i = 0; i < N; ++i) {
        sum_sq += h_output[i] * h_output[i];
    }
    printf("\nVerification: L2 norm of output = %.8f (should be ~1.0)\n", sqrt(sum_sq));

    printf("First 10 values:\n");
    for (int i = 0; i < 10 && i < N; ++i) {
        printf("in[%d] = %8.5f  -> out[%d] = %8.5f\n", i, h_input[i], i, h_output[i]);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}