#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess){ \
            fprintf(stderr, "CUDA CHECK ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// warp-reduction
__device__  __forceinline__ float warpReduceSum(float val){
    for (int offset= 16; offset>0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// block reduction
__device__ __forceinline__ float blockReduceSum(float val, float* shared){
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // warp-level reduction
    val = warpReduceSum(val);

    // store warp results in shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // the final reduction
    if (wid == 0){
        val = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        val = warpReduceSum(val);
    }
    return val;
}

// kernel-1: sum of squares
__global__ void L2SquaredKernel(const float * __restrict__ input, float *globalsum, int N){
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float acc = 0.0f;

    // vectorized load
    int vecN = N / 4;
    const float4 *vinput = reinterpret_cast<const float4*>(input);

    // process 4 elements at once
    for (int i=idx; i<vecN; i+= stride){
        float4 v = vinput[i];
        acc += v.x * v.x;
        acc += v.y * v.y;
        acc += v.z * v.z;
        acc += v.w * v.w;
    }
    // handle remaining elements
    for (int tail = vecN * 4 + idx; tail<N; tail+= stride){
        float x = input[tail];
        acc += x * x;
    }
    // reduce within block
    acc = blockReduceSum(acc, sdata);
    if (threadIdx.x == 0){
        atomicAdd(globalsum, acc);
    }
}

// kernel-2
__global__ void NormalizedKernel(const float * __restrict__ input, float * __restrict__ output, float invNorm, int N){
    int vecN = N / 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float4 *vinput = reinterpret_cast<const float4*>(input);
    float4 *voutput = reinterpret_cast<float4*>(output);

    for (int i = idx; i<vecN; i+=stride){
        float4 v = vinput[i];
        v.x *= invNorm;
        v.y *= invNorm;
        v.z *= invNorm;
        v.w *= invNorm;
        voutput[i] = v;
    }
    for (int tail = vecN * 4 + idx; tail<N; tail+= stride){
        output[tail] = input[tail] * invNorm;
    }
}

void L2Normalize_cuda_kernel(const float *d_input, float *d_output, int N){
    const int blockSizes[] = {128, 256, 512};
    int numTests = 3;
    
    float best_time = 1e9;
    int best_block = 256;

    printf("Tuning L2 normalization on N = %d (%.2f M elements)\n\n", N, N/1e6f);
    printf("%-8s %8s %12s %12s\n", "Block", "Grid", "Time [ms]", "GB/s");

    for (int t=0; t<numTests; t++){
        int threadsPerBlock = blockSizes[t];

        int min_grid = (N + threadsPerBlock*4 - 1) / (threadsPerBlock * 4);
        int grid = max(min_grid, 1);
        grid = min(grid, 65535);

        float *d_sum = nullptr;
        CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));

        int warps_per_block = (threadsPerBlock + 31) / 32;
        size_t bytes = warps_per_block * sizeof(float);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // warm-up
        CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
        L2SquaredKernel<<<grid, threadsPerBlock, bytes>>>(d_input, d_sum, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
        L2SquaredKernel<<<grid, threadsPerBlock, bytes>>>(d_input, d_sum, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        float h_sum = 0.0f;
        CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
        float norm = sqrtf(h_sum);
        float invNorm = (norm > 1e-12f) ? 1.0f / norm : 0.0f;

        NormalizedKernel<<<grid, threadsPerBlock>>>(d_input, d_output, invNorm, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        float gbs = (2.0f * N * sizeof(float)) / (ms * 1e6f); // read + write

        printf("%-8d %8d %12.3f %12.2f\n", threadsPerBlock, grid, ms, gbs);

        if (ms < best_time) {
            best_time = ms;
            best_block  = threadsPerBlock;
        }

        CUDA_CHECK(cudaFree(d_sum));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    printf("\nBest Block: block size = %d (%.3f ms, %.2f GB/s)\n", best_block, best_time,
           (2.0f*N*sizeof(float)/(best_time*1e6f)));
}

int main() {
    const int N = 1 << 24; 
    size_t bytes = N * sizeof(float);

    float *h_in  = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);

    for (int i = 0; i < N; ++i)
        h_in[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    L2Normalize_cuda_kernel(d_in, d_out, N);

    // copy back and verify
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    double norm2 = 0.0;
    for (int i = 0; i < N; ++i) norm2 += h_out[i] * h_out[i];
    printf("\nFinal output L2 norm = %.9f (should be ≈1.0)\n", sqrt(norm2));

    // Cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in); free(h_out);

    return 0;
}