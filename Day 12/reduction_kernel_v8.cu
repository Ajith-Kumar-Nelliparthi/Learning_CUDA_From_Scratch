#include <stdio.h>
#include <cuda_runtime.h>

__inline__ __device__ float warpReduceSum(float val){
    for (int offset = warpSize/2; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_v8(const float *i, float *o, int n){
    extern __shared__ float sdata[];
    
    // Cast input to float4 for vectorized loads
    const float4 *in_vec = reinterpret_cast<const float4*>(i);
    int num_vec = n / 4;
    
    int tid = threadIdx.x;
    // Each thread processes 8 floats = 2 float4 vectors per iteration
    int idx = blockIdx.x * (blockDim.x * 2) + tid;
    int stride = blockDim.x * 2 * gridDim.x;
    
    float sum = 0.0f;

    // Grid-stride loop: each iteration processes 2 float4 vectors (8 floats)
    while (idx < num_vec){
        // Load first float4 (4 floats)
        float4 v1 = in_vec[idx];
        sum += (v1.x + v1.y + v1.z + v1.w);
        
        // Load second float4 (4 more floats) if available
        if (idx + blockDim.x < num_vec){
            float4 v2 = in_vec[idx + blockDim.x];
            sum += (v2.x + v2.y + v2.z + v2.w);
        }
        
        idx += stride;
    }
    
    // Handle tail elements (if n is not divisible by 4)
    // Start from the last complete float4 position
    int tail_start = num_vec * 4;
    for (int k = tail_start + tid; k < n; k += blockDim.x){
        sum += i[k];
    }

    // Write partial sum to shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Block reduction in shared memory
    for (int s = blockDim.x/2; s >= 32; s >>= 1){
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp reduction
    float val = sdata[tid];
    if (tid < 32){
        val = warpReduceSum(val);
    }

    // Write block result
    if (tid == 0){
        o[blockIdx.x] = val;
    }
}

int main() {
    const int N = 1 << 20;  // 1M
    size_t size = N * sizeof(float);

    const int blockSizes[] = {128, 256, 512};
    const int numTests = 3;

    float *h_a = (float*)malloc(size);
    for (int i = 0; i < N; i++)
        h_a[i] = 1.0f;

    float *d_a;
    cudaMalloc(&d_a, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    for (int t = 0; t < numTests; t++) {
        int threads = blockSizes[t];
        // Calculate blocks: each thread processes 8 floats = 2 float4s
        int blocks = (N + threads*8 - 1) / (threads*8);

        float *d_b;
        cudaMalloc(&d_b, blocks * sizeof(float));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        reduce_v8<<<blocks, threads, threads * sizeof(float)>>>(
            d_a, d_b, N
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }

        float gpu_time = 0;
        cudaEventElapsedTime(&gpu_time, start, stop);

        float *h_b = (float*)malloc(blocks * sizeof(float));
        cudaMemcpy(h_b, d_b, blocks * sizeof(float), cudaMemcpyDeviceToHost);

        float final_sum = 0;
        for (int i = 0; i < blocks; i++) final_sum += h_b[i];

        printf("\nTesting block size: %d (blocks: %d)\n", threads, blocks);
        printf("Time taken for v8 (vectorized): %.4f ms\n", gpu_time);
        printf("Final Sum = %.0f (expected %d)\n", final_sum, N);

        free(h_b);
        cudaFree(d_b);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(d_a);
    free(h_a);
    return 0;
}