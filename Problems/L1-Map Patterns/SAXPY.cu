#include <stdio.h>
#include <cuda_runtime.h>

// warmup kernel
__global__ void warmup_kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == -1) printf("This will never print");
}

// naive kernel
__global__ void kernel1(const float *A, const float *B, float *C, float alpha, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = alpha * A[idx] + B[idx];
}

// grid-stride loop
__global__ void kernel2(const float *A, const float *B, float *C, float alpha, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i=idx; i<N; i+=stride){
        C[i] = alpha * A[i] + B[i];
    }
}

// vectorized kernel
__global__ void kernel3(const float *A, const float *B, float *C, float alpha, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vecIdx = idx * 4;
    if (vecIdx + 3 < N){
        float4 a = reinterpret_cast<const float4 *>(A)[idx];
        float4 b = reinterpret_cast<const float4 *>(B)[idx];
        float4 c;
        c.x = alpha * a.x + b.x;
        c.y = alpha * a.y + b.y;
        c.z = alpha * a.z + b.z;
        c.w = alpha * a.w + b.w;
        reinterpret_cast<float4 *>(C)[idx] = c;
    }
    // Handle remaining elements
    else if (vecIdx < N) {
        for (int i = vecIdx; i < N; i++) {
            C[i] = alpha * A[i] + B[i];
        }
    }
}

// warp level kernel
__global__ void kernel4(const float *A, const float *B, float *C, float alpha, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int warpId = idx / 32;
    int laneId = idx % 32;

    for (int i=warpId*32 + laneId; i<N; i+=stride){
        C[i] = alpha * A[i] + B[i];
    }
}

// ILP kernel
__global__ void kernel5(const float *A, const float *B, float *C, float alpha, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int i = idx * 4;
    for (; i+3 < N; i+=stride*4){
        C[i+0] = alpha * A[i+0] + B[i+0];
        C[i+1] = alpha * A[i+1] + B[i+1];
        C[i+2] = alpha * A[i+2] + B[i+2];
        C[i+3] = alpha * A[i+3] + B[i+3];
    }
    // Handle remaining elements
    for (int j = i; j < N; j += stride){
        C[j] = alpha * A[j] + B[j];
    }
}

// combine vectorised and grid-stride loop kernel
__global__ void kernel6(const float *A, const float *B, float *C, float alpha, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int vecN = N / 4;
    for (int i=idx; i<vecN; i+=stride){
        float4 a = reinterpret_cast<const float4 *>(A)[i];
        float4 b = reinterpret_cast<const float4 *>(B)[i];
        float4 c;
        c.x = alpha * a.x + b.x;
        c.y = alpha * a.y + b.y;
        c.z = alpha * a.z + b.z;
        c.w = alpha * a.w + b.w;
        reinterpret_cast<float4 *>(C)[i] = c;
    }
    // Handle remaining elements (tail)
    int remaining_start = vecN * 4;
    for (int i = remaining_start + idx; i < N; i += stride){
        C[i] = alpha * A[i] + B[i];
    }
}

// kernel launcher
typedef void (*kernelFunc)(const float *, const float *, float *, float, int);

struct kernelInfo {
    kernelFunc func;
    const char* name;
    bool needVectorBlocks;
};

void launchKernel(kernelInfo kernel, cudaStream_t stream, int blocks, int threads,
                  const float *d_a, const float *d_b, float *d_c, float alpha, int N){
    int actualBlocks = kernel.needVectorBlocks ? blocks / 4 : blocks;
    kernel.func<<<actualBlocks, threads, 0, stream>>>(d_a, d_b, d_c, alpha, N);
}

void runStreamTest(kernelInfo kernel, int testNum,
                   float *d_a, float *d_b, float *d_c, float alpha,
                   float *h_a, float *h_b, float *h_c,
                   int N, int half_N, size_t size, size_t halfSize,
                   int threadsPerBlock, int blocksPerStream,
                   cudaStream_t stream1, cudaStream_t stream2){
    printf("\n--- TEST %d: %s ---\n", testNum, kernel.name);
    
    // Clear output
    cudaMemset(d_c, 0, size);
    
    // Stream 1: First half
    cudaMemcpyAsync(d_a, h_a, halfSize, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_b, h_b, halfSize, cudaMemcpyHostToDevice, stream1);
    launchKernel(kernel, stream1, blocksPerStream, threadsPerBlock, d_a, d_b, d_c, alpha, half_N);
    cudaMemcpyAsync(h_c, d_c, halfSize, cudaMemcpyDeviceToHost, stream1);

    // Stream 2: Second half
    cudaMemcpyAsync(d_a + half_N, h_a + half_N, halfSize, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_b + half_N, h_b + half_N, halfSize, cudaMemcpyHostToDevice, stream2);
    launchKernel(kernel, stream2, blocksPerStream, threadsPerBlock, 
                 d_a + half_N, d_b + half_N, d_c + half_N, alpha, half_N);
    cudaMemcpyAsync(h_c + half_N, d_c + half_N, halfSize, cudaMemcpyDeviceToHost, stream2);
    
    // Synchronize both streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    // Verify results
    printf("Result: C[0-4] = %.1f, %.1f, %.1f, %.1f, %.1f\n", 
           h_c[0], h_c[1], h_c[2], h_c[3], h_c[4]);
}

int main(){
    const int N = 1<<20; // 1M elements
    size_t size = N * sizeof(float);
    size_t halfSize = size / 2;
    int half_N = N / 2;

    int threadsPerBlock = 256;
    int blocksPerStream = (half_N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate pinned host memory
    float *h_a, *h_b, *h_c;
    cudaHostAlloc((void **)&h_a, size, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_b, size, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_c, size, cudaHostAllocDefault);

    // Initialize data
    for (int i=0; i<N; i++){
        h_a[i] = 10.0f;
        h_b[i] = 2.0f;
    }
    float alpha = 2.0f;

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    printf("CUDA KERNEL BENCHMARK - LOOP-BASED EXECUTION\n");
    printf("Array Size: %d elements (%.2f MB)\n", N, size/(1024.0*1024.0));
    printf("Per Stream: %d elements (%.2f MB)\n", half_N, halfSize/(1024.0*1024.0));
    printf("Profiling with Nsight Systems for timing analysis...\n");

    // Define all kernels
    kernelInfo kernels[] = {
        {kernel1, "Basic (1:1 Thread-Element)", false},
        {kernel2, "Grid-Stride Loop", false},
        {kernel3, "Vectorized (float4)", true},
        {kernel4, "Warp-Level Optimized", false},
        {kernel5, "ILP (4 elements/thread)", true},
        {kernel6, "Vectorized + Grid-Stride", true}
    };
    
    int numKernels = sizeof(kernels) / sizeof(kernelInfo);

    // Run all kernels in loop
    for (int i = 0; i < numKernels; i++) {
        runStreamTest(kernels[i], i+1, 
                      d_a, d_b, d_c, alpha, h_a, h_b, h_c,
                      N, half_N, size, halfSize,
                      threadsPerBlock, blocksPerStream,
                      stream1, stream2);
    }

    printf("OPTIMIZATION SUMMARY\n");
    printf("Kernel 1: Basic           - Simplest 1:1 mapping\n");
    printf("Kernel 2: Grid-Stride     - Flexible, scalable\n");
    printf("Kernel 3: Vectorized      - 4x memory bandwidth\n");
    printf("Kernel 4: Warp-Optimized  - Warp-level coalescing\n");
    printf("Kernel 5: ILP             - Instruction-level parallelism\n");
    printf("Kernel 6: Vec+Grid        - Combined approach\n");

    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);
    
    return 0;
}