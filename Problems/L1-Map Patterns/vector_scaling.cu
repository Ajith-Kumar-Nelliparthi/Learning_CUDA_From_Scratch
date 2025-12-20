#include <stdio.h>
#include <cuda_runtime.h>


__global__ void warmup_kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == -1) printf("This will never print");
}

// Naive kernel
__global__ void kernel1(const float *A, float *B, float alpha, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        B[idx] = alpha * A[idx];
    }
}

// grid-stride loop kernel
__global__ void kernel2(const float *A, float *B, float alpha, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i=idx; i<N; i+=stride){
        B[i] = alpha * A[i];
    }
}

// vectorized kernel
__global__ void kernel3(const float *A, float *B, float alpha, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vecIdx = idx * 4;
    if (vecIdx + 3 < N){
        float4 a = reinterpret_cast<const float4 *>(A)[idx];
        float4 b;
        b.x = alpha * a.x;
        b.y = alpha * a.y;
        b.z = alpha * a.z;
        b.w = alpha * a.w;
        reinterpret_cast<float4 *>(B)[idx] = b;
    }
    // Handle remaining elements
    else if (vecIdx < N) {
        for (int i = vecIdx; i < N; i++) {
            B[i] = alpha * A[i];
        }
    }
}

// warp level kernel
__global__ void kernel4(const float *A, float *B, float alpha, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = idx / 32;
    int laneId = idx % 32;
    for (int i=warpId*32+laneId; i<N; i+=gridDim.x * blockDim.x){
        B[i] = alpha * A[i];
    }
}

// instruction-level parallelism kernel
__global__ void kernel5(const float *A, float *B, float alpha, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int i = idx * 4;
    for (; i + 3 < N; i += stride * 4){
        B[i+0] = alpha * A[i+0];
        B[i+1] = alpha * A[i+1];
        B[i+2] = alpha * A[i+2];
        B[i+3] = alpha * A[i+3];
    }
    
    // Handle remaining elements
    for (int j = i; j < N; j += stride){
        B[j] = alpha * A[j];
    }
}

// combine vectorised and grid-stride loop kernel
__global__ void kernel6(const float *A, float *B, float alpha, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int vecN = N / 4;
    for (int i=idx; i<vecN; i+=stride){
        float4 a = reinterpret_cast<const float4 *>(A)[i];
        float4 b;
        b.x = alpha * a.x;
        b.y = alpha * a.y;
        b.z = alpha * a.z;
        b.w = alpha * a.w;
        reinterpret_cast<float4 *>(B)[i] = b;
    }
    // Handle remaining elements (tail)
    int remaining_start = vecN * 4;
    for (int i = remaining_start + idx; i < N; i += stride){
        B[i] = alpha * A[i];
    }
}

// Optimized kernel using shared memory for better cache utilization
__global__ void kernel7(const float *A, float *B, float alpha, int N){
    extern __shared__ float shared_data[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = idx; i < N; i += stride){
        shared_data[tid] = A[i];
        __syncthreads();
        
        B[i] = alpha * shared_data[tid];
        __syncthreads();
    }
}


// kernel launcher - unified function
typedef void (*kernelFunc)(const float *, float *, float, int);

struct kernelInfo{
    kernelFunc func;
    const char *name;
    bool needVectorBlocks;
};

void launchKernel(kernelInfo kernel, cudaStream_t stream, int blocks, int threads,
                  const float *d_A, float *d_B, float alpha, int N){
    int actualBlocks = kernel.needVectorBlocks ? blocks / 4 : blocks;
    kernel.func<<<actualBlocks, threads, 0, stream>>>(d_A, d_B, alpha, N);
}

void runStreamTest(kernelInfo kernel, int testNum,
                   float *d_a, float *d_b, float alpha,
                   float *h_a, float *h_b,
                   int N, int half_N, size_t size, size_t halfSize,
                   int threadsPerBlock, int blocksPerStream,
                   cudaStream_t stream1, cudaStream_t stream2) {
    
    printf("\n--- TEST %d: %s ---\n", testNum, kernel.name);
    
    // Clear output
    cudaMemset(d_b, 0, size);
    
    // Stream 1: First half
    cudaMemcpyAsync(d_a, h_a, halfSize, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_b, h_b, halfSize, cudaMemcpyHostToDevice, stream1);
    launchKernel(kernel, stream1, blocksPerStream, threadsPerBlock, d_a, d_b, alpha, half_N);
    cudaMemcpyAsync(h_b, d_b, halfSize, cudaMemcpyDeviceToHost, stream1);

    // Stream 2: Second half
    cudaMemcpyAsync(d_a + half_N, h_a + half_N, halfSize, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_b + half_N, h_b + half_N, halfSize, cudaMemcpyHostToDevice, stream2);
    launchKernel(kernel, stream2, blocksPerStream, threadsPerBlock, 
                 d_a + half_N, d_b + half_N, alpha, half_N);
    cudaMemcpyAsync(h_b + half_N, d_b + half_N, halfSize, cudaMemcpyDeviceToHost, stream2);
    
    // Synchronize both streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    // Verify results
    printf("Result: C[0-4] = %.1f, %.1f, %.1f, %.1f, %.1f\n", 
           h_b[0], h_b[1], h_b[2], h_b[3], h_b[4]);
}

int main(){

    float *d_dummy;
    cudaMalloc(&d_dummy, 4);
    // This triggers context creation and ramps up the GPU clock frequency
    warmup_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("GPU is now warm and ready for benchmarking...\n");


    const int N = 1<<20;
    size_t size = N * sizeof(float);
    size_t halfSize = size / 2;
    int half_N = N / 2;

    int threadsPerBlock = 256;
    int blocksPerStream = (half_N + threadsPerBlock - 1) / threadsPerBlock;

    // allocate pinned host memory
    float *h_a, *h_b;
    cudaHostAlloc((void **)&h_a, size, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_b, size, cudaHostAllocDefault);

    // initialize host data
    for (int i=0; i<N; i++){
        h_a[i] = 1.0f;
    }
    float alpha = 2.0f;

    // allocate device memory
    float *d_a, *d_b;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    printf("CUDA KERNEL BENCHMARK - LOOP-BASED EXECUTION\n");
    printf("Array Size: %d elements (%.2f MB)\n", N, size/(1024.0*1024.0));
    printf("Per Stream: %d elements (%.2f MB)\n", half_N, halfSize/(1024.0*1024.0));
    printf("Profiling with Nsight Systems for timing analysis...\n");

    // Define kernels to test
    kernelInfo kernels[] = {
        {kernel1, "Basic (1:1 Thread-Element)", false},
        {kernel2, "Grid-Stride Loop", false},
        {kernel3, "Vectorized (float4)", true},
        {kernel4, "Warp-Level Optimized", false},
        {kernel5, "ILP (4 elements/thread)", true},
        {kernel6, "Vectorized + Grid-Stride", true},
        {kernel7, "Kernel 7: Shared Memory Cache", false},
    };

    int numKernels = sizeof(kernels) / sizeof(kernelInfo);

    for (int i = 0; i < numKernels; i++) {
        runStreamTest(kernels[i], i+1, 
                      d_a, d_b, alpha, h_a, h_b,
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
    printf("Kernel 7: Shared Memory   - Improved cache utilization\n");

    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_a); cudaFree(d_b);
    cudaFreeHost(h_a); cudaFreeHost(h_b);
    cudaFree(d_dummy);
    return 0;
}