#include <stdio.h>
#include <cuda_runtime.h>

// kernel :- 1
__global__ void addition(const float *A, const float *B, float *C, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        C[idx] = A[idx] + B[idx];
    }
}

// kernel :- 2
__global__ void addition2(const float *A, const float *B, float *C, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // grid-stride loop
    for (int i=idx; i<N; i += blockDim.x * gridDim.x){
        C[i] = A[i] + B[i];
    } 
}

// kernel :- 3 vectorized memory access
__global__ void adddition3(const float *A, const float *B, float *C, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;  // each thread processes 4 elements

    if (vec_idx + 3 < N){
        float4 a = reinterpret_cast<const float4 *>(A)[idx];
        float4 b = reinterpret_cast<const float4 *>(B)[idx];
        float4 c;

        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;

        reinterpret_cast<float4 *>(C)[idx] = c;
    }

}

// kernel :- 4 warp-level optimization
__global__ void addition4(const float *A, const float *B, float *C, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = idx / 32; // warp size is 32
    int landId = idx % 32;
    for (int i=warpId*32 + landId; i<N; i+= (blockDim.x * gridDim.x)){
        C[i] = A[i] + B[i];
    }
}

float measureKernel(void (*kernel_func)(cudaStream_t, int, int, float*, float*, float*, int),
                    cudaStream_t stream, int blocks, int threads, 
                    float *d_a, float *d_b, float *d_c, int N, const char* name){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, stream);
    kernel_func(stream, blocks, threads, d_a, d_b, d_c, N);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("%s: %.3f ms\n", name, milliseconds);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds;
}

// Kernel launch wrappers
void launch_kernel1(cudaStream_t s, int b, int t, float *a, float *bb, float *c, int n){
    addition<<<b, t, 0, s>>>(a, bb, c, n);
}

void launch_kernel2(cudaStream_t s, int b, int t, float *a, float *bb, float *c, int n){
    addition2<<<b, t, 0, s>>>(a, bb, c, n);
}

void launch_kernel3(cudaStream_t s, int b, int t, float *a, float *bb, float *c, int n){
    adddition3<<<b/4, t, 0, s>>>(a, bb, c, n); // Fewer blocks needed
}

void launch_kernel4(cudaStream_t s, int b, int t, float *a, float *bb, float *c, int n){
    addition4<<<b, t, 0, s>>>(a, bb, c, n);
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
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    printf("=================================================================\n");
    printf("TESTING KERNELS WITH STREAMS (N=%d, Half=%d per stream)\n", N, half_N);
    printf("Threads per block: %d, Blocks per stream: %d\n", threadsPerBlock, blocksPerStream);
    printf("=================================================================\n\n");

    // ========== TEST 1: Basic Kernel ==========
    printf("--- TEST 1: Basic Thread-to-Element Kernel ---\n");
    cudaMemcpyAsync(d_a, h_a, halfSize, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_b, h_b, halfSize, cudaMemcpyHostToDevice, stream1);
    measureKernel(launch_kernel1, stream1, blocksPerStream, threadsPerBlock, d_a, d_b, d_c, half_N, "Stream1 (Basic)");
    cudaMemcpyAsync(h_c, d_c, halfSize, cudaMemcpyDeviceToHost, stream1);

    cudaMemcpyAsync(d_a + half_N, h_a + half_N, halfSize, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_b + half_N, h_b + half_N, halfSize, cudaMemcpyHostToDevice, stream2);
    measureKernel(launch_kernel1, stream2, blocksPerStream, threadsPerBlock, d_a + half_N, d_b + half_N, d_c + half_N, half_N, "Stream2 (Basic)");
    cudaMemcpyAsync(h_c + half_N, d_c + half_N, halfSize, cudaMemcpyDeviceToHost, stream2);
    
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    printf("Result verification: C[0-4] = %.1f, %.1f, %.1f, %.1f, %.1f\n\n", 
           h_c[0], h_c[1], h_c[2], h_c[3], h_c[4]);

    // ========== TEST 2: Grid-Stride Kernel ==========
    printf("--- TEST 2: Grid-Stride Loop Kernel ---\n");
    cudaMemset(d_c, 0, size); // Clear output
    
    cudaMemcpyAsync(d_a, h_a, halfSize, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_b, h_b, halfSize, cudaMemcpyHostToDevice, stream1);
    measureKernel(launch_kernel2, stream1, blocksPerStream, threadsPerBlock, d_a, d_b, d_c, half_N, "Stream1 (Grid-Stride)");
    cudaMemcpyAsync(h_c, d_c, halfSize, cudaMemcpyDeviceToHost, stream1);

    cudaMemcpyAsync(d_a + half_N, h_a + half_N, halfSize, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_b + half_N, h_b + half_N, halfSize, cudaMemcpyHostToDevice, stream2);
    measureKernel(launch_kernel2, stream2, blocksPerStream, threadsPerBlock, d_a + half_N, d_b + half_N, d_c + half_N, half_N, "Stream2 (Grid-Stride)");
    cudaMemcpyAsync(h_c + half_N, d_c + half_N, halfSize, cudaMemcpyDeviceToHost, stream2);
    
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    printf("Result verification: C[0-4] = %.1f, %.1f, %.1f, %.1f, %.1f\n\n", 
           h_c[0], h_c[1], h_c[2], h_c[3], h_c[4]);

    // ========== TEST 3: Vectorized Kernel ==========
    printf("--- TEST 3: Vectorized Memory Access (float4) ---\n");
    cudaMemset(d_c, 0, size);
    
    int vecBlocks = blocksPerStream / 4; // Process 4 elements per thread
    cudaMemcpyAsync(d_a, h_a, halfSize, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_b, h_b, halfSize, cudaMemcpyHostToDevice, stream1);
    measureKernel(launch_kernel3, stream1, vecBlocks, threadsPerBlock, d_a, d_b, d_c, half_N, "Stream1 (Vectorized)");
    cudaMemcpyAsync(h_c, d_c, halfSize, cudaMemcpyDeviceToHost, stream1);

    cudaMemcpyAsync(d_a + half_N, h_a + half_N, halfSize, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_b + half_N, h_b + half_N, halfSize, cudaMemcpyHostToDevice, stream2);
    measureKernel(launch_kernel3, stream2, vecBlocks, threadsPerBlock, d_a + half_N, d_b + half_N, d_c + half_N, half_N, "Stream2 (Vectorized)");
    cudaMemcpyAsync(h_c + half_N, d_c + half_N, halfSize, cudaMemcpyDeviceToHost, stream2);
    
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    printf("Result verification: C[0-4] = %.1f, %.1f, %.1f, %.1f, %.1f\n\n", 
           h_c[0], h_c[1], h_c[2], h_c[3], h_c[4]);

    // ========== TEST 4: Warp-Optimized Kernel ==========
    printf("--- TEST 4: Warp-Level Optimized ---\n");
    cudaMemset(d_c, 0, size);
    
    cudaMemcpyAsync(d_a, h_a, halfSize, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_b, h_b, halfSize, cudaMemcpyHostToDevice, stream1);
    measureKernel(launch_kernel4, stream1, blocksPerStream, threadsPerBlock, d_a, d_b, d_c, half_N, "Stream1 (Warp-Opt)");
    cudaMemcpyAsync(h_c, d_c, halfSize, cudaMemcpyDeviceToHost, stream1);

    cudaMemcpyAsync(d_a + half_N, h_a + half_N, halfSize, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_b + half_N, h_b + half_N, halfSize, cudaMemcpyHostToDevice, stream2);
    measureKernel(launch_kernel4, stream2, blocksPerStream, threadsPerBlock, d_a + half_N, d_b + half_N, d_c + half_N, half_N, "Stream2 (Warp-Opt)");
    cudaMemcpyAsync(h_c + half_N, d_c + half_N, halfSize, cudaMemcpyDeviceToHost, stream2);
    
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    printf("Result verification: C[0-4] = %.1f, %.1f, %.1f, %.1f, %.1f\n\n", 
           h_c[0], h_c[1], h_c[2], h_c[3], h_c[4]);

    printf("=================================================================\n");
    printf("SUMMARY OF KERNEL OPTIMIZATIONS\n");
    printf("=================================================================\n");
    printf("1. Basic: 1 thread = 1 element (simplest, good for small arrays)\n");
    printf("2. Grid-Stride: Reuses threads, flexible grid size\n");
    printf("3. Vectorized: 4x memory bandwidth (float4), fewer transactions\n");
    printf("4. Warp-Optimized: Maximizes coalescing at warp level\n");
    printf("=================================================================\n");

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);
    
    return 0;
}