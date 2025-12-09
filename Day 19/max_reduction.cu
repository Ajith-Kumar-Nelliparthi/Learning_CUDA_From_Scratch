#include <stdio.h>
#include <cuda_runtime.h>

// Naive Kernel
__global__ void naiveMaxRed(float *input, float *output, int N){
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

// host code
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA CHECK ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// cpu reference
float cpuMax(float *data, int n){
    float max_val = -INFINITY;
    for (int i=0; i<n; i++){
        if (data[i] > max_val) max_val = data[i];
    }
    return max_val;
}

// initalize random data
void initInput(float *data, int n){
    for (int i=0; i<n; i++){
        data[i] = (float)(rand() % 100000) / 100.0f;
    }
    // Manually placing a known maximum to ensure edge cases
    data[n/2] = 123456.0f;
}

int main(){
    int n = 1 << 24;
    size_t bytes = n * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    initInput(h_input, n);

    printf("Calculating CPU reference for N=%d....\n", n);
    clock_t start_cpu = clock();
    float cpu_result = cpuMax(h_input, n);
    clock_t end_cpu = clock();
    printf("CPU result: %.2f (Time : %.4f of ms)\n", cpu_result, (double)(end_cpu - start_cpu)/CLOCKS_PER_SEC * 1000);

    // allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_input, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    int max_grid = (n + 255) / 256;
    float *h_output = (float*)malloc(max_grid * 256 * sizeof(float));
    CUDA_CHECK(cudaMalloc((void**)&d_output, max_grid * 256 * sizeof(float)));

    int blockSize = 256;
    int gridSizeStandard = (n + blockSize - 1) / blockSize;
    int gridSizePersistent = 2048;

    auto runKernel = [&](const char* name, void (*kernel)(float*, float*, int), int grid, int block, int smem, bool isNaive) {
        
        // Clear output
        CUDA_CHECK(cudaMemset(d_output, 0, grid * (isNaive ? block : 1) * sizeof(float)));

        // Setup Events
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Launch
        CUDA_CHECK(cudaEventRecord(start));
        kernel<<<grid, block, smem>>>(d_input, d_output, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        // Timing
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        // Copy Result
        int output_count = isNaive ? (grid * block) : grid;
        CUDA_CHECK(cudaMemcpy(h_output, d_output, output_count * sizeof(float), cudaMemcpyDeviceToHost));

        // Final CPU Reduction of partial sums
        float gpu_result = -INFINITY;
        for(int i=0; i<output_count; i++){
            gpu_result = fmaxf(gpu_result, h_output[i]);
        }

        // Verify
        bool match = fabsf(gpu_result - cpu_result) < 1e-4; // float precision tolerance
        printf("%-25s | Grid: %4d | Time: %6.3f ms | Result: %.2f | %s\n", 
               name, grid, milliseconds, gpu_result, match ? "PASS" : "FAIL");

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    };
    printf("\nKernel Performance Comparison:\n");
    printf("--------------------------------------------------------------------------------\n");

    // RUN TESTS
    // 1. Naive (Grid Stride)
    runKernel("Naive (Grid Stride)", naiveMaxRed, gridSizePersistent, blockSize, 0, true);

    // 2. Interleaved 1 (Standard Grid)
    runKernel("Interleaved 1 (Mod)", MaxRed1, gridSizeStandard, blockSize, blockSize * sizeof(float), false);

    // 3. Interleaved 2 (Standard Grid)
    runKernel("Interleaved 2 (Strided)", MaxRed2, gridSizeStandard, blockSize, blockSize * sizeof(float), false);

    // 4. Sequential (Standard Grid)
    runKernel("Sequential", MaxRed3, gridSizeStandard, blockSize, blockSize * sizeof(float), false);

    // 5. Thread-Level (Grid Stride + Sequential Red)
    runKernel("MaxRed4 (Grid Stride)", MaxRed4, gridSizePersistent, blockSize, blockSize * sizeof(float), false);

    // 6. Vectorized (float4 + Warp Red)
    runKernel("MaxRed5 (Vec4 + Warp)", MaxRed5, gridSizePersistent, blockSize, (blockSize/32) * sizeof(float), false);

    printf("--------------------------------------------------------------------------------\n");

    // Clean up
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}