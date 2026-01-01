// Double Buffering (also known as Software Pipelining) is a technique used to overlap
// Global Memory data transfers with Arthemetic computations.
// with this, we use two shared memory buffers, while the gpu is computing using Buffer A,
// simultaneously fetching data from global memory into buffer B.
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define TILE_SIZE 256
#define ELEMENTS_PER_BLOCK 2048

__device__ float op(float data, int iterations){
    float val = data;
    for (int i=0; i<iterations; i++){
        val = fabsf(sinf(val)) + fabsf(cosf(val));
    }
    return val;
}

__global__ void double_buffer_kernel(float *g_in, float *g_out, int n, int iterations) {
    // Two buffers in shared memory
    __shared__ float s_data[2][TILE_SIZE];

    int tid = threadIdx.x;
    int block_start = blockIdx.x * ELEMENTS_PER_BLOCK;

    // 1. PREFETCH the first tile into Buffer 0
    int current_tile_idx = block_start + tid;
    if (current_tile_idx < n) {
        s_data[0][tid] = g_in[current_tile_idx];
    }
    __syncthreads();

    // 2. PIPELINE LOOP
    int num_tiles = ELEMENTS_PER_BLOCK / TILE_SIZE;
    for (int t = 0; t < num_tiles; t++) {
        int active_buf = t % 2;       // Buffer currently being used for COMPUTE
        int next_buf = (t + 1) % 2;   // Buffer currently being used for LOAD

        // Load NEXT tile into the OTHER buffer while computing the current one
        int next_tile_start = block_start + (t + 1) * TILE_SIZE;
        if (next_tile_start + tid < n && t < num_tiles - 1) {
            s_data[next_buf][tid] = g_in[next_tile_start + tid];
        }

        // COMPUTE using the CURRENT active buffer
        float result = op(s_data[active_buf][tid], iterations);

        // Store result to global memory
        int out_idx = block_start + t * TILE_SIZE + tid;
        if (out_idx < n) {
            g_out[out_idx] = result;
        }

        // Synchronize before swapping buffers
        __syncthreads();
    }
}

int main() {
    int n = 1 << 20; // 1 million elements
    int iterations = 100;
    size_t size = n * sizeof(float);

    float *h_in, *h_out;
    cudaMallocHost(&h_in, size);
    cudaMallocHost(&h_out, size);

    for (int i = 0; i < n; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, size));
    CHECK(cudaMalloc(&d_out, size));

    CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    int threads = TILE_SIZE;
    int blocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    double_buffer_kernel<<<blocks, threads>>>(d_in, d_out, n, iterations);
    cudaEventRecord(stop);
    
    CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Execution Time: %.3f ms\n", ms);

    // Cleanup
    cudaFree(d_in); cudaFree(d_out);
    cudaFreeHost(h_in); cudaFreeHost(h_out);

    return 0;
}