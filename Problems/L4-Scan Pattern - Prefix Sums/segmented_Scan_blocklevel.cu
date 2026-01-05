#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

#define CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess){ \
        fprintf(stderr, "cuda error in file '%s' at line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}while(0)

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("warmup complete.\n");
}

__global__ void segmented_scan_block(const int* __restrict__ A,
                                     const int* __restrict__ flags,
                                     int* out, int n) {
    extern __shared__ int shared[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    int val = (idx < n) ? A[idx] : 0;
    int flag = (idx < n) ? flags[idx] : 0;

    // Load into shared memory
    shared[tid] = val;
    shared[blockDim.x + tid] = flag;
    __syncthreads();

    // Upsweep (Hillis-Steele style)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int temp_val = val;
        int temp_flag = flag;

        if (tid >= stride) {
            int prev_val = shared[tid - stride];
            int prev_flag = shared[blockDim.x + tid - stride];

            if (prev_flag == 0) {
                temp_val = prev_val + val;
            } else {
                temp_val = val;
                temp_flag = 1;
            }
        }

        __syncthreads();
        shared[tid] = temp_val;
        shared[blockDim.x + tid] = temp_flag;
        __syncthreads();
    }

    // Write result
    if (idx < n) {
        out[idx] = shared[tid];
    }
}

int main() {
    const int n = 1 << 20;
    const int threadsPerBlock = 256;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    size_t size = n * sizeof(int);

    int *h_A = (int*)malloc(size);
    int *h_flags = (int*)malloc(size);
    int *h_out = (int*)malloc(size);

    srand(time(0));
    for (int i = 0; i < n; i++) {
        h_A[i] = rand() % 10;
        h_flags[i] = (rand() % 10 == 0) ? 1 : 0;
    }

    int *d_A, *d_flags, *d_out;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_flags, size);
    cudaMalloc(&d_out, size);

    // create cuda streams
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // warmup
    warmup<<< 1, 1, 0, stream>>>();

    CHECK(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(d_flags, h_flags, size, cudaMemcpyHostToDevice));

    // Shared memory size: 2 * threadsPerBlock * sizeof(int)
    size_t sharedMemSize = 2 * threadsPerBlock * sizeof(int);

    segmented_scan_block<<<blocks, threadsPerBlock, sharedMemSize>>>(d_A, d_flags, d_out, n);
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpyAsync(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // Print first 20 results
    printf("First 20 output values:\n");
    for (int i = 0; i < 20 && i < n; i++) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    // Cleanup
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d_A)); CHECK(cudaFree(d_flags)); CHECK(cudaFree(d_out));
    free(h_A); free(h_flags); free(h_out);

    return 0;
}