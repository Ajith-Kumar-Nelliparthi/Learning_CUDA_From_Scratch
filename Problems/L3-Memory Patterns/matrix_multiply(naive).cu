#include <stdio.h>
#include <cuda_runtime.h>

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("warmup complete.\n");
}

// error checking macro
#define CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess){ \
        fprintf(stderr, "cuda error in file '%s' in line %i : %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define TILE_DIM 16
__global__ void matrix_multiply(const float *A, const float *B, float *C, int M, int N, int K){
    int col = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (row < M && col < K){
        float sum = 0.0f;
        for (int k=0; k<N; k++){
            sum += A[row * N + k] * B[k * K + col];
        }
        C[row * K + col] = sum;
    }
}

int main() {
    // Warmup
    warmup<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    // Matrix dimensions
    int M = 512, N = 512, K = 512;
    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * K * sizeof(float);
    size_t size_C = M * K * sizeof(float);

    // Host pointers (using Pinned Memory for efficiency)
    float *h_A, *h_B, *h_C_naive, *h_C_cpu;
    CHECK(cudaMallocHost(&h_A, size_A));
    CHECK(cudaMallocHost(&h_B, size_B));
    CHECK(cudaMallocHost(&h_C_naive, size_C));
    h_C_cpu = (float*)malloc(size_C);

    // Initialize data
    for (int i = 0; i < M * N; i++) h_A[i] = 1.0f;
    for (int i = 0; i < N * K; i++) h_B[i] = 2.0f;

    // Device pointers
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, size_A));
    CHECK(cudaMalloc(&d_B, size_B));
    CHECK(cudaMalloc(&d_C, size_C));

    // Copy to Device
    CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // Grid and Block setup
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((K + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // Timing Setup
    cudaEvent_t start, stop;
    float ms_naive, ms_tiled;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Run Naive
    cudaEventRecord(start);
    matrix_multiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_naive, start, stop);

    CHECK(cudaMemcpy(h_C_naive, d_C, size_C, cudaMemcpyDeviceToHost));

    // Verification
    printf("Matrix Size: %d x %d x %d\n", M, N, K);
    printf("Naive Kernel Time: %.3f ms\n", ms_naive);
    
    // Check first element
    printf("Verification: Naive[0]=%.1f, (Expected %.1f)\n", 
            h_C_naive[0], (float)N * 1.0f * 2.0f);

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C_naive);
    free(h_C_cpu);

    return 0;
}