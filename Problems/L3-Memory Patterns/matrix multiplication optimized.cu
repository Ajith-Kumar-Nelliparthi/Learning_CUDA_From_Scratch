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
__global__ void matrix_multiply_optimized(const float *A, const float *B, float *C, int M, int N, int K){
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    int col = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index
    float sum = 0.0f;

    for (int t=0; t< (N + TILE_DIM - 1) / TILE_DIM; t++){
        // load tile A
        if (row < M && (t * TILE_DIM + threadIdx.x) < N){
            s_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_DIM + threadIdx.x];
        }
        else{
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // load tile B
        if (col < K && (t * TILE_DIM + threadIdx.y) < N)
            s_B[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * K + col];
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();

        for (int k=0; k<TILE_DIM; k++){
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < K) C[row * K + col] = sum;
}

int main() {
    // Matrix dimensions
    int M = 512, N = 512, K = 512;
    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * K * sizeof(float);
    size_t size_C = M * K * sizeof(float);

    // Host pointers (using Pinned Memory for efficiency)
    float *h_A, *h_B, *h_C_tiled, *h_C_cpu;
    CHECK(cudaMallocHost(&h_A, size_A));
    CHECK(cudaMallocHost(&h_B, size_B));
    CHECK(cudaMallocHost(&h_C_tiled, size_C));
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
    float ms_tiled;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Run Tiled
    cudaEventRecord(start);
    matrix_multiply_optimized<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_tiled, start, stop);

    CHECK(cudaMemcpy(h_C_tiled, d_C, size_C, cudaMemcpyDeviceToHost));

    // Verification
    printf("Matrix Size: %d x %d x %d\n", M, N, K);
    printf("Tiled Kernel Time: %.3f ms\n", ms_tiled);
    
    // Check first element
    printf("Verification: Tiled[0]=%.1f (Expected %.1f)\n", h_C_tiled[0], (float)N * 1.0f * 2.0f);

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C_tiled);
    free(h_C_cpu);

    return 0;
}