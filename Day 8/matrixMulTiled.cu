#include <stdio.h>
#include <cuda_runtime.h>

// Tiled matrix multiplication kernel
__global__ void matrixMulTiled(const float *a, float *b, float *c, int cols, int rows, int k_dim){
    extern __shared__ float shared_mem[];
    int tile_size = blockDim.x;
    float *tile_A = shared_mem;
    float *tile_B = &shared_mem[tile_size * tile_size];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // loop over the matrix in steps of tile_size
    for (int t=0; t< k_dim; t+= tile_size){
        // load tile A, row must be valid and the column (t + threadIdx.x ,must be within k_dim)
        if (row < rows && (t + threadIdx.x) < k_dim){
            tile_A[threadIdx.y * tile_size + threadIdx.x] = a[row * k_dim + (t + threadIdx.x)];
        } else {
            tile_A[threadIdx.y * tile_size + threadIdx.x] = 0.0f;
        }

        // load tile B, column must be valid and the row (t + threadIdx.y) must be within k_dim
        if (col < cols && (t + threadIdx.y) < k_dim){
            tile_B[threadIdx.y * tile_size + threadIdx.x] = b[(t + threadIdx.y) * cols + col];
        } else {
            tile_B[threadIdx.y * tile_size + threadIdx.x] = 0.0f;
        }

        // wait for all threads to load their data
        __syncthreads();

        // compute matrix multiplication for the tile
        for (int k=0; k<tile_size; ++k){
            sum += tile_A[threadIdx.y * tile_size + k] * tile_B[k * tile_size + threadIdx.x];
        }

        // wait for all threads to finish computing before loading new tiles
        __syncthreads();
    }
    
    // write the result
    if (row < rows && col < cols){
        c[row * cols + col] = sum;
    }
}

int main(){
    const int ROWS = 100;
    const int COLS = 100;
    const int K_DIM = 100;
    size_t size = ROWS * COLS * sizeof(float);
    const dim3 blockSizes[] = {dim3(16, 16), dim3(32, 32)};
    const int numTests = 2;

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    // Initialize matrices
    for (int i=0; i< ROWS * K_DIM; i++){
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy matrices to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    for (int t=0; t< numTests; t++){
        dim3 blockSize = blockSizes[t];
        dim3 gridSize((COLS + blockSize.x - 1) / blockSize.x, 
        (ROWS + blockSize.y - 1) / blockSize.y);

        size_t sharedMemSize = 2 * blockSize.x * blockSize.y * sizeof(float);
        printf("\nTesting block size: %dx%d (grid: %dx%d, shared mem: %zu bytes)\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y, sharedMemSize);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        matrixMulTiled<<<gridSize, blockSize, sharedMemSize>>>(d_a, d_b, d_c, COLS, ROWS, K_DIM);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess){
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            return -1;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_time = 0;
        cudaEventElapsedTime(&gpu_time, start, stop);
        printf("GPU execution time: %.3f ms\n", gpu_time);

        cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
        printf("First 5 results (row 0):\n");
        for (int j=0; j< 5; j++){
            int idx = 0 * COLS + j;
            printf("C[0][%d] = %.2f\n", j, h_c[idx]);
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}