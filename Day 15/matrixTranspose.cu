#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define M 1024   // No.of rows
#define N 1024   // No.of cols
#define TILE_SIZE 32   // 32 * 32 tiles

__global__ void matrixtranspose(float *in, float *out, int rows, int cols){
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // read input into shared memory
    if (x < cols && y < rows){
        tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    }
    __syncthreads();

    // compute transposed co-ordinates
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    // write transposed data to output
    if (x < rows && y < cols){
        out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Initalize matrix with random values
void inimatrix(float *mat, int rows, int cols){
    for (int i=0; i< rows * cols; i++){
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main(){
    int rows = M;
    int cols = N;

    size_t size_in = rows * cols * sizeof(float);
    size_t size_out = rows * cols * sizeof(float);

    float *h_in = (float *)malloc(size_in);
    float *h_out = (float *)malloc(size_out);

    // initalize input
    srand(2025);
    inimatrix(h_in, rows, cols);

    float *d_in, *d_out;
    cudaMalloc(&d_in, size_in);
    cudaMalloc(&d_out, size_out);

    cudaMemcpy(d_in, h_in, size_in, cudaMemcpyHostToDevice);

    // launch kernel
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((cols + TILE_SIZE - 1) / TILE_SIZE,
                    (rows + TILE_SIZE - 1) / TILE_SIZE);

    // warmup
    matrixtranspose<<<gridSize, blockSize>>>(d_in, d_out, rows, cols);
    cudaDeviceSynchronize();

    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixtranspose<<<gridSize, blockSize>>>(d_in, d_out, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Time taken for gpu %.3f\n", gpu_time);

    cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost);
    printf("Matrix size: %dx%d, Tiled transpose time: %.3f ms\n", rows, cols, gpu_time);
    printf("Effective bandwidth: %.2f GB/s\n",
           (double)(rows * cols * sizeof(float) * 2) / (gpu_time / 1000.0) / 1e9);

    // Cleanup
    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}