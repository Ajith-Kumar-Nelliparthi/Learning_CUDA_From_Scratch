#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matrixMulKernel(float *A, float *B, float *C, int N, int K, int M, int TILE_SIZE) {
    extern __shared__ float shared[];
    float* tileA = shared;
    float* tileB = shared + TILE_SIZE * TILE_SIZE;

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    for (int m = 0; m < (K + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        int tiledRow = row;
        int tiledCol = m * TILE_SIZE + threadIdx.x;

        if (tiledRow < N && tiledCol < K) {
            tileA[threadIdx.y * TILE_SIZE + threadIdx.x] = A[tiledRow * K + tiledCol];
        } else {
            tileA[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
        }

        tiledRow = m * TILE_SIZE + threadIdx.y;
        tiledCol = col;

        if (tiledRow < K && tiledCol < M) {
            tileB[threadIdx.y * TILE_SIZE + threadIdx.x] = B[tiledRow * M + tiledCol];
        } else {
            tileB[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            value += tileA[threadIdx.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < M) {
        C[row * M + col] = value;
    }
}

void randomInit(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() % 100) / 100.0f;
    }
}

int main() {
    int N = 512;
    int K = 512;
    int M = 512;

    size_t sizeA = N * K * sizeof(float);
    size_t sizeB = K * M * sizeof(float);
    size_t sizeC = N * M * sizeof(float);

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    randomInit(h_A, N * K);
    randomInit(h_B, K * M);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    int tile_sizes[] = {8, 16, 32, 64};

    for (int t = 0; t < 4; t++) {
        int TILE_SIZE = tile_sizes[t];
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
        size_t sharedMemSize = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);

        // Timing with CUDA events
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        matrixMulKernel<<<grid, block, sharedMemSize>>>(d_A, d_B, d_C, N, K, M, TILE_SIZE);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("Tile Size %d: Time = %f ms\n", TILE_SIZE, milliseconds);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}