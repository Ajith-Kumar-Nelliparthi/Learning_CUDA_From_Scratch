#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixMultiply(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float value = 0;
        for (int i = 0; i < N; i++) {
            value += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = value;
    }
}

int main() {
    int M = 2, N = 3, K = 2;
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    A = (float *)malloc(M * N * sizeof(float));
    B = (float *)malloc(N * K * sizeof(float));
    C = (float *)malloc(M * K * sizeof(float));

    // Initialize A and B
    for (int i = 0; i < M * N; i++) A[i] = i + 1;
    for (int i = 0; i < N * K; i++) B[i] = i + 1;

    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));

    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((K + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Result of A x B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("%.2f ", C[i * K + j]);
        }
        printf("\n");
    }

    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}