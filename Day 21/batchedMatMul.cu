#include <cuda_runtime.h>

#define BLOCK_SIZE 16
// A, B, C are device pointers
__global__ void batchedMatMul (const float* A, const float* B, float* C, int M, int N, int K){
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N){
        float sum = 0.0f;
        for (int k=0; k<K; k++){
            sum += A[batch * M * K + row * K + k] * B[batch * N * K + k * N + col];
        }
        C[batch * M * N + row * N + col] = sum;
    }
}
extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE, BATCH);

	batchedMatMul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
	cudaDeviceSynchronize();
}