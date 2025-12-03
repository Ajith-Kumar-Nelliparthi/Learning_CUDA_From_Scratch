#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hadamardProduct(const float *A, const float *B, float *C, int m, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // y for rows
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // x for columns

    if (row < m && col < n){
        int idx = row * n + col;
        C[idx] = A[idx] * B[idx];
    }
}

void solve(const float *A, const float *B, float *C, int m, int n){
    const int BLOCK_SIZE = 16;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    hadamardProduct<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, m, n);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

int main() {
	int m = 1024;
	int n = 512;
	size_t bytes = m * n * sizeof(float);

	float *h_A = new float[m * n];
	float *h_B = new float[m * n];
	float *h_C = new float[m * n];

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			int idx = i * n + j;
			h_A[idx] = static_cast<float>(i + j + 1);
			h_B[idx] = static_cast<float>(1.0 / (i + j + 1));
		}
	}

	float *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, bytes);
	cudaMalloc((void **)&d_B, bytes);
	cudaMalloc((void **)&d_C, bytes);

	cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

	solve(d_A, d_B, d_C, m, n);

	cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

	printf("First 5 elements: \n");
	for (int i = 0; i < 5 && i < m * n; i++) {
		printf("C[%d] = %f (A[%d] = %f, B[%d] = %f)\n", i, h_C[i], i, h_A[i], i, h_B[i]);
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

	return 0;
}