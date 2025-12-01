#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dotProduct(const float *A, const float *B, float *result, int N){
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    while (idx < N){
        sum += A[idx] * B[idx];
        idx += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s >0; s >>= 1){
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0){
        atomicAdd(result, sdata[0]);
    }
}
// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    cudaMemset(result, 0, sizeof(float));
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t byte = threadsPerBlock * sizeof(float);
    dotProduct<<<blocks, threadsPerBlock, byte>>>(A, B, result, N);
	cudaDeviceSynchronize();
}
int main() {
	int N = 4;
	size_t bytes = N * sizeof(float);

	float h_A[] = {1.0, 2.0, 3.0, 4.0};
	float h_B[] = {5.0, 6.0, 7.0, 8.0};
	float h_result = 0.0f;

	float *d_A, *d_B, *d_result;
	cudaMalloc((void **)&d_A, bytes);
	cudaMalloc((void **)&d_B, bytes);
	cudaMalloc((void **)&d_result, sizeof(float));

	cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

	solve(d_A, d_B, d_result, N);

	cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

	printf("Dot Product: %.1f\n", h_result);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_result);
	free(h_A);
	free(h_B);

	return 0;
}