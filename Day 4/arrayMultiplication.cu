#include <stdio.h>
#include <cuda_runtime.h>

__global__ void arrayMultiply(const float *a, const float *b, float *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(; idx < size; idx += blockDim.x * gridDim.x) {
        c[idx] = a[idx] * b[idx];
    }
}

int main(){
    const int N = 10000;
    size_t size = N * sizeof(float);

    const int blocksizes[] = {64, 128, 256};
    const int numtests = 3;

    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    for (int i=0; i<N; i++){
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    for (int t=0; t<numtests; t++){
        int threadsPerBlock = blocksizes[t];
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

        printf("\nTesting block size: %d (blocks: %d)\n", threadsPerBlock, blocks);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        arrayMultiply<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_time = 0;
        cudaEventElapsedTime(&gpu_time, start, stop);
        printf("GPU execution time: %.3f ms\n", gpu_time);

		cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

		printf("First 5 results:\n");
		for (int i = 0; i < 5; i++) {
			printf("c[%d] = %.2f (a[%d] = %.2f * b[%d] = %.2f)\n", i, h_c[i], i, h_a[i], i, h_b[i]);
		}

		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(h_a);
	free(h_b);
	free(h_c);

	return 0;

}