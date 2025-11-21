#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

// Initalize kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < size; idx += blockDim.x * gridDim.x) {
        C[idx] = A[idx] + B[idx];
    }
}

void vectorAddCPU(const float *A, const float *B, float *C, int size) {
    for(int i=0; i<size; i++) {
        C[i] = A[i] + B[i];
    }
}

int main(){
    // vector size
    const int N = 1000000;
    size_t size = N * sizeof(float);

    // allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);
    float *h_c_cpu = (float *)malloc(size);

    // initialize vectors
    for (int i=0; i< N; i++){
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // copy vectors from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // timing GPU vector addition
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU Vector Addition Time: %f ms\n", gpu_time);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // timing CPU vector addition
    auto cpu_start = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_a, h_b, h_c_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    float cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    printf("CPU Vector Addition Time: %f ms\n", cpu_time);

    bool correct = true;
	for (int i = 0; i < N; i++) {
		if (fabs(h_c[i] - h_c_cpu[i]) > 1e-5) {
			printf("verification failed at index %d: GPU = %.2f, CPU = %.2f\n", i, h_c[i], h_c_cpu[i]);

			correct = false;
			break;
		}
	}

	if (correct) {
		printf("vector addition completed successfully\n");
		printf("speedup: %.2fX\n", cpu_time / gpu_time);
	}

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_cpu);

    return 0;
}