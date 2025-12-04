#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void L2Norm(const float *input, float *output, float *globalsum, int N){
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float squares = 0.0f;
    if (idx < N){
        squares = input[idx] * input[idx];
    }
    sdata[tid] = squares;
    __syncthreads();

    for (int stride=blockDim.x/2; stride > 0; stride >>= 1){
        if (tid < stride){
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0){
        atomicAdd(globalsum, sdata[0]);
    }
    __syncthreads();

    volatile float *norm = globalsum;
    float L2norm = sqrt(*norm);
    if (idx < N && L2norm > 0.0f){
        output[idx] = input[idx] / L2norm;
    }
}
void solve(const float *input, float *output, int N){
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *d_globalsum;
    cudaMalloc((void **)&d_globalsum, sizeof(float));
    cudaMemset(d_globalsum, 0, sizeof(float));

    L2Norm<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(input, output, d_globalsum, N);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaFree(d_globalsum);
}

int main() {
	int N = 1 << 20;
	size_t bytes = N * sizeof(float);

	float* h_input = new float[N];
	float* h_output = new float[N];

	for (int i = 0; i < N; i++) {
		h_input[i] = static_cast<float>(rand()) / RAND_MAX;
	}

	float *d_input, *d_output;
	cudaMalloc((void **)&d_input, bytes);
	cudaMalloc((void **)&d_output, bytes);

	cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

	solve(d_input, d_output, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    printf("GPU execution time: %.3f ms\n", ms);
	cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

	printf("First 5 normalized values:\n");
	for (int i = 0; i < 5 && i < N; i++) {
		printf("output[%d] = %f (input[%d] = %f)\n", i, h_output[i], i, h_input[i]);
	}

	cudaFree(d_input);
	cudaFree(d_output);
	free(h_input);
	free(h_output);
	return 0;
	}