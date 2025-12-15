#include <stdio.h>
#include <cuda_runtime.h>

__global__ void conv_1D(const float* input, const float* kernel, float* output, int input_size, int kernel_size){
    extern __shared__ float sdata[];

    int output_size = input_size - kernel_size + 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    for (int i=tid; i < kernel_size; i += blockDim.x){
        sdata[i] = kernel[i];
    }
    __syncthreads();

    if (idx >= output_size) return;
    float sum = 0.0f;
    for(int j=0; j<kernel_size; j++){
        sum += input[idx + j] * sdata[j];
    }
    output[idx] = sum;
}
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
	int output_size = input_size - kernel_size + 1;
	int threadsPerBlock = 1024;
	int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
	size_t kMemBytes = kernel_size * sizeof(float);

	conv_1D<<<blocksPerGrid, threadsPerBlock, kMemBytes>>>(input, kernel, output, input_size, kernel_size);
	cudaDeviceSynchronize();
}