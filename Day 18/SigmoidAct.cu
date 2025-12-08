#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void Sigmoid(const float *A, float *O, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N){
        O[idx] = 1.0f / (1.0f + expf(-A[idx]));
    }
}

void solve(const float *A, float *O, int N){
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    Sigmoid<<<blocks, threadsPerBlock>>>(A, O, N);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Cuda Error: %s\n", cudaGetErrorString(err));
    }
}

int main(){
    const int N = 1 << 24;
    size_t size = N * sizeof(float);

    float *h_i = new float[N];
    float *h_o = new float[N];

    for (int i=0; i< N; i++){
        h_i[i] = -0.5f + 10.0f * static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_i, *d_o;
    cudaMalloc((void **)&d_i, size);
    cudaMalloc((void **)&d_o, size);

    cudaMemcpy(d_i, h_i, size, cudaMemcpyHostToDevice);
    solve(d_i, d_o, N);
    cudaMemcpy(h_o, d_o, size, cudaMemcpyDeviceToHost);

    printf("First 5 sigmoid outputs: \n");
	for (int i = 0; i < 5 && i < N; i++) {
		printf("y[%d] = %f (x[%d] = %f)\n", i, h_o[i], i, h_i[i]);
	}

	cudaFree(d_i);
	cudaFree(d_o);
	delete[] h_i;
    delete[] h_o;
	return 0;
}