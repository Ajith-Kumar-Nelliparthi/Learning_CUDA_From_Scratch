#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess){ \
        fprintf(stderr, "CUDA error in '%s' at line %i: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define N 1024
#define COEF_SIZE 16

// Constant memory
__constant__ float d_coeff[COEF_SIZE];

// Kernel using constant memory
__global__ void multiplyConstantMemory(float *A, float *O, int coeffsize){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        int coeffIdx = idx % coeffsize;
        O[idx] = A[idx] * d_coeff[coeffIdx];
    }
}

// Kernel using global memory
__global__ void multiplyGlobalMemory(const float *input, const float *coeff, float *output, int coeffSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int coeffIdx = idx % coeffSize;
        output[idx] = input[idx] * coeff[coeffIdx];
    }
}

int main(){
    float h_i[N], h_o_const[N], h_o_global[N], h_coeff[COEF_SIZE];
    size_t size = N * sizeof(float);

    // Initialize inputs
    for (int i=0; i<N; i++) h_i[i] = i * 0.5f;
    for (int i=0; i<COEF_SIZE; i++) h_coeff[i] = 1.0f + i;

    // Device memory pointers
    float *d_i, *d_o_const, *d_o_global, *d_coef_global;
    
    CHECK(cudaMalloc((void **)&d_i, size));
    CHECK(cudaMalloc((void **)&d_o_const, size));
    CHECK(cudaMalloc((void **)&d_o_global, size));
    CHECK(cudaMalloc((void **)&d_coef_global, COEF_SIZE * sizeof(float)));

    // Copy data to device
    CHECK(cudaMemcpy(d_i, h_i, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_coef_global, h_coeff, COEF_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Copy to Constant Memory symbol
    CHECK(cudaMemcpyToSymbol(d_coeff, h_coeff, COEF_SIZE * sizeof(float)));

    int threadsPerblock = 256;
    int blocks = (N + threadsPerblock - 1) / threadsPerblock;

    // Launch Kernels
    multiplyConstantMemory<<<blocks, threadsPerblock>>>(d_i, d_o_const, COEF_SIZE);
    multiplyGlobalMemory<<<blocks, threadsPerblock>>>(d_i, d_coef_global, d_o_global, COEF_SIZE);
    
    // Check for kernel launch errors
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Copy results back to correct host arrays
    CHECK(cudaMemcpy(h_o_const, d_o_const, size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_o_global, d_o_global, size, cudaMemcpyDeviceToHost));

    // Verify correctness
    printf("Idx | Input | ConstMemOut | GlobalMemOut\n");
    printf("----------------------------------------\n");
    for (int i = 0; i < 10; i++) {
        printf("%02d  | %.2f  | %.2f      | %.2f\n", i, h_i[i], h_o_const[i], h_o_global[i]);
    }

    // Cleanup Device Memory
    cudaFree(d_i);
    cudaFree(d_o_const);
    cudaFree(d_o_global);
    cudaFree(d_coef_global);
    
    return 0;
}