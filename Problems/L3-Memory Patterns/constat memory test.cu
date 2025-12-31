#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess){ \
        fprintf(stderr, "CUDA error in '%s' int line %i: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("warmup complete.\n");
}

#define N 1024
#define COEF_SIZE 16

// constant memory declaration
__constant__ float d_coeff[COEF_SIZE];

// kernel using shared memory
__global__ void multiplyconstantmemory(float *A, float *O, int coeffsize){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <N){
        int coeffIdx = idx % coeffsize;
        O[idx] = A[idx] * d_coeff[coeffIdx];
    }
}
// kernel using global memory
__global__ void multiplyGlobalMemory(const float *input, const float *coeff, float *output, int coeffSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int coeffIdx = idx % coeffSize;
        output[idx] = input[idx] * coeff[coeffIdx];
    }
}

int main(){
    float h_i[N], float h_o_const[N], float h_o_global[N];
    float h_coeff[COEF_SIZE];
    size_t size = N * sizeof(float);

    // initalize inputs
    for (int i=0; i<N; i++) h_i[i] = i * 0.5f;
    for (int i=0; i<COEF_SIZE; i++) h_coeff[i] = 1.0f + i;

    // device memory
    float *d_i, *d_o_const, *d_o_global, *d_coef_global;
    cudaMalloc((void **)&d_i, size);
    cudaMalloc((void **)&d_o_const, size);
    cudaMalloc((void **)&d_o_global, size);
    cudaMalloc((void **)&d_coef_global, COEF_SIZE * sizeof(float));

    cudaMemcpy(d_i, h_i, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_coef_global, h_coeff, COEF_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_coeff, h_coeff, COEF_SIZE * sizeof(float));

    int threadsPerblock = 256;
    int blocks = (N + threadsPerblock - 1) / threadsPerblock;

    multiplyconstantmemory<<<blocks, threadsPerblock>>>(d_i, d_o_const, COEF_SIZE);
    multiplyGlobalMemory<<<blocks, threadsPerblock>>>(d_i, d_coef_global, d_o_global, COEF_SIZE);

    // Copy results back
    cudaMemcpy(h_i, d_o_const, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_o_global, d_o_global, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify correctness
    printf("Index | Input | ConstMemOut | GlobalMemOut\n");
    for (int i = 0; i < 16; i++) {
        printf("%d | %.2f | %.2f | %.2f\n", i, h_i[i], h_o_const[i], h_o_global[i]);
    }

    // Cleanup
    cudaFree(d_i);
    cudaFree(d_o_const);
    cudaFree(d_o_global);
    cudaFree(d_coef_global);
    free(h_i); free(h_o_const); free(h_o_global); free(h_coeff);

    return 0;

}