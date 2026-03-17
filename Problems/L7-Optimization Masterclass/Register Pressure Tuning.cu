#include <stdio.h>
#include <cuda.h>

#define N (1 << 20)   // 1M elements
#define BLOCK_SIZE 256

// Kernel with artificial register usage
__global__ void dotProduct(const float *a, const float *b, float *result) {
    __shared__ float cache[BLOCK_SIZE];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0.0f;

    // Artificially inflate register usage
    float r1 = 1.0f, r2 = 2.0f, r3 = 3.0f, r4 = 4.0f;
    float r5 = 5.0f, r6 = 6.0f, r7 = 7.0f, r8 = 8.0f;

    while (tid < N) {
        temp += a[tid] * b[tid];
        // Dummy math to keep registers busy
        temp += (r1 + r2 + r3 + r4) * (r5 + r6 + r7 + r8);
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    // Reduction in shared memory
    int i = BLOCK_SIZE / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        atomicAdd(result, cache[0]);
}

int main() {
    float *a, *b, *d_a, *d_b, *d_result;
    float result = 0.0f;

    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result, sizeof(float), cudaMemcpyHostToDevice);

    dotProduct<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_result);

    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Dot product result = %f\n", result);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    free(a);
    free(b);

    return 0;
}