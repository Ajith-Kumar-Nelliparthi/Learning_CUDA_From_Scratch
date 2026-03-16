#include <stdio.h>
#include <cuda_runtime.h>

__global__ void manual_unroll(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 4 + 3 < N) {
        c[idx * 4] = a[idx * 4] + b[idx * 4];
        c[idx * 4 + 1] = a[idx * 4 + 1] + b[idx * 4 + 1];
        c[idx * 4 + 2] = a[idx * 4 + 2] + b[idx * 4 + 2];
        c[idx * 4 + 3] = a[idx * 4 + 3] + b[idx * 4 + 3];
    }
}

__global__ void pragma_unroll(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx * 4;
    if (base + 3 < N) {
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            c[base + i] = a[base + i] + b[base + i];
        }
    }
}

int main() {
    const int N = 1 << 20;
    size_t size = N * sizeof(float);

    float h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    manual_unroll<<<1, N / 4>>>(d_a, d_b, d_c, N);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) {
        printf("manual_unroll: c[%d] = %f\n", i, h_c[i]);
    }
    printf("...\n");

    pragma_unroll<<<1, N / 4>>>(d_a, d_b, d_c, N);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) {
        printf("pragma_unroll: c[%d] = %f\n", i, h_c[i]);
    }
    printf("...\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}