#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void distancekernel(const float *a, const float *b, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float dx = a[idx] - b[idx];
        out[idx] = sqrtf(dx * dx);
    }
}

__global__ void fast_math(const float *a, const float *b, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float dx = a[idx] - b[idx];
        out[idx] = __fsqrt_rn(dx * dx);
    }
}

int main() {
    const int N = 1 << 20;
    size_t size = N * sizeof(float);
    float *a, *b, *out, *out_fast;
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&out, size);
    cudaMallocManaged(&out_fast, size);

    for (int i = 0; i < N; i++) {
        a[i] = sinf(i * 0.01f);
        b[i] = cosf(i * 0.01f);
    }

    distancekernel<<<(N + 255) / 256, 256>>>(a, b, out, N);
    fast_math<<<(N + 255) / 256, 256>>>(a, b, out_fast, N);
    cudaDeviceSynchronize();

    double maxError = 0.0;
    for (int i = 0; i < N; i++) {
        double err = fabs(out[i] - out_fast[i]);
        if (err > maxError) {
            maxError = err;
        }
    }
    printf("Max error: %e\n", maxError);

    cudaFree(a); cudaFree(b); cudaFree(out); cudaFree(out_fast);
    return 0;
}