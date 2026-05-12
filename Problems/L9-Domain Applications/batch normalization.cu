#include <stdio.h>
#include <cuda_runtime.h>

__global__ void batchNorm(float *in, float *out, float *mean, float *var, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // compute mean
    float val = (idx < N) ? in[idx] : 0.0f;
    sdata[tid] = val;
    __syncthreads();

    // reduce to get the mean
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
    }
    if (tid == 0) {
        mean[blockIdx.x] = sdata[0] / N;
    }

    // compute variance
    float m = mean[0];
    val = (idx < N) ? ((in[idx] - m) * (in[idx] - m)) : 0.0f;
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
    }
    if (tid == 0) {
        var[blockIdx.x] = sdata[0] / N;
    }
    __syncthreads();

    // normalize
    if (idx < N) {
        float v = in[idx];
        out[idx] = (v - m) / sqrtf(var[0] + 1e-5f);
    }
}

int main() {
    const int N = 1024;
    float *h_in = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(N * sizeof(float));
    float *h_mean = (float*)malloc(sizeof(float));
    float *h_var = (float*)malloc(sizeof(float));

    for (int i = 0; i < N; i++) {
        h_in[i] = static_cast<float>(i);
    }

    float *d_in, *d_out, *d_mean, *d_var;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMalloc(&d_mean, sizeof(float));
    cudaMalloc(&d_var, sizeof(float));

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    batchNorm<<<1, 256, 256 * sizeof(float)>>>(d_in, d_out, d_mean, d_var, N);

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_var, d_var, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Mean: %f\n", h_mean[0]);
    printf("Variance: %f\n", h_var[0]);

    free(h_in);
    free(h_out);
    free(h_mean);
    free(h_var);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_mean);
    cudaFree(d_var);

    return 0;
}