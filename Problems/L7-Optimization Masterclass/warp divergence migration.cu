#include <stdio.h>
#include <cuda_runtime.h>

// naive
__global__ void warp_divergence(float *in, float *out, int threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = in[idx];
    if (val > threshold) {
        out[idx] = val * 2.0f;
    } else {
        out[idx] = val;
    }
}

// optimized
__global__ void warp_divergence_optimized(float *in, float *out, int threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = in[idx];

    out[idx] = (val > threshold) ? val * 2.0f : val;
}