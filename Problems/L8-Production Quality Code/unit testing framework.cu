#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <cassert>
#include <iostream>
#include <cmath>

__global__ void testkernel(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

void testkernelcpu(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

