#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define THREADS_PER_BLOCK 256
#define M 1024/THREADS_PER_BLOCK // Number of blocks for histogram (N/M)

// phase-1: global histogram
__global__ void phase1_histogram(int *input, int *histogram, int n, int bit_pos, int num_bins){
    extern __shared__ float local_hist[];
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    // step: a- initialize shared memory
    if (tid < num_bins) local_hist[tid] = 0;
    __syncthreads();

    // step- b: each thread reads data and updates shared memory histogram
    if (idx < n){
        int val = input[idx];
        int digit = (val >> bit_pos) & (num_bins - 1);
        atomicAdd(&local_hist[digit], 1);
    }
    __syncthreads();

    // step- c: block-level reduction to global histogram
    if (tid < num_bins){
        int global_hist_idx = blockIdx.x * num_bins + tid;
        histogram[global_hist_idx] = local_hist[tid];
    }
}