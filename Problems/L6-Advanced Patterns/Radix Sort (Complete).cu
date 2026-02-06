#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <time.h>

#define N 1048576             // 2^20 elements
#define THREADS_PER_BLOCK 256
#define BITS 4                // Bits per pass (Radix-16)
#define BINS (1 << BITS)      // 2^4 = 16 bins
#define M (N / THREADS_PER_BLOCK) // Number of blocks

// phase-1: global histogram
__global__ void phase1_histogram(int* input, int* global_hist, int n, int bit_shift){
    extern __shared__ int local_hist[];
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    // step: a- initialize shared memory
    if (tid < BINS) local_hist[tid] = 0;
    __syncthreads();

    // step- b: each thread reads data and updates shared memory histogram
    if (idx < n){
        int val = input[idx];
        int digit = (val >> bit_shift) & (BINS - 1);
        atomicAdd(&local_hist[digit], 1);
    }
    __syncthreads();

    // step- c: block-level reduction to global histogram
    if (tid < BINS){
        global_hist[blockIdx.x * BINS + tid] = local_hist[tid];
    }
}

// phase-2: exclusive scan on histogram
__global__ void phase2_exclusive_scan(int* global_hist, int* scan_output, int total_elements){
    extern __shared__ int temp[];

    int tid = threadIdx.x;

    // step- a: load histogram into shared memory
    if (tid < total_elements) temp[tid] = global_hist[tid];
    __syncthreads();

    // step- b: perform exclusive scan (Blelloch scan)
    for (int offset = 1; offset < total_elements; offset *= 2){
        int val = 0;
        if (tid >= offset) val = temp[tid - offset];
        __syncthreads();
        if (tid >= offset) temp[tid] += val;
        __syncthreads();
    }
    // step- c: write back to global memory
    if (tid < total_elements) scan_output[tid] = (tid == 0) ? 0 : temp[tid - 1];
}

// phase-3: global scatter
__global__ void phase3_scatter(int* input, int* output, int* scan_offsets, int n, int bit_shift){
    __shared__ int local_offsets[BINS];
    __shared__ int shared_data[THREADS_PER_BLOCK];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    // load base offsets for this block
    if (tid < BINS){
        local_offsets[tid] = scan_offsets[blockIdx.x * BINS + tid];
    }

    // load input into shared memory
    int val = (idx < n) ? input[idx] : 0;
    shared_data[tid] = val;
    __syncthreads();

    if (idx < n){
        int digit = (val >> bit_shift) & (BINS - 1);
        
        int local_rank = 0;
        for (int i = 0; i < tid; i++) {
            int other_val = shared_data[i];
            int other_digit = (other_val >> bit_shift) & (BINS - 1);
            if (other_digit == digit) local_rank++;
        }

        int destination = local_offsets[digit] + local_rank;
        output[destination] = val;
    }
}