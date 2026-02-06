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
__global__ void histogram_kernel(int* input, int* global_hist, int n, int bit_shift){
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
__global__ void scan_kernel(int* global_hist, int* scan_output, int total_elements){
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
__global__ void scatter_kernel(int* input, int* output, int* scan_offsets, int n, int bit_shift){
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

int main() {
    size_t size = N * sizeof(int);
    int *h_in = (int*)malloc(size);
    for (int i = 0; i < N; i++) h_in[i] = rand() % 1000000;

    int *d_in, *d_out, *d_hist, *d_scan;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMalloc(&d_hist, M * BINS * sizeof(int));
    cudaMalloc(&d_scan, M * BINS * sizeof(int));

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    // RADIX SORT LOOP (8 passes for 32 bits, 4 bits at a time) ---
    for (int bit = 0; bit < 32; bit += BITS) {
        histogram_kernel<<<M, THREADS_PER_BLOCK>>>(d_in, d_hist, N, bit);
        
        // Scan the entire M * BINS histogram table
        // We use 1 block, assuming M*BINS <= 1024
        scan_kernel<<<1, 1024>>>(d_hist, d_scan, M * BINS);
        
        scatter_kernel<<<M, THREADS_PER_BLOCK>>>(d_in, d_out, d_scan, N, bit);

        // Ping-pong pointers
        int* temp = d_in;
        d_in = d_out;
        d_out = temp;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // THRUST COMPARISON
    thrust::device_vector<int> d_thrust(h_in, h_in + N);
    clock_t t_start = clock();
    thrust::sort(d_thrust.begin(), d_thrust.end());
    clock_t t_end = clock();
    float thrust_ms = (float)(t_end - t_start) * 1000.0 / CLOCKS_PER_SEC;

    // Verify
    cudaMemcpy(h_in, d_in, size, cudaMemcpyDeviceToHost);
    bool success = true;
    for (int i = 0; i < N - 1; i++) {
        if (h_in[i] > h_in[i+1]) { success = false; break; }
    }

    printf("Sort %s\n", success ? "SUCCESSFUL" : "FAILED");
    printf("Custom Radix Sort Time: %f ms\n", ms);
    printf("Thrust Sort Time:       %f ms\n", thrust_ms);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_hist); cudaFree(d_scan);
    free(h_in);
    return 0;
}