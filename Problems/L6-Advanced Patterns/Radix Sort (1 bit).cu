#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define N 512 // Must be a power of 2 for this scan
#define THREADS_PER_BLOCK (N / 2)

__global__ void phase1_flag(int *input, int *flags, int n, int bit_pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Flag is 1 if bit is 0, Flag is 0 if bit is 1
        flags[idx] = ((input[idx] >> bit_pos) & 1) == 0 ? 1 : 0;
    }
}

// Exclusive scan (prefix sum) kernel
__global__ void exclusive_scan_kernel(int *flags, int *scan, int *d_total_zeros, int n){
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory
    temp[2 * tid] = flags[2 * tid];
    temp[2 * tid + 1] = flags[2 * tid + 1];

    // Up-sweep (reduce) phase
    for (int d = n >> 1; d >0; d >>= 1){
        __syncthreads();
        if (tid < d){
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    // Clear the last element
    if (tid == 0) {
        *d_total_zeros = temp[n - 1];
        temp[n - 1] = 0;
    }

    // Down-sweep phase
    for (int d=1; d<n; d*=2){
        offset >>= 1;
        __syncthreads();

        if (tid < d){
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write results to scan array
    scan[2 * tid] = temp[2 * tid];
    scan[2 * tid + 1] = temp[2 * tid + 1];
}

// phase 3 : scatter to sorted positions
__global__ void scatter_kernel(int *input, int *output, int *scan, int total_zeros, int n, int bit_pos){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        int val = input[idx];
        int bit = (val >> bit_pos) & 1;
        int dest;

        if (bit == 0){
            dest = scan[idx];
        } else {
            int zeros_before = scan[idx];
            int ones_before = idx - zeros_before;
            dest = total_zeros + ones_before;
        }
        output[dest] = val;
    }
}

int main(){
    int h_input[N];
    for (int i=0; i<N; i++) h_input[i] = rand() % 1000;

    int *d_in, *d_out, *d_flags, *d_scan, *d_total_zeros;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMalloc(&d_flags, N * sizeof(int));
    cudaMalloc(&d_scan, N * sizeof(int));
    cudaMalloc(&d_total_zeros, sizeof(int));

    cudaMemcpy(d_in, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Sort bit by bit (32 passes for 32-bit int)
    for (int bit = 0; bit < 32; bit++) {
        // Phase 1: Flag
        phase1_flag<<<1, N>>>(d_in, d_flags, N, bit);

        // Phase 2: Scan
        exclusive_scan_kernel<<<1, THREADS_PER_BLOCK, N * sizeof(int)>>>(d_flags, d_scan, d_total_zeros, N);

        int h_total_zeros;
        cudaMemcpy(&h_total_zeros, d_total_zeros, sizeof(int), cudaMemcpyDeviceToHost);

        // Phase 3: Scatter
        scatter_kernel<<<1, N>>>(d_in, d_out, d_scan, h_total_zeros, N, bit);

        // Ping-pong buffers: output of this pass is input for next pass
        int *temp = d_in;
        d_in = d_out;
        d_out = temp;
    }

    cudaMemcpy(h_input, d_in, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify
    bool sorted = true;
    for (int i = 0; i < N - 1; i++) {
        if (h_input[i] > h_input[i + 1]) sorted = false;
    }
    printf("Sort %s!\n", sorted ? "SUCCESSFUL" : "FAILED");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_flags); cudaFree(d_scan); cudaFree(d_total_zeros);
    return 0; 
}