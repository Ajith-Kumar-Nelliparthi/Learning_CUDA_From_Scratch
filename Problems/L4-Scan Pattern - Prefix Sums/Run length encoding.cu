#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void rle_kernel(int *input, int *out_vals, int *out_lens, int *total_runs, int n){
    __shared__ int temp_indices[BLOCK_SIZE];
    __shared__ int head_flags[BLOCK_SIZE];

    int tid = threadIdx.x;
    if (tid >= n) return;

    // identify heads
    int is_head = 0;
    if (tid == 0){
        is_head = 1;
    } else if (input[tid] != input[tid - 1]){
        is_head = 1;
    }

    head_flags[tid] = is_head;
    temp_indices[tid] = is_head;
    __syncthreads();

    for (int s=1; s<BLOCK_SIZE; s*=2){
        int v = 0;
        if (tid >= s) v = temp_indices[tid - s];
        __syncthreads();
        temp_indices[tid] += v;
        __syncthreads();
    }

    int out_idx = temp_indices[tid] - 1;

    if (is_head){
        out_vals[out_idx] = input[tid];
        int next_head_idx = n;
        for (int j=tid + 1; j<n; j++){
            if (input[j] != input[tid]){
                next_head_idx = j;
                break;
            }
        }
        out_lens[out_idx] = next_head_idx - tid;
    }

    if (tid == n-1){
        *total_runs = temp_indices[tid];
    }
}

int main() {
    const int n = 12;
    int h_input[] = {1, 1, 1, 2, 2, 3, 3, 3, 3, 1, 1, 4};

    int *d_input, *d_out_vals, *d_out_lens, *d_total_runs;
    int h_out_vals[n], h_out_lens[n], h_total_runs;

    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_out_vals, n * sizeof(int));
    cudaMalloc(&d_out_lens, n * sizeof(int));
    cudaMalloc(&d_total_runs, sizeof(int));

    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    rle_kernel<<<1, BLOCK_SIZE>>>(d_input, d_out_vals, d_out_lens, d_total_runs, n);

    cudaMemcpy(h_out_vals, d_out_vals, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_lens, d_out_lens, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_total_runs, d_total_runs, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Input: ");
    for(int i=0; i<n; i++) printf("%d ", h_input[i]);
    
    printf("\n\nRun-Length Encoding:\n");
    for (int i = 0; i < h_total_runs; i++) {
        printf("Value: %d, Count: %d\n", h_out_vals[i], h_out_lens[i]);
    }

    cudaFree(d_input); cudaFree(d_out_vals); cudaFree(d_out_lens); cudaFree(d_total_runs);
    return 0;
}