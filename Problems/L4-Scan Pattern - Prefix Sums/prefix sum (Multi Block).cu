#include <stdio.h>
#include <cuda_runtime.h>

// local block scan
__global__ void local_scan(const int* __restrict__ input, int* output, int* block_sums, int n){
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // load data from global mem to shared mem
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();

    // perform a local scan
    for (int offset=1; offset < blockDim.x; offset <<= 1){
        float val = 0;
        if (tid >= offset) val = sdata[tid - offset];
        __syncthreads();
        sdata[tid] += val;
        __syncthreads();
    }

    // write results to global mem
    if (idx < n) output[idx] = sdata[tid];

    // last thread of each block writes the total sum to auxi;ary array
    if (tid == blockDim.x - 1) block_sums[blockIdx.x] = sdata[tid];
}
// scan of block sums
__global__ void scan_block_sum(int* block_sums, int num_blocks){
    int tid = threadIdx.x;
    extern __shared__ int sdata_aux[];

    sdata_aux[tid] = (tid < num_blocks) ? block_sums[tid] : 0;
    __syncthreads();

    for (int offset=1; offset < blockDim.x; offset <<= 1){
        int val = 0;
        if (tid >= offset) val = sdata_aux[tid - offset];
        __syncthreads();
        sdata_aux[tid] += val;
        __syncthreads();
    }
    if (tid < num_blocks) block_sums[tid] = sdata_aux[tid];
}
// add offsets
__global__ void add_offsets(int* output, int* block_sums, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && idx < n){
        output[idx] += block_sums[blockIdx.x - 1];
    }
}

int main() {
    const int N = 1 << 20;
    size_t size = N * sizeof(int);
    int THREADS_PER_BLOCK = 1024;
    int num_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Host allocation
    int *h_in = (int*)malloc(size);
    int *h_out = (int*)malloc(size);
    for (int i = 0; i < N; i++) h_in[i] = 1;

    // Device allocation
    int *d_in, *d_out, *d_block_sums;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMalloc(&d_block_sums, num_blocks * sizeof(int));

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Pass 1: Local Scan
    local_scan<<<num_blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(d_in, d_out, d_block_sums, N);
    // Pass 2: Scan Block Sums
    scan_block_sum<<<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(d_block_sums, num_blocks);
    // Pass 3: Add Offsets
    add_offsets<<<num_blocks, THREADS_PER_BLOCK>>>(d_out, d_block_sums, N);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Verification
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != i + 1) {
            printf("Error at index %d: Expected %d, Got %d\n", i, i + 1, h_out[i]);
            success = false;
            break;
        }
    }

    if (success) printf("Success! Prefix sum calculated correctly for %d elements.\n", N);

    // Cleanup
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_block_sums);
    free(h_in); free(h_out);

    return 0;
}