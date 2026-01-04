// Two pass algorihtm
#include <stdio.h>
#include <cuda_runtime.h>

// block level prefix max
__global__ void block_prefix_max(const int* __restrict__ A, int *out, int n){
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) sdata[tid] = A[idx];
    else sdata[tid] = INT_MIN;
    __syncthreads();

    for (int offset=1; offset < blockDim.x; offset <<=1){
        int val = sdata[tid];
        if (tid >= offset){
            val = max(sdata[tid - offset], val);
        }
        __syncthreads();
        sdata[tid] = val;
        __syncthreads();
    }
    if (idx < n) out[tid] = sdata[tid];
}
// collect block results
__global__ void collect_block_max(const int* out, int *block_max, int n){
    int block_end = (blockIdx.x + 1) * blockDim.x - 1;
    if (block_end < n){
        block_max[blockIdx.x] = out[block_end];
    } else {
        block_max[blockIdx.x] = out[n - 1];
    }
}

// propagate to blocks
__global__ void add_block_prefix(const int* block_prefix, int *out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && blockDim.x > 0){
        out[idx] = max(out[idx], block_prefix[blockIdx.x - 1]);
    }
}

int main(){
    const int n = 1 << 20;
    size_t size = n * sizeof(int);

    int *h_a = (int *)malloc(size);
    int *h_o = (int *)malloc(size);
    for (int i=0; i<n; i++){
        h_a[i] = rand() / RAND_MAX;
    }

    int *d_a, *d_o;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_o, size);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    size_t shared_mem = threads * sizeof(int);

    // Step 1: block-level prefix max
    block_prefix_max<<<blocks, threads, shared_mem>>>(d_a, d_o, n);

    // Step 2: collect block results
    int *d_block_max;
    cudaMalloc(&d_block_max, blocks * sizeof(int));
    collect_block_max<<<blocks, 1>>>(d_o, d_block_max, n);

    // Step 3: scan block results (reuse kernel)
    int *d_block_prefix;
    cudaMalloc(&d_block_prefix, blocks * sizeof(int));
    block_prefix_max<<<1, blocks, blocks * sizeof(int)>>>(d_block_max, d_block_prefix, blocks);

    // Step 4: propagate
    add_block_prefix<<<blocks, threads>>>(d_block_prefix, d_o, n);

    for (int i=0; i<257; i++){
        printf("%d", h_o[i]);
    }

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_o);
    free(h_a);
    free(h_o);

    return 0;
}