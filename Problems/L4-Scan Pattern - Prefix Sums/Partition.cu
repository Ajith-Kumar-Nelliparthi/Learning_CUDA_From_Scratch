#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
__global__ void partition1(int* input, int* out, int pivot, int n,
                          int* lessCount, int* equalCount, int* greaterCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int val = input[idx];

    if (val < pivot) {
        int pos = atomicAdd(lessCount, 1);
        out[pos] = val;
    } else if (val == pivot) {
        int pos = atomicAdd(equalCount, 1);
        out[*lessCount + pos] = val;
    } else {
        int pos = atomicAdd(greaterCount, 1);
        out[*lessCount + *equalCount + pos] = val;
    }
}

__global__ void partition2(int* input, int *output, int *scan_array, int n, int pivot, int *total_true_count){
    __shared__ int temp[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int val = 0;
    int flag = 0;

    if (idx < n){
        val = input[idx];
        flag = (val < pivot) ? 1 : 0;
    }

    // load flags into shared memory for scan
    temp[tid] = flag;
    __syncthreads();

    // partial prefix sum within the block
    for (int offset=1; offset < BLOCK_SIZE; offset *= 2){
        int v = 0;
        if (tid >= offset){
            v = temp[tid - offset];
        }
        __syncthreads();
        temp[tid] += v;
        __syncthreads();
    }
    int exclusive_scan = temp[tid] - flag;
    // last element holds true elements in the block
    if (tid == BLOCK_SIZE - 1 || idx == n - 1){
        *total_true_count = temp[tid];
    }
    __syncthreads();

    // scatter using dual scan logic
    if (idx < n){
        if (flag == 1){
            output[exclusive_scan] = val;
        } else {
            int flag_pos = (*total_true_count) + (idx - exclusive_scan);
            output[flag_pos] = val;
        }
    }
}

int main(){
    const int n = 10;
    const int pivot = 5;
    size_t size = n * sizeof(int);
    int h_input[n] = {8, 1, 4, 9, 2, 7, 3, 5, 0, 6};
    int h_output[n];

    int *d_i, *d_o, *d_total_true;
    cudaMalloc((void **)&d_i, size);
    cudaMalloc((void **)&d_o, size);
    cudaMalloc((void **)&d_total_true, sizeof(int));

    cudaMemcpy(d_i, h_input, size, cudaMemcpyHostToDevice);
    partition2<<<1, BLOCK_SIZE>>>(d_i, d_o, NULL, n, pivot, d_total_true);
    cudaMemcpy(h_output, d_o, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i=0; i<n; i++){
        printf("%d ", h_output[i]);
    }

    cudaFree(d_i); cudaFree(d_o); cudaFree(d_total_true);
    free(h_input); free(h_output);

    return 0;
}