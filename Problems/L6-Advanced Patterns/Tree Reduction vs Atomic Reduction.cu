#include <stdio.h>
#include <cuda_runtime.h>

__global__ void tree_reduction(int *data, int *result, int n){
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? data[idx] : 0;
    __syncthreads();

    for (int s=blockDim.x/2; s>0; s>>=1){
        if (tid < s){
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0){
        result[blockIdx.x] = sdata[0];
    }
}

// atomic reduction
__global__ void atomic_reduction(int *data, int *result, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n){
        atomicMax(result, data[idx]);
    }
}

int main(){
    const int n = 1 << 16;
    size_t size = n * sizeof(int);
    int threadsPerblock = 256;
    int block = (n + threadsPerblock - 1) / threadsPerblock;

    int *h_data = (int *)malloc(size);
    for (int i=0; i<n; i++) h_data[i] = rand();

    int *d_data, *d_result;
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_result, sizeof(int) * block);

    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    tree_reduction<<<block, threadsPerblock, threadsPerblock * sizeof(int)>>>(d_data, d_result, n);
    int *h_result = (int *)malloc(sizeof(int) * block);
    cudaMemcpy(h_result, d_result, sizeof(int) * block, cudaMemcpyDeviceToHost);

    int max_tree = 0;
    for (int i=0; i<block; i++){
        if (h_result[i] > max_tree) max_tree = h_result[i];
    }

    int max_atomic;
    cudaMemset(d_result, 0, sizeof(int));
    atomic_reduction<<<block, threadsPerblock>>>(d_data, d_result, n);
    cudaMemcpy(&max_atomic, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Max from tree reduction: %d\n", max_tree);
    printf("Max from atomic reduction: %d\n", max_atomic);

    free(h_data);
    free(h_result);
    cudaFree(d_data);
    cudaFree(d_result);

    return 0;
}