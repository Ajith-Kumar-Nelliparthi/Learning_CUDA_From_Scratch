#include <stdio.h>
#include <cuda_runtime.h>

// 1. create mask kernel
__global__ void compute_mask(const int* input, int* mask, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        mask[idx] = (input[idx] != 0) ? 1 : 0;
    }
}

// 2. inclusive scan (hills - steele)
__global__ void inclusive_scan(int* mask, int* scanned, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        scanned[idx] = mask[idx];
    }
    __syncthreads();

    for (int stride=1; stride<n; stride *=2){
        int temp = 0;
        if (idx >= stride){
            temp = scanned[idx - stride];
        }
        __syncthreads();
        if (idx >= stride){
            scanned[idx] += temp;
        }
        __syncthreads();
    }
}

// 3. scatter kernel
__global__ void scatter(const int* input, const int* mask, const int* scanned, int *output, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        if (mask[idx] == 1){
            int dest = scanned[idx] - 1;
            output[dest] = input[idx];
        }
    }
}

int main() {
    const int N = 8;
    int h_input[N] = {1, 0, 2, 0, 3, 0, 4, 5};
    int h_output[N] = {0};
    int count = 0;

    int *d_input, *d_mask, *d_scanned, *d_output;

    // Allocate Device Memory
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_mask, N * sizeof(int));
    cudaMalloc(&d_scanned, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    int threadsPerBlock = N;
    int blocksPerGrid = 1;

    // Step 1: Create Mask
    compute_mask<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_mask, N);

    // Step 2: Scan Mask
    inclusive_scan<<<blocksPerGrid, threadsPerBlock>>>(d_mask, d_scanned, N);

    // Step 3: Scatter
    scatter<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_mask, d_scanned, d_output, N);

    // Copy Result back to Host
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&count, &d_scanned[N-1], sizeof(int), cudaMemcpyDeviceToHost);

    // Print Results
    printf("Input:  [");
    for(int i=0; i<N; i++) printf("%d%s", h_input[i], i==N-1?"":", ");
    printf("]\n");

    printf("Output: [");
    for(int i=0; i<count; i++) printf("%d%s", h_output[i], i==count-1?"":", ");
    printf("]\n");

    printf("Count: %d\n", count);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_scanned);
    cudaFree(d_output);

    return 0;
}