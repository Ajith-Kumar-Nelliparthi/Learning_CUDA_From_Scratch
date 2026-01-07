#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// kernel-1: mark non zero elements
__global__ void mark_nonzeros(const float* input, int n, int* flags){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        flags[idx] = (input[idx] != 0.0f) ? 1 : 0;
    }
}

// kernel-2: inclusive scan on flags to get write positions
__global__ void inclusive_scan(int *data, int n){
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? data[idx] : 0;
    __syncthreads();

    for (int stride=1; stride < blockDim.x; stride *= 2){
        int temp = 0;
        if (tid >= stride){
            temp = sdata[tid - stride];
        }
        __syncthreads();
        if (tid >= stride){
            sdata[tid] += temp;
        }
        __syncthreads();
    }
    if (idx < n){
        data[idx] = sdata[tid];
    }
}

// convert inclusive to exclusive scan in-place
__global__ void to_exclusive(int *scan, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        if (idx == 0){
            scan[idx] = 0;
        } else {
            scan[idx] = scan[idx - 1];
        }
    }
}

// kernel-3: scatter non zero elements and indices
__global__ void scatter_nonzero(const float* input, const int* position, int n, float* values_out, int* indices_out){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && input[idx] != 0.0f){
        int pos = position[idx];
        values_out[pos] = input[idx];
        indices_out[pos] = idx;
    }
}

int main() {
    const int n = 16;
    float h_input[] = {0.0f, 3.0f, 0.0f, 0.0f, 7.0f, 0.0f, 2.0f, 0.0f,
                       5.0f, 0.0f, 1.0f, 0.0f, 0.0f, 4.0f, 0.0f, 9.0f};

    // Device pointers
    float *d_input, *d_values;
    int *d_flags, *d_positions, *d_indices;

    // Allocate memory
    cudaMalloc(&d_input,     n * sizeof(float));
    cudaMalloc(&d_flags,     n * sizeof(int));
    cudaMalloc(&d_positions, n * sizeof(int));
    cudaMalloc(&d_values,    n * sizeof(float));
    cudaMalloc(&d_indices,   n * sizeof(int));

    // Copy input
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads = BLOCK_SIZE;
    int blocks  = (n + threads - 1) / threads;

    // Pass 1: Mark non-zeros
    mark_nonzeros<<<blocks, threads>>>(d_input, n, d_flags);
    cudaDeviceSynchronize();

    // Pass 2: Inclusive scan on flags
    int shared_bytes = BLOCK_SIZE * sizeof(int);
    inclusive_scan<<<blocks, threads, shared_bytes>>>(d_flags, n);
    cudaDeviceSynchronize();

    // Convert to exclusive scan
    to_exclusive<<<blocks, threads>>>(d_flags, n);
    cudaDeviceSynchronize();

    // Copy positions back to use as output addresses
    cudaMemcpy(d_positions, d_flags, n * sizeof(int), cudaMemcpyDeviceToDevice);

    // Get nnz: last position + last flag
    int last_pos, last_flag;
    cudaMemcpy(&last_pos,  &d_positions[n-1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_flag, &d_flags[n-1],      sizeof(int), cudaMemcpyDeviceToHost);
    int nnz = last_pos + last_flag;
    printf("Number of non-zero elements: %d\n", nnz);

    // Pass 3: Scatter values and indices
    scatter_nonzero<<<blocks, threads>>>(d_input, d_positions, n, d_values, d_indices);
    cudaDeviceSynchronize();

    // Copy results back
    float* h_values = new float[nnz];
    int*   h_indices = new int[nnz];
    cudaMemcpy(h_values,  d_values,  nnz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices, d_indices, nnz * sizeof(int),   cudaMemcpyDeviceToHost);

    // Print results
    printf("Non-zero values: ");
    for (int i = 0; i < nnz; ++i) printf("%.0f ", h_values[i]);
    printf("\n");

    printf("Indices: ");
    for (int i = 0; i < nnz; ++i) printf("%d ", h_indices[i]);
    printf("\n");

    // Cleanup
    delete[] h_values;
    delete[] h_indices;
    cudaFree(d_input); cudaFree(d_flags); cudaFree(d_positions);
    cudaFree(d_values); cudaFree(d_indices);

    return 0;
}