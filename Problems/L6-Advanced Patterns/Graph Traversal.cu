#include <stdio.h>
#include <cuda_runtime.h>

__global__ void bfs_kernel(int num_nodes, const int* row_offsets, const int* col_indices, int* distances, int* changed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_nodes && distances[idx] != -1) {
        int current_distance = distances[idx];
        int row_start = row_offsets[idx];
        int row_end = row_offsets[idx + 1];

        for (int i=row_start; i< row_end; i++) {
            int neighbour = col_indices[i];
            if (atomicCAS(&distances[neighbour], -1, current_distance + 1) == -1) {
                atomicOr(changed, true);
            }
        }
    }
}

int main() {
    int num_nodes = 4;
    int h_row_offsets[] = {0, 2, 3, 5, 6};
    int h_col_indices[] = {1, 2, 2, 0, 3, 3};

    int *d_row_offsets, *d_col_indices, *d_distances, *d_changed;
    cudaMalloc(&d_row_offsets, (num_nodes + 1) * sizeof(int));
    cudaMalloc(&d_col_indices, 6 * sizeof(int));
    cudaMalloc(&d_distances, num_nodes * sizeof(int)); 
    cudaMalloc(&d_changed, sizeof(int));

    cudaMemcpy(d_row_offsets, h_row_offsets, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, h_col_indices, 6 * sizeof(int), cudaMemcpyHostToDevice);

    int h_distances[4];
    for (int i=0; i<num_nodes; i++) h_distances[i] = -1;
    h_distances[0] = 0; // Starting node
    cudaMemcpy(d_distances, h_distances, num_nodes * sizeof(int), cudaMemcpyHostToDevice);

    int has_changed;
    do {
        has_changed = 0;
        cudaMemcpy(d_changed, &has_changed, sizeof(int), cudaMemcpyHostToDevice);

        int threads_per_block = 256;
        int blocks = (num_nodes + threads_per_block - 1) / threads_per_block;
        bfs_kernel<<<blocks, threads_per_block>>>(num_nodes, d_row_offsets, d_col_indices, d_distances, d_changed);

        cudaDeviceSynchronize();

        cudaMemcpy(&has_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    } while (has_changed);

    cudaMemcpy(h_distances, d_distances, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Node distances from root (0):\n");
    for (int i = 0; i < num_nodes; i++) {
        printf("Node %d: %d\n", i, h_distances[i]);
    }

    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_distances);
    cudaFree(d_changed);
    return 0;
}