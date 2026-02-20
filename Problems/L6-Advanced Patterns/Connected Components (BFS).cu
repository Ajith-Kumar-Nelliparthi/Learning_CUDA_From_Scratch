#include <stdio.h>
#include <cuda_runtime.h>

// cuda error macro
#define CHECK(call) \
do{ \
    cudaError_t err = call; \
    if (err != cudaSuccess){ \
        fprintf(stderr, "cuda error in '%s' at line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// initalize labels
__global__ void init_labels(int *labels, int num_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        labels[idx] = idx; // each node is its own component initially
    }
}

// label propagation kernel
__global__ void label_propagation(int num_nodes, const int* row_offsets, const int* col_indices, int *labels, bool *changed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_nodes) {
        int original_label = labels[idx];
        int min_label = original_label;

        int start_edge = row_offsets[idx];
        int end_edge = row_offsets[idx + 1];

        for (int edge = start_edge; edge < end_edge; edge++) {
            int vertice = col_indices[edge];
            int vertice_label = labels[vertice];

            if (vertice_label < min_label) {
                min_label = vertice_label;
            }
        }

        if (min_label < original_label) {
            atomicMin(&labels[idx], min_label); // update label to the smallest label found
            *changed = true; // mark that a change has occurred
        }
    }
}

