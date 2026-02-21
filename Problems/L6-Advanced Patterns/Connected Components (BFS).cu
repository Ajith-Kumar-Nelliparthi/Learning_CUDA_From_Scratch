#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

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


void adjListToCSR(const std::vector<std::vector<int>> &adj, 
                  std::vector<int> &row_offsets, 
                  std::vector<int> &col_indices) {
    
    int current_offset = 0;
    row_offsets.push_back(current_offset);

    for (const auto &neighbors : adj) {
        for (int neighbor : neighbors) {
            col_indices.push_back(neighbor);
        }
        current_offset += neighbors.size();
        row_offsets.push_back(current_offset);
    }
}

int main() {
    int num_nodes = 6;
    std::vector<std::vector<int>> adj_list(num_nodes);

    auto add_edge = [&](int u, int v) {
        adj_list[u].push_back(v);
        adj_list[v].push_back(u);
    };

    add_edge(0, 1);
    add_edge(1, 2);
    add_edge(0, 2);
    add_edge(3, 4);

    std::cout << "Graph Created with " << num_nodes << " nodes.\n";

    // 2. Convert to CSR for GPU
    std::vector<int> h_row_offsets; // Size: num_nodes + 1
    std::vector<int> h_col_indices; // Size: num_edges
    adjListToCSR(adj_list, h_row_offsets, h_col_indices);

    int num_edges = h_col_indices.size();

    // 3. Allocate Device Memory
    int *d_row_offsets, *d_col_indices, *d_labels;
    bool *d_changed;

    CHECK(cudaMalloc(&d_row_offsets, h_row_offsets.size() * sizeof(int)));
    CHECK(cudaMalloc(&d_col_indices, h_col_indices.size() * sizeof(int)));
    CHECK(cudaMalloc(&d_labels, num_nodes * sizeof(int)));
    CHECK(cudaMalloc(&d_changed, sizeof(bool)));

    // Copy Graph Data
    CHECK(cudaMemcpy(d_row_offsets, h_row_offsets.data(), h_row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_col_indices, h_col_indices.data(), h_col_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

    // 4. Run Connected Components Algorithm
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_nodes + threadsPerBlock - 1) / threadsPerBlock;

    // A. Initialize Labels
    init_labels<<<blocksPerGrid, threadsPerBlock>>>(d_labels, num_nodes);
    CHECK(cudaDeviceSynchronize());

    // B. Iterate until convergence
    bool h_changed = true;
    int iterations = 0;

    std::cout << "Starting CUDA Label Propagation...\n";

    while (h_changed) {
        h_changed = false;
        // Reset device flag
        CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));

        // Launch Kernel
        label_propagation<<<blocksPerGrid, threadsPerBlock>>>(
            num_nodes, d_row_offsets, d_col_indices, d_labels, d_changed
        );
        CHECK(cudaDeviceSynchronize());

        // Check if anything changed
        CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
        iterations++;
    }

    std::cout << "Converged in " << iterations << " iterations.\n";

    // 5. Retrieve Results
    std::vector<int> h_labels(num_nodes);
    CHECK(cudaMemcpy(h_labels.data(), d_labels, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));

    // 6. Print Output
    std::cout << "\nFinal Labels:\n";
    for(int i = 0; i < num_nodes; i++) {
        std::cout << "Node " << i << ": Label " << h_labels[i] << "\n";
    }

    // Free memory
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_labels);
    cudaFree(d_changed);

    return 0;
}