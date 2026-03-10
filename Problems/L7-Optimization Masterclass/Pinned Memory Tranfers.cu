#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define SIZE (1 << 26)
#define NBYTES (SIZE * sizeof(float))

double elapsed_ms(std::chrono::high_resolution_clock::time_point start,
                  std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    float *h_pageable, *h_pinned, *d_data;

    // Allocate pageable host memory
    h_pageable = (float*)malloc(NBYTES);

    // Allocate pinned host memory
    cudaMallocHost((void **)&h_pinned, NBYTES);

    cudaMalloc((void **)&d_data, NBYTES);

    for (int i = 0; i < SIZE; i++) {
        h_pageable[i] = 1.0f;
        h_pinned[i] = 1.0f;
    }

    // Pageable transfer timing
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_data, h_pageable, NBYTES, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Pageable H2D transfer: " << elapsed_ms(start, end) << " ms\n";

    // Pinned transfer timing
    start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_data, h_pinned, NBYTES, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Pinned H2D transfer:   " << elapsed_ms(start, end) << " ms\n";

    // Cleanup
    cudaFree(d_data);
    cudaFreeHost(h_pinned);
    free(h_pageable);

    return 0;

}