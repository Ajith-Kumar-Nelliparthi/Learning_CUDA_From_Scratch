#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>

#define N 10000
#define K 3
#define DIM 2
#define MAX_ITER 100
#define THREADS_PER_BLOCK 256

__global__ void assign_clusters(const float* data, float* centeroids, int* labels, float *new_sums, int* counts){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float min_dist = FLT_MAX;
    int best_cluster = 0;

    // find the closest centeroid
    for (int k=0; k<K; k++){
        float dist = 0.0f;
        for (int d=0; d<DIM; d++){
            float diff = data[idx * DIM + d] - centeroids[k * DIM + d];
            dist += diff * diff;
        }
        if (dist < min_dist){
            min_dist = dist;
            best_cluster = k;
        }
    }

    // assign the label
    labels[idx] = best_cluster;

    // atomic add to the new sums and counts
    for (int d=0; d<DIM; d++){
        atomicAdd(&new_sums[best_cluster * DIM + d], data[idx * DIM + d]);
    }
    atomicAdd(&counts[best_cluster], 1);
}

__global__ void centeroid_update(float *centeroids, float *new_sums, int* counts){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;

    if (counts[idx] > 0){
        for (int d=0; d<DIM; d++){
            centeroids[idx * DIM + d] = new_sums[idx * DIM + d] / counts[idx];
        }
    }
}

int main(){
    size_t points_size = N * DIM * sizeof(float);
    size_t centeroids_size = K * DIM * sizeof(float);

    std::vector<float> h_points(N * DIM);
    std::vector<float> h_centeroids(K * DIM);

    for (int i=0; i<N*DIM; i++){
        h_points[i] = (float)rand() / RAND_MAX * 100.0f;
    }
    for (int i=0; i<K*DIM; i++){
        h_centeroids[i] = h_points[i]; // initialize centeroids to first K points
    }

    // Allocate device memory
    float *d_points, *d_centeroids, *d_new_sums;
    int *d_labels, *d_counts;

    cudaMalloc(&d_points, points_size);
    cudaMalloc(&d_centeroids, centeroids_size);
    cudaMalloc(&d_labels, N * sizeof(int));
    cudaMalloc(&d_new_sums, centeroids_size);
    cudaMalloc(&d_counts, K * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_points, h_points.data(), points_size, cudaMemcpyHostToDevice);

    int num_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    for (int iter=0; iter<MAX_ITER; iter++){
        cudaMemset(d_new_sums, 0, centeroids_size);
        cudaMemset(d_counts, 0, K * sizeof(int));
        cudaMemcpy(d_centeroids, h_centeroids.data(), centeroids_size, cudaMemcpyHostToDevice);

        assign_clusters<<<num_blocks, THREADS_PER_BLOCK>>>(d_points, d_centeroids, d_labels, d_new_sums, d_counts);
        centeroid_update<<<1, K>>>(d_centeroids, d_new_sums, d_counts);

        cudaMemcpy(h_centeroids.data(), d_centeroids, centeroids_size, cudaMemcpyDeviceToHost);
    }

    std::cout << "K-Means Completed." << std::endl;
    std::cout << "Final Centroids for " << K << " clusters:" << std::endl;
    for (int i = 0; i < K; i++) {
        std::cout << "Cluster " << i << ": (" << h_centeroids[i * DIM] << ", " << h_centeroids[i * DIM + 1] << ")" << std::endl;
    }

    // Clean up
    cudaFree(d_points);
    cudaFree(d_centeroids);
    cudaFree(d_labels);
    cudaFree(d_new_sums);
    cudaFree(d_counts);

    return 0;
}