#include <stdio.h>
#include <cuda_runtime.h>

__global__ void merge_sort(int *A, int *B, int left, int mid, int right){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= right - left + 1) return;

    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right){
        if (A[i] <= A[j]) B[k++] = A[i++];
        else B[k++] = A[j++];
    }
    // handle remaining elements
    while (i <= mid) B[k++] = A[i++];
    while (j <= right) B[k++] = A[j++];
}

void mergeSort(int *A, int n){
    int *d_a, *d_b;
    cudaMalloc(&d_a, n * sizeof(int));
    cudaMalloc(&d_b, n * sizeof(int));
    cudaMemcpy(d_a, A, n * sizeof(int), cudaMemcpyHostToDevice);
    for (int s = 0; s < n; s *= 2){
        for (int left = 0; left < n - s; left += 2 * s){
            int mid = left + s - 1;
            int right = min(left + 2 * s - 1, n - 1);
            merge_sort<<<1, 256>>>(d_a, d_b, left, mid, right);
        }
        std::swap(d_a, d_b);
    }
    cudaMemcpy(A, d_a, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b);
}