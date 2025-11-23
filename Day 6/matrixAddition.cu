#include <stdio.h>
#include <cuda_runtime.h>

// Kernel for matrix addition
__global__ void matrixAdd(const float *a, float *b, float *c, int rows, int cols){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;
    
    for(; row < rows; row += stride_y){
        for (int j = col; j < cols; j+= stride_x){
            int idx = row * cols + j;
            c[idx] = a[idx] + b[idx];
        }
    }
}

int main(){
    // Define matrix dimensions
    const int ROWS = 100;
    const int COLS = 100;
    size_t size = ROWS * COLS * sizeof(float);
    const dim3 blockSizes[] = {dim3(16, 16), dim3(32, 32)};
    int numtests = 2;

    // allocate memory for matrices
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    // Initialize matrices
    for (int i = 0; i < ROWS * COLS; i++){
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy matrices from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    for (int t=0; t< numtests; t++){
        dim3 blockSize = blockSizes[t];
        dim3 gridSize((COLS + blockSize.x - 1) / blockSize.x, (ROWS + blockSize.y - 1) / blockSize.y);
        printf("Testing block size: %dx%d (grid: %d x %d)\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);


        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        matrixAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, ROWS, COLS);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess){
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return -1;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_time = 0;
        cudaEventElapsedTime(&gpu_time, start, stop);
        printf("GPU execution time: %.3f ms\n", gpu_time);

        // Copy result back to host
        cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

        // Verify results
        printf("First 5 result (row 0):/n");
        for (int j=0; j<5; j++){
            int idx = 0 * COLS + j;
			printf("c[0][%d] = %.2f (a[0][%d] = %.2f + b[0][%d] = %.2f)\n", j, h_c[idx], j, h_a[idx], j, h_b[idx]);
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}