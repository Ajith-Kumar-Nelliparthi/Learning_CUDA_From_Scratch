#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

// Kernel for matrix addition
__global__ void matrixAdd(const float *a, const float *b, float *c, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;   // calculate strides - no of threads in x and y direction
    for (; row < rows; row += stride_y){
        for(int j=col; j<cols; j+=stride_x){
            int idx = row * cols + j;
            c[idx] = a[idx] + b[idx];
        }
    }
}

void matrixAddCPU(const float *a, const float *b, float *c, int rows, int cols){
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            int idx = i * cols +j;
            c[idx] = a[idx] + b[idx];
        }
    }
}

int main(){

    // Matrix dimensions
    const int ROWS = 100;
    const int COLS = 100;
    size_t size = ROWS * COLS * sizeof(float);
    const dim3 blockSizes[] = {dim3(16 , 16), dim3(32, 32)};
    int numtests = 2;

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);
    float *h_c_cpu = (float *)malloc(size);

    // Initialize matrices
    for (int i=0; i< ROWS * COLS; i++){
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy matrices from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // GPU matrix addition
    for (int t=0; t<numtests; t++){
        dim3 blockSize = blockSizes[t];
        dim3 gridSize ( (COLS + blockSize.x - 1) / blockSize.x,
                           (ROWS + blockSize.y - 1) / blockSize.y );

        printf("Testing block size: %dx%d (grid: %d x %d)\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        matrixAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, ROWS, COLS);

        cudaError_t err = cudaGetLastError();
        if ( err != cudaSuccess ) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_time = 0;
        cudaEventElapsedTime(&gpu_time, start, stop);
        printf("GPU execution time: %.3f ms\n", gpu_time);

        // Copy result back to host
        cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

        auto cpu_start = std::chrono::high_resolution_clock::now();
        matrixAddCPU(h_a, h_b, h_c_cpu, ROWS, COLS);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        float cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
		printf("CPU execution time: %.3f ms\n", cpu_time);


        // verify results
        bool correct = true;
        for (int i=0; i < ROWS * COLS; i++){
            if (fabs(h_c_cpu[i] - h_c[i]) > 1e-5){
                correct = false;
                printf("Mismatch at index %d: CPU %f, GPU %f\n", i, h_c_cpu[i], h_c[i]);
                break;
            }
        }
        if (correct) {
            printf("matrix addition completed successfully!\n");
			printf("speedup: %.2fX\n", cpu_time / gpu_time);
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
    free(h_c_cpu);

    return 0;
}