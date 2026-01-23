#include <stdio.h>
#include <cuda_runtime.h>

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("Warmup kernel executed.\n");
}

__global__ void laplace2D(float *u_old, float *u_new, int w, int h){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < w-1 && y > 0 && y < h-1){
        int idx = y * w + x;
        u_new[idx] = 0.25f * (u_old[idx - 1] + u_old[idx + 1] + u_old[idx - w] + u_old[idx + w]);
    }
}

int main(){
    const int width = 1024;
    const int height = 1024;
    const int size = width * height * sizeof(float);
    const int MAX_ITER = 1000;

    float *h_u = (float*)malloc(size);
    for (int i=0; i<height; i++){
        for (int j=0; j<width; j++){
            int idx = i * width + j;
            if (i == 0) h_u[idx] = 1.0f; // top edge hot
            else h_u[idx] = 0.0f; // elsewhere cold
        }
    }

    float *d_u_old, *d_u_new;
    cudaMalloc((void**)&d_u_old, size);
    cudaMalloc((void**)&d_u_new, size);

    cudaMemcpy(d_u_old, h_u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_new, h_u, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    warmup<<<1, 1>>>();
    cudaDeviceSynchronize();

    for (int iter=0; iter<MAX_ITER; iter++){
        laplace2D<<<gridSize, blockSize>>>(d_u_old, d_u_new, width, height);
        cudaDeviceSynchronize();
        // Swap pointers
        float *temp = d_u_old;
        d_u_old = d_u_new;
        d_u_new = temp;

        if (iter % 200 == 0) printf("Completed iteration %d\n", iter);
    }

    cudaMemcpy(h_u, d_u_new, size, cudaMemcpyDeviceToHost);
    printf("Result at [height/2][width/2]: %f\n", h_u[(height/2) * width + (width/2)]);

    // Cleanup
    cudaFree(d_u_old);
    cudaFree(d_u_new);
    free(h_u);

    return 0;
}