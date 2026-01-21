#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess){ \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define K_SIZE 3
#define R (K_SIZE / 2)
#define BLOCK_SIZE 16
#define TILE_SIZE (BLOCK_SIZE + 2*R)

// put the kernel in constant memory for faster access
__constant__ float c_K[K_SIZE * K_SIZE];

// 2d convolution kernel using shared memory
__global__ void convolution_2d(const float* __restrict__ in, float *out, int width, int height) {
    // allocate shared memory
    __shared__ float sdata[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // global indices
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    for (int i=ty; i<TILE_SIZE; i+=BLOCK_SIZE){
        for (int j=tx; j<TILE_SIZE; j+=BLOCK_SIZE){
            
            int i_global = (blockIdx.y * BLOCK_SIZE - R) + i;
            int j_global = (blockIdx.x * BLOCK_SIZE - R) + j;

            // clamping for boundary conditions
            int safe_row = max(0, min(height - 1, i_global));
            int safe_col = max(0, min(width - 1, j_global));

            sdata[i][j] = in[safe_row * width + safe_col];
        }
    }
    __syncthreads();

    if (col < width && row < height){
        float sum = 0.0f;
        for (int i=0; i<K_SIZE; i++){
            for (int j=0; j<K_SIZE; j++){
                sum += sdata[ty + i][tx + j] * c_K[i * K_SIZE + j];
            }
        }
        out[row * width + col] = sum;
    }
}

int main(){
    const int W = 1024, H = 1024;
    size_t size = W * H * sizeof(float);

    float *h_i = (float *)malloc(size);
    float *h_o = (float *)malloc(size);

    float h_kernel[K_SIZE * K_SIZE] = {
        0, -1, 0,
       -1,  5,-1,
        0, -1, 0
    };

    for (int i=0; i<W*H; i++) h_i[i] = (float)(rand() % 255);

    float *d_i, *d_o;
    CHECK(cudaMalloc((void **)&d_i, size));
    CHECK(cudaMalloc((void **)&d_o, size));

    CHECK(cudaMemcpy(d_i, h_i, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(c_K, h_kernel, K_SIZE * K_SIZE * sizeof(float)));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((W + BLOCK_SIZE - 1) / BLOCK_SIZE, (H + BLOCK_SIZE - 1) / BLOCK_SIZE);
    convolution_2d<<<blocks, threads>>>(d_i, d_o, W, H);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_o, d_o, size, cudaMemcpyDeviceToHost));

    printf("Done! Pixel [512,512] changed from %f to %f\n", h_i[512*W+512], h_o[512*W+512]);

    CHECK(cudaFree(d_i));
    CHECK(cudaFree(d_o));
    free(h_i);
    free(h_o);
    return 0;
}