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

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("Warmup kernel executed.\n");
}

#define K_SIZE 3
#define R (K_SIZE / 2)
#define BLOCK_SIZE 16
#define TILE_SIZE (BLOCK_SIZE + 2 * R)

// Define Sobel filters in constant memory
// Standard Sobel X: Vertical edges
__constant__ float Gx[K_SIZE * K_SIZE] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

// Standard Sobel Y: Horizontal edges
__constant__ float Gy[K_SIZE * K_SIZE] = {
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
};

__global__ void sobelEdgeDetection(const float* __restrict__ in, float *out, int width, int height) {
    // Shared memory tile (18x18 for 16x16 threads)
    __shared__ float sdata[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global indices
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    // Collaborative Tile Loading (Handles Halos)
    // We use a loop because 256 threads must load 324 pixels (18x18)
    for (int i = ty; i < TILE_SIZE; i += BLOCK_SIZE) {
        for (int j = tx; j < TILE_SIZE; j += BLOCK_SIZE) {
            
            // Map shared memory local index to global image index
            int i_global = (blockIdx.y * BLOCK_SIZE - R) + i;
            int j_global = (blockIdx.x * BLOCK_SIZE - R) + j;

            // Boundary clamping (Clamps to edge pixel if out of bounds)
            int safe_row = max(0, min(height - 1, i_global));
            int safe_col = max(0, min(width - 1, j_global));

            sdata[i][j] = in[safe_row * width + safe_col];
        }
    }
    __syncthreads();

    // Sobel Magnitude Calculation
    if (col < width && row < height) {
        float sumX = 0.0f;
        float sumY = 0.0f;

        // The center of our neighborhood in sdata is at (ty + R, tx + R)
        for (int i = -R; i <= R; i++) {
            for (int j = -R; j <= R; j++) {
                float pixel = sdata[ty + R + i][tx + R + j];
                
                // Indexing into the 1D constant arrays
                int k_idx = (i + R) * K_SIZE + (j + R);
                sumX += pixel * Gx[k_idx];
                sumY += pixel * Gy[k_idx];
            }
        }

        // Calculate Gradient Magnitude
        int out_idx = row * width + col;
        out[out_idx] = sqrtf(sumX * sumX + sumY * sumY);
    }
}

int main() {
    warmup<<<1, 32>>>();
    CHECK(cudaDeviceSynchronize());
    
    const int W = 1024;
    const int H = 1024;
    size_t size = W * H * sizeof(float);

    float *h_i = (float *)malloc(size);
    float *h_o = (float *)malloc(size);

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            h_i[i * W + j] = (j < W / 2) ? 0.0f : 255.0f;
        }
    }

    float *d_i, *d_o;
    CHECK(cudaMalloc((void **)&d_i, size));
    CHECK(cudaMalloc((void **)&d_o, size));

    CHECK(cudaMemcpy(d_i, h_i, size, cudaMemcpyHostToDevice));

    float h_Gx[K_SIZE * K_SIZE] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    float h_Gy[K_SIZE * K_SIZE] = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };

    CHECK(cudaMemcpyToSymbol(Gx, h_Gx, K_SIZE * K_SIZE * sizeof(float)));
    CHECK(cudaMemcpyToSymbol(Gy, h_Gy, K_SIZE * K_SIZE * sizeof(float)));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((W + BLOCK_SIZE - 1) / BLOCK_SIZE, (H + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Launching Sobel Edge Detection kernel (%d x %d)...\n", W, H);
    sobelEdgeDetection<<<blocks, threads>>>(d_i, d_o, W, H);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_o, d_o, size, cudaMemcpyDeviceToHost));

    printf("Done!\n");
    printf("Original Pixel [512, 512]: %f\n", h_i[512 * W + 512]);
    printf("Sobel Edge Value [512, 512]: %f\n", h_o[512 * W + 512]);

    // 13. Cleanup
    CHECK(cudaFree(d_i));
    CHECK(cudaFree(d_o));
    free(h_i);
    free(h_o);

    return 0;
}