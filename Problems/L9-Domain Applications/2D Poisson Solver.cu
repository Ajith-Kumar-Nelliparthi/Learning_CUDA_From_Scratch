#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define BDIMX 16
#define BDIMY 16

// This kernel solves the Poisson Equation using Jacobi Iteration
__global__ void poissonKernel(const float *in, float *out, const float *f, int width, int height, float h2) {
    __shared__ float sdata[BDIMY + 2][BDIMX + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;
    int idx = row * width + col;

    // 1. Load Center into Shared Memory
    if (col < width && row < height) {
        sdata[ty + 1][tx + 1] = in[idx];
    } else {
        sdata[ty + 1][tx + 1] = 0.0f;
    }

    // 2. Load Halos
    if (tx == 0) sdata[ty + 1][0] = (col > 0) ? in[idx - 1] : 0.0f;
    if (tx == BDIMX - 1) sdata[ty + 1][BDIMX + 1] = (col < width - 1) ? in[idx + 1] : 0.0f;
    if (ty == 0) sdata[0][tx + 1] = (row > 0) ? in[idx - width] : 0.0f;
    if (ty == BDIMY - 1) sdata[BDIMY + 1][tx + 1] = (row < height - 1) ? in[idx + width] : 0.0f;

    __syncthreads();

    // 3. Compute Stencil (Jacobi Method)
    if (col > 0 && col < width - 1 && row > 0 && row < height - 1) {
        float west  = sdata[ty + 1][tx];
        float east  = sdata[ty + 1][tx + 2];
        float north = sdata[ty][tx + 1];
        float south = sdata[ty + 2][tx + 1];
        
        out[idx] = 0.25f * (west + east + north + south - (h2 * f[idx]));
    } else if (col < width && row < height) {
        out[idx] = in[idx];
    }
}

int main() {
    const int W = 1024, H = 1024;
    float h2 = 0.01f; // Step size squared
    size_t size = W * H * sizeof(float);

    float *h_i = (float *)malloc(size);
    float *h_f = (float *)malloc(size); 
    float *h_o = (float *)malloc(size);

    // Initialize: Set everything to 0, and a "Heat Source" in the middle
    for (int i = 0; i < W * H; i++) {
        h_i[i] = 0.0f;
        h_f[i] = 0.0f;
    }
    // Create a "hot square" in the center as a source term f(x,y)
    for (int r = 400; r < 600; r++) {
        for (int c = 400; c < 600; c++) {
            h_f[r * W + c] = -5.0f; 
        }
    }

    float *d_i, *d_o, *d_f;
    CHECK(cudaMalloc((void **)&d_i, size));
    CHECK(cudaMalloc((void **)&d_o, size));
    CHECK(cudaMalloc((void **)&d_f, size));

    CHECK(cudaMemcpy(d_i, h_i, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_f, h_f, size, cudaMemcpyHostToDevice));

    dim3 threads(BDIMX, BDIMY);
    dim3 blocks((W + BDIMX - 1) / BDIMX, (H + BDIMY - 1) / BDIMY);

    // Run 1000 iterations to let the "heat" diffuse
    for (int i = 0; i < 1000; i++) {
        poissonKernel<<<blocks, threads>>>(d_i, d_o, d_f, W, H, h2);
        float *temp = d_i; 
        d_i = d_o; 
        d_o = temp;
    }

    CHECK(cudaMemcpy(h_o, d_i, size, cudaMemcpyDeviceToHost));

    FILE* fp = fopen("heat_data.bin", "wb");
    if (fp) {
        fwrite(h_o, sizeof(float), W * H, fp);
        fclose(fp);
        printf("Data saved to heat_data.bin\n");
    }

    printf("Simulation Complete. Center Value: %f\n", h_o[(H/2)*W+(W/2)]);

    cudaFree(d_i); cudaFree(d_o); cudaFree(d_f);
    free(h_i); free(h_o); free(h_f);
    return 0;
}