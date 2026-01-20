#include <stdio.h>
#include <cuda_runtime.h>

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

__global__ void heatEquationKernel(const float *in, float *out, int width, int height, int alpha){
    __shared__ float sdata[BDIMY + 2][BDIMX + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    // global index mapping
    int idx = row * width + col;

    // load center
    if (col < width && row < height){
        sdata[ty + 1][tx + 1] = in[idx];
    } else {
        sdata[ty + 1][tx + 1] = 0.0f;
    }

    // load halo regions
    if (tx == 0) sdata[ty + 1][0] = (col > 0) ? in[idx - 1] : 0.0f;
    if (tx == BDIMX - 1) sdata[ty + 1][BDIMX + 1] = (col < width - 1) ? in[idx + 1] : 0.0f;
    if (ty == 0) sdata[0][tx + 1] = (row > 0) ? in[idx - width] : 0.0f;
    if (ty == BDIMY - 1) sdata[BDIMY + 1][tx + 1] = (row < height - 1) ? in[idx + width] : 0.0f;
    __syncthreads();

    // stencil computation
    if (col < width && row < height){
        float center = sdata[ty + 1][tx + 1];
        float west = sdata[ty + 1][tx];
        float east = sdata[ty + 1][tx + 2];
        float north = sdata[ty][tx + 1];
        float south = sdata[ty + 2][tx + 1];

        out[idx] = center + alpha * (north + south + east + west - 4.0f * center);
    }
}

int main(){
    const int W = 1024, H = 1024;
    float alpha = 0.2f;
    size_t size = W * H * sizeof(float);

    float *h_i = (float *)malloc(size);
    float *h_o = (float *)malloc(size);
    for (int i=0; i<W*H; i++) h_i[i] = (float)(rand() % 10);

    float *d_i, *d_o;
    CHECK(cudaMalloc((void **)&d_i, size));
    CHECK(cudaMalloc((void **)&d_o, size));
    CHECK(cudaMemcpy(d_i, h_i, size, cudaMemcpyHostToDevice));

    dim3 threads(BDIMX, BDIMY);
    dim3 blocks((W + BDIMX - 1) / BDIMX, (H + BDIMY - 1) / BDIMY);

    for (int i=0; i<100; i++){
        heatEquationKernel<<<blocks, threads>>>(d_i, d_o, W, H, alpha);
        float *temp = d_i; d_i = d_o; d_o = temp;
    }
    CHECK(cudaMemcpy(h_o, d_i, size, cudaMemcpyDeviceToHost));
    printf("2D Heat Simulation Complete. Sample Pixel [512,512]: %f\n", h_o[512*W+512]);

    cudaFree(d_i); cudaFree(d_o);
    free(h_i); free(h_o);
    return 0;
}