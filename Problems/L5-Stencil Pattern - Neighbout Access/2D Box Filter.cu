#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// kernel-1: horizontal sliding window
__global__ void box_filter_horizontal(const float *in, float *out, int width, int height, int r){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;

    int row_offset = width * row;
    float scale = 1.0f / (2 * r + 1);
    float running_sum = 0;

    // initalize the sum for first window
    for (int i=-r; i<=r; i++){
        int col = max(0, min(width-1, i));
        running_sum += in[row_offset + col];
    }
    out[row_offset] = running_sum * scale;

    // slide the window across the row
    for (int col=1; col<width; col++){
        int next_col = min(width - 1, col + r);
        int prev_col = max(0, col - r - 1);

        running_sum += in[row_offset + next_col];
        running_sum -= in[row_offset + prev_col];

        out[row_offset + col] = running_sum * scale;
    }
}

// kernel-2: vertical sliding
__global__ void box_filter_vertical(const float *in, float *out, int width, int height, int r){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= width) return;

    float scale = 1.0f / (2 * r + 1);
    float running_sum = 0;

    for (int i=-r; i<=r; i++){
        int row = max(0, min(height-1, i));
        running_sum += in[row * width + col];
    }
    out[col] = running_sum * scale;

    for (int row=1; row < height; row++){
        int next_row = min(height-1, row+r);
        int prev_row = max(0, row-r-1);

        running_sum += in[next_row * width + col];
        running_sum -= in[prev_row * width + col];

        out[row * width + col] = running_sum * scale;
    }
}

int main() {
    const int W = 1024, H = 1024;
    const int radius = 10; // Box size is (2*10 + 1) = 21x21
    size_t size = W * H * sizeof(float);

    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);
    for (int i = 0; i < W * H; i++) h_in[i] = (float)(rand() % 255);

    float *d_in, *d_temp, *d_out;
    CHECK(cudaMalloc(&d_in, size));
    CHECK(cudaMalloc(&d_temp, size)); // Intermediate buffer
    CHECK(cudaMalloc(&d_out, size));

    CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // Launch Horizontal Pass: One thread per Row
    int threads_h = 256;
    int blocks_h = (H + threads_h - 1) / threads_h;
    box_filter_horizontal<<<blocks_h, threads_h>>>(d_in, d_temp, W, H, radius);

    // Launch Vertical Pass: One thread per Column
    int threads_v = 256;
    int blocks_v = (W + threads_v - 1) / threads_v;
    box_filter_vertical<<<blocks_v, threads_v>>>(d_temp, d_out, W, H, radius);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    printf("Processing complete. Pixel [512,512] value: %f\n", h_out[512 * W + 512]);

    cudaFree(d_in); cudaFree(d_temp); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}