#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define M_TILE 64 // Each block computes 64x64 tile
#define N_TILE 64
#define K_TILE 16

#define THREADS_X 16 // threads per row direction (N)
#define THREADS_Y 8  // threads per column direction (M)
#define BLOCK_SIZE (THREADS_X * THREADS_Y) // 128 threads

__global__ void gemm(const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C,
                     int M, int N, int K)
{
    // Shared memory: A tile is M_TILE x K_TILE, B tile is K_TILE x N_TILE
    __shared__ float sA[M_TILE][K_TILE];
    __shared__ float sB[K_TILE][N_TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int row_start = by * M_TILE;
    const int col_start = bx * N_TILE;

    // Each thread covers 2 rows and 4 cols inside the tile
    const int thread_row = ty * 2;     // 0,2,4,...,14  -> covers 16 rows (8 threads_y * 2)
    const int thread_col = tx * 4;     // 0,4,8,...,60   -> covers 64 cols (16 threads_x * 4)

    // accumulators: accum[2 rows][4 cols]
    float accum[2][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            accum[i][j] = 0.0f;

    // iterate over K in K_TILE chunks
    for (int tile_k = 0; tile_k < K; tile_k += K_TILE) {
        // load A tile (M_TILE x K_TILE) into sA
        for (int r = thread_row; r < M_TILE; r += THREADS_Y * 2) {
            for (int c = tx; c < K_TILE; c += THREADS_X) {
                int global_r = row_start + r;
                int global_c = tile_k + c;
                float val = 0.0f;
                if (global_r < M && global_c < K) val = A[global_r * K + global_c];
                sA[r][c] = val;
            }
        }

        // load B tile (K_TILE x N_TILE) into sB
        for (int r = ty; r < K_TILE; r += THREADS_Y) {
            for (int c = thread_col; c < N_TILE; c += THREADS_X * 4) {
                int global_r = tile_k + r;
                for (int sub = 0; sub < 4; ++sub) {
                    int global_c = col_start + c + sub;
                    float val = 0.0f;
                    if (global_r < K && global_c < N) val = B[global_r * N + global_c];
                    if ((c + sub) < N_TILE) sB[r][c + sub] = val;
                }
            }
        }

        __syncthreads();

        // Compute partials: iterate k inside K_TILE
        #pragma unroll
        for (int kk = 0; kk < K_TILE; ++kk) {
            // load two A elements (for 2 rows) for this thread's row positions
            float a0 = sA[thread_row + 0][kk];
            float a1 = sA[thread_row + 1][kk];

            // load 4 B elements for this thread's col positions from sB[kk][*]
            float b0 = sB[kk][thread_col + 0];
            float b1 = sB[kk][thread_col + 1];
            float b2 = sB[kk][thread_col + 2];
            float b3 = sB[kk][thread_col + 3];

            // update 8 accumulators
            accum[0][0] += a0 * b0;
            accum[0][1] += a0 * b1;
            accum[0][2] += a0 * b2;
            accum[0][3] += a0 * b3;

            accum[1][0] += a1 * b0;
            accum[1][1] += a1 * b1;
            accum[1][2] += a1 * b2;
            accum[1][3] += a1 * b3;
        }

        __syncthreads();
    }

    // write results back
    int out_row0 = row_start + thread_row + 0;
    int out_row1 = row_start + thread_row + 1;
    int out_col_base = col_start + thread_col;

    // row 0
    if (out_row0 < M) {
        for (int j = 0; j < 4; ++j) {
            int col = out_col_base + j;
            if (col < N) {
                C[out_row0 * N + col] = accum[0][j];
            }
        }
    }
    // row 1
    if (out_row1 < M) {
        for (int j = 0; j < 4; ++j) {
            int col = out_col_base + j;
            if (col < N) {
                C[out_row1 * N + col] = accum[1][j];
            }
        }
    }
}

// host main
int main() {
    int M = 2048;
    int N = 2048;
    int K = 2048;

    size_t sizeA = (size_t)M * K;
    size_t sizeB = (size_t)K * N;
    size_t sizeC = (size_t)M * N;

    float *h_A = (float*)malloc(sizeA * sizeof(float));
    float *h_B = (float*)malloc(sizeB * sizeof(float));
    float *h_C = (float*)malloc(sizeC * sizeof(float));

    srand(123);
    for (size_t i = 0; i < sizeA; ++i) h_A[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < sizeB; ++i) h_B[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < sizeC; ++i) h_C[i] = 0.0f;

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, sizeA * sizeof(float));
    cudaMalloc(&d_B, sizeB * sizeof(float));
    cudaMalloc(&d_C, sizeC * sizeof(float));
    cudaMemset(d_C, 0, sizeC * sizeof(float));

    cudaMemcpy(d_A, h_A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((N + N_TILE - 1) / N_TILE, (M + M_TILE - 1) / M_TILE);

    // Warmup
    gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaGetLastError();
    cudaDeviceSynchronize();

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    const int iterations = 10;
    for (int i = 0; i < iterations; ++i) {
        gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    if (ms <= 0.0f) {
        printf("Timing measurement failed (ms == 0). Aborting gflops print.\n");
    } else {
        double gflops = 2.0 * (double)M * N * K * iterations / (ms * 1e-3) * 1e-9;
        printf("Performance: %.2f GFLOPS (avg %.3f ms per run)\n", gflops, ms / iterations);
    }

    cudaMemcpy(h_C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sample C[0..4]:\n");
    for (int i = 0; i < 5; ++i) {
        printf("C[0][%d] = %.6f\n", i, h_C[i]);
    }

    // cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
