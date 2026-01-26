#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define STEPS 100
#define N 512

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("Warmup kernel executed.\n");
}

__global__ void gameOfLife(int *grid, int *new_grid){
    __shared__ int tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    // load center
    tile[tx][ty] = grid[(x % N) * N + (y % N)];

    // load halos
    if (threadIdx.x == 0)
        tile[0][ty] = grid[((x - 1 + N) % N) * N + (y % N)];
    if (threadIdx.x == BLOCK_SIZE - 1)
        tile[BLOCK_SIZE + 1][ty] = grid[((x + 1) % N) * N + (y % N)];
    if (threadIdx.y == 0)
        tile[tx][0] = grid[(x % N) * N + ((y - 1 + N) % N)];
    if (threadIdx.y == BLOCK_SIZE - 1)
        tile[tx][BLOCK_SIZE + 1] = grid[(x % N) * N + ((y + 1) % N)];

    // Load corners
    if (threadIdx.x == 0 && threadIdx.y == 0)
        tile[0][0] = grid[((x - 1 + N) % N) * N + ((y - 1 + N) % N)];
    if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE - 1)
        tile[0][BLOCK_SIZE + 1] = grid[((x - 1 + N) % N) * N + ((y + 1) % N)];
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == 0)
        tile[BLOCK_SIZE + 1][0] = grid[((x + 1) % N) * N + ((y - 1 + N) % N)];
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == BLOCK_SIZE - 1)
        tile[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = grid[((x + 1) % N) * N + ((y + 1) % N)];

    __syncthreads();

    // Count neighbors
    int count = 0;
    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            if (dx != 0 || dy != 0)
                count += tile[tx + dx][ty + dy];

    int idx = x * N + y;
    int cell = tile[tx][ty];
    new_grid[idx] = (cell && (count == 2 || count == 3)) || (!cell && count == 3);
}

void print_grid(int *grid) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%c ", grid[i * N + j] ? 'O' : '.');
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    int *h_grid = (int *)malloc(N * N * sizeof(int));
    int *d_grid, *d_new_grid;

    // Initialize host grid randomly
    for (int i = 0; i < N * N; i++)
        h_grid[i] = rand() % 2;

    cudaMalloc(&d_grid, N * N * sizeof(int));
    cudaMalloc(&d_new_grid, N * N * sizeof(int));
    cudaMemcpy(d_grid, h_grid, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);

    for (int step = 0; step < STEPS; step++) {
        gameOfLife<<<numBlocks, threadsPerBlock>>>(d_grid, d_new_grid);
        cudaDeviceSynchronize();

        // Swap grids
        int *temp = d_grid;
        d_grid = d_new_grid;
        d_new_grid = temp;

        // Copy to host and print
        cudaMemcpy(h_grid, d_grid, N * N * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Step %d:\n", step + 1);
        print_grid(h_grid);
    }

    cudaFree(d_grid);
    cudaFree(d_new_grid);
    free(h_grid);
    return 0;
}