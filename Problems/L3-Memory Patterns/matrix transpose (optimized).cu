#include <stdio.h>
#include <cuda_runtime.h>

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("warmup complete.\n");
}

#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess){ \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define M 1024 // no.of rows
#define N 1024 // no.of cols
#define TILE_SIZE 32

__global__ void matrixtranspose(const float* __restrict__ in, int rows, int cols, float *out){
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x; // column index
    int y = blockIdx.y * TILE_SIZE + threadIdx.y; // row index

    // load tiles into shared memory
    if (x < cols && y <rows){
        tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    }
    __syncthreads();

    // compute transposed co-ordinates
    x = blockIdx.y * TILE_SIZE + threadIdx.x;  // blockIdx.y
    y = blockIdx.x * TILE_SIZE + threadIdx.y;  //blockIdx.x

    // write transposed data to output
    if (x < rows && y < cols){
        out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// initialize matrix with random inputs
void inimatrix (float *mat, int rows, int cols){
    for (int i=0; i<rows * cols ; i++){
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main(){

    warmup<<<1,1>>>();
    cudaDeviceSynchronize();
    
    int rows = M;
    int cols = N;

    size_t size_in = rows * cols * sizeof(float);
    size_t size_out = rows * cols * sizeof(float);

    // allocate host memory
    float *h_in = (float *)malloc(size_in);
    float *h_out = (float *)malloc(size_out);

    // initalize input
    srand(2025);
    inimatrix(h_in, rows, cols);

    // allocate device memory
    float *d_in, *d_out;
    CHECK(cudaMalloc((void **)&d_in, size_in));
    CHECK(cudaMalloc((void **)&d_out, size_out));

    CHECK(cudaMemcpy(d_in, h_in, size_in, cudaMemcpyHostToDevice));
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((cols + TILE_SIZE - 1) / TILE_SIZE,
                    (rows + TILE_SIZE - 1) / TILE_SIZE);
    matrixtranspose<<<gridSize, blockSize>>>(d_in, rows, cols, d_out);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost));
    
    printf("\nOriginal (first 4x4):\n");
    for (int r=0; r<4; r++){
        for (int c=0; c<4; c++){
            printf("%f ", h_in[r*cols + c]);
        }
        printf("\n");
    }

    printf("\nTransposed (first 4x4):\n");
    for (int r=0; r<4; r++){
        for (int c=0; c<4; c++){
            printf("%f ", h_out[r*rows + c]);
        }
        printf("\n");
    }
    // Cleanup
    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
}