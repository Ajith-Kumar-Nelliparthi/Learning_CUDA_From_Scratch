#include <stdio.h>
#include <cuda_runtime.h>

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("warmup complete\n");
}

#define CHECK(call) \
do{ \
    cudaError_t err = call; \
    if (err != cudaSuccess){ \
        fprintf(stderr, "cuda error in '%s' at line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void gaussian_blur(const int* __restrict__ in, int *out, int n){
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // load data to shared memory from 2nd index
    int s_idx = tid + 2;
    sdata[s_idx] = (idx < n) ? in[idx] : 0;

    // load left halo
    if (tid == 0) {
        // load two pixels to the left side of block's start
        sdata[0] = (idx >=2) ? in[idx - 2] : 0;
        sdata[1] = (idx >= 1) ? in[idx - 1] : 0;
    }
    
    // load right halo
    if (tid == blockDim.x -1){
        // load two pixels to the right of block's end
        sdata[s_idx + 1] = (idx + 1 < n) ? in[idx + 1] : 0;
        sdata[s_idx + 2] = (idx + 2 < n) ? in[idx + 2] : 0;
    }
    __syncthreads();

    if (idx < n){
        out[idx] = (sdata[s_idx-2]*1 + sdata[s_idx-1]*2 + sdata[s_idx]*4 + sdata[s_idx+1]*2 + sdata[s_idx+2]*1) / 10;
    }
}

int main(){
    warmup<<<1, 1>>>();

    const int n = 1 << 24;
    size_t size = n * sizeof(int);
    int threadsperblock = 256;
    int blocks = (n + threadsperblock - 1) / threadsperblock;
    int sharedmem = (threadsperblock + 4) * sizeof(int);

    int *h_i = (int *)malloc(size);
    int *h_o = (int *)malloc(size);
    for (int i=0; i<n; i++){
        h_i[i] = rand() % 10; // load 0 - 9
    }

    int *d_i, *d_o;
    CHECK(cudaMalloc((void **)&d_i, size));
    CHECK(cudaMalloc((void **)&d_o, size));

    CHECK(cudaMemcpy(d_i, h_i, size, cudaMemcpyHostToDevice));
    gaussian_blur<<<blocks, threadsperblock, sharedmem>>>(d_i, d_o, n);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_o, d_o, size, cudaMemcpyDeviceToHost));

    for (int i=0; i<10; i++){
        printf("%d ", h_o[i]);
    }

    CHECK(cudaFree(d_i)); CHECK(cudaFree(d_o));
    free(h_i); free(h_o);

    return 0;
}