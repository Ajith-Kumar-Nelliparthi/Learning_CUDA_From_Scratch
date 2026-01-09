#include <stdio.h>
#include <math.h>
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

__global__ void box_filter(const float* __restrict__ in, float *out, int n){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread load its own element into sdata[i+1];
    sdata[tid + 1] = (idx < n) ? in[idx] : 0;

    // first thread loads left halo if it exists
    if (tid == 0) sdata[0] = (idx > 0) ? in[idx - 1] : 0;

    // last thread loads right halo
    if (tid == blockDim.x - 1) sdata[blockDim.x + 1] = (idx + 1 < n) ? in[idx+1] : 0;
    __syncthreads();

    if (idx < n){
        out[idx] = (sdata[tid] + sdata[tid+1] + sdata[tid+2])/3.0f;
    }
}

int main(){
    // warmup
    warmup<<<1, 1>>>();
    
    const int n = 1 << 20;
    size_t size = n * sizeof(float);
    int threadsperblock = 256;
    int blocks = (n + threadsperblock - 1) / threadsperblock;
    int sharedmem = (threadsperblock + 2) * sizeof(float);

    float *h_i = (float *)malloc(size);
    float *h_o = (float *)malloc(size);
    // intialization
    for (int i=0; i<n; i++){
        h_i[i] = 1.0f + (rand() % 9); // load 1.0 - 9.0 elements
    }

    float *d_i, *d_o;
    CHECK(cudaMalloc((void **)&d_i, size));
    CHECK(cudaMalloc((void **)&d_o, size));

    CHECK(cudaMemcpy(d_i, h_i, size, cudaMemcpyHostToDevice));
    box_filter<<<blocks, threadsperblock, sharedmem>>>(d_i, d_o, n);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_o, d_o, size, cudaMemcpyDeviceToHost));

    for (int i=0; i<257; i++){
        printf("%f ", h_o[i]);
    }

    CHECK(cudaFree(d_i)); CHECK(cudaFree(d_o));
    free(h_i); free(h_o);
    return 0;
}