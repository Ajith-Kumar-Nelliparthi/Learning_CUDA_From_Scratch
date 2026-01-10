#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

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

__global__ void wave_kernel(float *prev, float *curr, float *next, float c2, int n){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // load current state into shared memory with halos
    int s_idx = tid + 1;
    sdata[s_idx] = (idx < n) ? curr[idx] : 0.0f;

    if (tid == 0){
        sdata[0] = (idx > 0) ? curr[idx - 1] : 0.0f;
    }

    if (tid == blockDim.x - 1){
        sdata[s_idx + 1] = (idx + 1 < n) ? curr[idx + 1] : 0.0f;
    }
    __syncthreads();

    if (idx < n){
        // physics Calculation
        // next = 2*curr - prev + c2*(left - 2*curr + right)
        float u_curr = sdata[s_idx];
        float u_prev = prev[idx];
        float laplacian = sdata[s_idx - 1] - 2.0f * u_curr + sdata[s_idx + 1];

        next[idx] = 2.0f * u_curr - u_prev + (c2 * laplacian);
    }
}

int main(){
    // warmup
    warmup<<<1, 1>>>();
    
    const int n = 1<<24;
    size_t size = n * sizeof(float);
    int threadsperblock = 256;
    int blocks = (n + threadsperblock - 1) / threadsperblock;
    int sharedmem = (threadsperblock + 2) * sizeof(float);
    
    const int steps = 1000;
    const float c2 = 0.25f;

    float *h_curr = (float *)malloc(size);
    for (int i=0; i<n; i++){
        float x = (float)i - (n / 2);
        h_curr[i] = expf(-x * x / 1000.0f);
    }

    float *d_prev, *d_curr, *d_next;
    CHECK(cudaMalloc((void **)&d_curr, size));
    CHECK(cudaMalloc((void **)&d_next, size));
    CHECK(cudaMalloc((void **)&d_prev, size));

    CHECK(cudaMemcpy(d_curr, h_curr, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_prev, h_curr, size, cudaMemcpyHostToHost));

    printf("Starting wave simulation in %d steps..\n", steps);

    for (int t=0; t<steps; t++){
        wave_kernel<<<blocks, threadsperblock, sharedmem>>>(d_prev, d_curr, d_next, c2, n);

        // pointer rotation : triple buffering
        float* temp = d_prev;
        d_prev = d_curr;
        d_curr = d_next;
        d_next = temp;
    }

    CHECK(cudaMemcpy(h_curr, d_curr, size, cudaMemcpyDeviceToHost));

    for (int i = (n/2)-5; i < (n/2)+5; i++) printf("%f ", h_curr[i]);
    printf("\n");

    // Cleanup
    CHECK(cudaFree(d_prev)); CHECK(cudaFree(d_curr)); CHECK(cudaFree(d_next));
    free(h_curr);

    return 0;
}