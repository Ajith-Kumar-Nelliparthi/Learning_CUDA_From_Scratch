#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess){ \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("warmup complete.\n");
}

#define SHARED_MEM_SIZE 1024
#define ITERATIONS 100000

// kernel with no conflicts
__global__ void KernelNoConflicts(float *out){
    __shared__ float sdata[SHARED_MEM_SIZE];
    int tid = threadIdx.x;

    sdata[tid] = (float)tid;
    __syncthreads();

    float val = 0.0f;
    for (int i=0; i<ITERATIONS; i++){
        val += sdata[tid];
    }
    out[tid] = val;
}

// kernel -2 : bank conflicts
__global__ void KernelBankconflicts(float *out, int stride){
    __shared__ float sdata[SHARED_MEM_SIZE * 32];
    int tid = threadIdx.x;

    sdata[tid * stride] = (float)tid;
    __syncthreads();

    float val = 0.0f;
    for (int i=0; i<ITERATIONS; i++){
        val += sdata[tid * stride];
    }
    out[tid] = val;
}

int main() {

     warmup<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());
    
    int nThreads = 32;
    float *d_out;
    CHECK(cudaMalloc(&d_out, nThreads * sizeof(float)));

    cudaEvent_t start, stop;
    float elapsed;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // TEST 1: NO CONFLICT
    cudaEventRecord(start);
    KernelNoConflicts<<<1, nThreads>>>(d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("No Bank Conflict Time:   %.3f ms\n", elapsed);

    // TEST 2: (2, 16, 32)-WAY CONFLICT
    int strides[] = {2, 16, 32};
    int numtests = 3;
    for (int s=0; s<numtests; s++){
        int stride = strides[s];
        cudaEventRecord(start);
        KernelBankconflicts<<<1, nThreads>>>(d_out, stride);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        printf("%d-Way Bank Conflict Time: %.3f ms\n", stride, elapsed);

    }
    // Cleanup
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}