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
    if (idx == 0) printf("Warmup complete.\n");
}

#define SHARED_SIZE 1024
#define ITERATIONS 100000

__global__ void KernelNoConflicts(float *out){
    __shared__ float sdata[SHARED_SIZE];
    int tid = threadIdx.x;

    sdata[tid] = (float)tid;
    __syncthreads();

    float val = 0.0f;
    for (int i=0; i<ITERATIONS; i++){
        val += sdata[tid];
    }
    out[tid] = val;
}

__global__ void KernelBankconflicts(float *out, int stride){
    __shared__ float sdata[SHARED_SIZE]; 
    int tid = threadIdx.x;
    int index = tid * stride; 

    sdata[index] = (float)tid;
    __syncthreads();

    float val = 0.0f;
    for (int i=0; i<ITERATIONS; i++){
        val += sdata[index];
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
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // TEST 1: NO CONFLICT
    CHECK(cudaEventRecord(start));
    KernelNoConflicts<<<1, nThreads>>>(d_out);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("No Bank Conflict Time:    %.3f ms\n", elapsed);

    // TEST 2: STRIDED CONFLICTS
    int strides[] = {2, 16, 32};
    for (int s=0; s<3; s++){
        int stride = strides[s];
        CHECK(cudaEventRecord(start));
        KernelBankconflicts<<<1, nThreads>>>(d_out, stride);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        printf("Stride %d Conflict Time:   %.3f ms\n", stride, elapsed);
    }

    CHECK(cudaFree(d_out));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return 0;
}