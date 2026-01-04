#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess){ \
        fprintf(stderr, "cuda error in file '%s' at line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}while(0)

__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) printf("warmup complete.\n");
}

__global__ void scan_with_max(const int* __restrict__ A, int *out, int n){
    extern __shared__ int sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n){
        sdata[tid] = A[idx];
    }
    else{
        sdata[tid] = -INT_MIN;
    }

    for (int offset=1; offset < blockDim.x; offset <<= 1){
        int val = sdata[tid];
        if (tid >= offset){
            val = max(sdata[tid - offset], val);
        }
        __syncthreads();
        sdata[tid] = val;
        __syncthreads();
    }
    if (idx < n) out[idx] = sdata[tid];
}

int main(){
    const int n = 1 << 20;
    size_t size = n * sizeof(int);

    int *h_a = (int *)malloc(size);
    int *h_o = (int *)malloc(size);
    for (int i=0; i<n; i++){
        h_a[i] = rand() / RAND_MAX;
    }

    int *d_a, *d_o;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_o, size);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    size_t shared_mem = threads * sizeof(int);

    // create cuda streams
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // warmup
    warmup<<< 1, 1, 0, stream>>>();

    // copy host to device
    CHECK(cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream));
    scan_with_max<<<blocks, threads, shared_mem, stream>>>(d_a, d_o, n);
    CHECK(cudaMemcpyAsync(h_o, d_o, size, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    for (int i=0; i<257; i++){
        printf("%d", h_o[i]);
    }

    // Cleanup
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_o));
    free(h_a);
    free(h_o);

    return 0;
}