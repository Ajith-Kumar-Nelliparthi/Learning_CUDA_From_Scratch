#include <stdio.h>
#include <stdlib.h>
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

__global__ void prefix_sum_hills_steele(const float* __restrict__ A, float *out, int n){
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int pout = 0, pin = 1;

    // load input into shared memory and right shift by 1, so first index would be 0.
    sdata[pout * n + tid] = (tid > 0) ? A[tid - 1] : 0;
    __syncthreads();

    for (int offset=1; offset<n; offset *=2){
        pout = 1 - pout;
        pin = 1 - pout;

        if (tid >= offset){
            sdata[pout * n + tid] += sdata[pin * n + tid - offset];
        }
        else{
            sdata[pout * n + tid] = sdata[pin * n + tid];
        }
        __syncthreads();
    }
    out[tid] = sdata[pout * n + tid];
}

int main(){
    const int n = 512;
    const size_t size = n * sizeof(float);

    // host memory allocation
    float *h_a = (float *)malloc(size);
    float *h_o = (float *)malloc(size);
    float *h_ref = (float *)malloc(size);

    if (h_a == NULL || h_o == NULL || h_ref == NULL){
        fprintf(stderr, "failed to allocate host memory.\n");
        return 1;
    }
    // initlaize data
    for (int i=0; i<n; i++){
        h_a[i] = 1.0f;
        h_ref[i] = (i == 0) ? 0.0f : h_ref[i - 1] + h_a[i - 1];
    }

    // device memory allocation
    float *d_a, *d_o;
    CHECK(cudaMalloc((void **)&d_a, size));
    CHECK(cudaMalloc((void **)&d_o, size));

    // create cuda streams
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // warmup
    warmup<<< 1, 1, 0, stream>>>();

    // copy host to device
    CHECK(cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream));
    // shared memory size = 2 buffers * n elements * sizeof(float)
    size_t shared_mem = 2 * n * sizeof(float);
    prefix_sum_hills_steele<<<1, n, shared_mem, stream>>>(d_a, d_o, n);
    CHECK(cudaMemcpyAsync(h_o, d_o, size, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    // verify results
    int success = 1;
    for (int i = 0; i < n; i++) {
        if (h_o[i] != h_ref[i]) {
            printf("Mismatch at index %d: GPU %f, CPU %f\n", i, h_o[i], h_ref[i]);
            success = 0;
            break;
        }
    }

    if (success) {
        printf("Scan Successful! All %d elements match the CPU reference.\n", n);
    }

    // Cleanup
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_o));
    free(h_a);
    free(h_o);
    free(h_ref);

    return 0;
}