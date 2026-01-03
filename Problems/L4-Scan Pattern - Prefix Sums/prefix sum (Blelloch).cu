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

__global__ void prefix_sum_blelloch(const float* __restrict__ A, float *out, int n){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int offset = 1;

    // load input into shared memory
    sdata[2 * tid] = A[2 * tid];
    sdata[2 * tid + 1] = A[2 * tid + 1];

    // phase 1: up sweep (sum reduction)
    // we go from d = n/2 to 1
    for (int d= n>>1; d>0; d>>=1){
        __syncthreads();
        if (tid < d){
            int ai = offset * ( 2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;

            sdata[bi] += sdata[ai];
        }
        offset *= 2;
    }

    // set last element to 0
    if (tid == 0){
        sdata[n - 1] = 0;
    }

    // phase 2: down sweep
    for (int d=1; d<n; d*=2){
        offset >>= 1;
        __syncthreads();
        if (tid < d){
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;

            float t = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += t;
        }
    }
    __syncthreads();

    out[2 * tid] = sdata[2 * tid];
    out[2 * tid + 1] = sdata[2 * tid + 1];
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
    // shared memory size = n elements * sizeof(float)
    size_t shared_mem = n * sizeof(float);
    prefix_sum_blelloch<<<1, n/2, shared_mem, stream>>>(d_a, d_o, n);
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