#include <stdio.h>
#include <cuda_runtime.h>

template <typename T, typename Op>
__global__ void reducekernel(const T* __restrict__ input, T* output, int N, Op op, T identity) {
    extern __shared__ T sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + tid;

    T val = identity;
    if (idx < N) val = input[idx];
    if (idx + blockDim.x < N) val = op(val, input[idx + blockDim.x]);

    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = op(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// host wrapper
template <typename T, typename Op>
T reduce(const T* d_input, int n, Op op, T identity) {
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

    T* d_intermediate;
    cudaMalloc(&d_intermediate, blocks * sizeof(T));

    reducekernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(d_input, d_intermediate, N, op, identity);

    T result;
    if (blocks > 1) {
        result = reduce(d_intermediate, blocks, op, identity);
    } else {
        cudaMemcpy(&result, d_intermediate, sizeof(T), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_intermediate);
    return result;
}

// operator functions
struct Sum {
    __device__ __host__ float operator()(float a, float b) const {return a + b; }
};

struct Max {
    __device__ __host__ int operator()(int a, int b) const { return a > b ? a : b; }
};

int main() {
    const int N = 1 << 20;
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;

    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    float sumResult = reduce(d_data, N, Sum(), 0.0f);
    printf("Sum result: %f\n", sumResult);

    cudaFree(d_data);
    delete[] h_data;
    return 0;
}