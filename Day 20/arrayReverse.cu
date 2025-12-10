#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i=idx; i< N/2; i+=stride){
        float temp = input[i];
        input[i] = input[N - 1 -i];
        input[N -1 -i] = temp;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}