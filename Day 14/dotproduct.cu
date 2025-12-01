#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dotProduct(const float *A, const float *B, float *C, int N){
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float temp = 0.0f;
    if (idx < N){
        temp = A[idx] * B[idx];
    }
    sdata[threadIdx.x] = temp;
    __syncthreads();

    for (int stride = blockDim.x/2; stride >0; stride >>= 1){
        if (threadIdx.x < stride){
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0){
        atomicAdd(C, sdata[0]);
    }
}

void solve(const float *A, const float *B, float *C, int N){
    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemset(C, 0, sizeof(float));
    dotProduct<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(A, B, C, N);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Cuda error: %s\n", cudaGetErrorString(err));
    }
}

int main(){
    const int N = 1000;
    size_t bytes = N * sizeof(float);
    float *h_a = new float[N];
    float *h_b = new float[N];
    float h_c;

    for (int i=0; i<N; i++){
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, sizeof(float));

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    solve(d_a, d_b, d_c, N);
    cudaEventRecord(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU execution time: %.3f ms\n", gpu_time);
    cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Dot Product = %f\n", h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    return 0;
}