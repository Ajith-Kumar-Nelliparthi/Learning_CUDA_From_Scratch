#include <stdio.h>
#include <cuda_runtime.h>

// warmup kernel
__global__ void warmup(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == -1) {
        printf("This is a warmup kernel.\n");
    }
}

// cubic evaluation kernel
__global__ void cubic1(float a, float b, float c, float d, const float *X, float *Y, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i=idx; i<N; i+=stride){
        float x = X[i];
        Y[i] = a*x*x*x + b*x*x + c*x + d;
    }
}

__global__ void cubic(float a, float b, float c, float d, const float *X, float *Y, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int vecN = N / 4;

    for (int i=idx; i<vecN; i+=stride){
        float4 x = reinterpret_cast<const float4 *>(X)[i];
        float4 y;
        y.x = a*x.x*x.x + b*x.x*x.x + c*x.x + d;
        y.y = a*x.y*x.y + b*x.y*x.y + c*x.y + d;
        y.z = a*x.z*x.z + b*x.z*x.z + c*x.z + d;
        y.w = a*x.w*x.w + b*x.w*x.w + c*x.w + d;
        reinterpret_cast<float4 *>(Y)[i] = y;
    }
    for (int i=vecN + idx; i<N; i+=stride){
        Y[i] = a*X[i]*X[i]*X[i] + b*X[i]*X[i] + c*X[i] + d;
    }
}

int main(){
    const int N = 1<<20;
    size_t size = N * sizeof(float);
    
    float *h_x, *h_y;
    cudaHostAlloc((void **)&h_x, size, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_y, size, cudaHostAllocDefault);

    for (int i=0; i<N; i++){
        h_x[i] = rand() / (float)RAND_MAX;
    }

    float a = 1.0f, b = -2.0f, c = 3.0f, d = -4.0f;

    float *d_x, *d_y;
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    warmup<<<1, 1>>>();

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cubic<<<blocks, threadsPerBlock>>>(a, b, c, d, d_x, d_y, N);
    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i=0; i<10; i++){
        printf("%f ", h_y[i]);
    }
    printf("\n");
    
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFreeHost(h_x);
    cudaFreeHost(h_y);
    return 0;

}