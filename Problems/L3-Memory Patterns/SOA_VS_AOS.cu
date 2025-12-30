// Structure of Array vs Array of Structure
#include <stdio.h>
#include <cuda_runtime.h>

struct Particle { float x, y, z; };

__global__ void AOS(Particle* data, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <N){
        data[idx].x += 1.0f;
    }
}

struct Particles {
    float *x;
    float *y;
    float *z;
};

__global__ void SOA(float *x, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        x[idx] += 1.0f;
    }
}

int main(){
    int N = 1 << 20;
    size_t size = N * sizeof(float);
    int threadsPerblock = 256;
    int blocks = (N + threadsPerblock - 1) / threadsPerblock;

    size_t aos_size = N * sizeof(Particle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // AOS
    Particle *h_aos = (Particle *)malloc(aos_size);
    Particle *d_aos;
    cudaMalloc((void **)&d_aos, aos_size);
    for (int i=0; i<N; i++) { h_aos[i].x = 1.0f; h_aos[i].y = 2.0f; h_aos[i].z = 3.0f; }
    cudaMemcpy(d_aos, h_aos, size, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    AOS<<<blocks, threadsPerblock>>>(d_aos, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    printf("AoS Kernel Time: %f ms\n", ms);

    // SOA
    float *h_soa = (float *)malloc(size);
    float *d_soa;
    cudaMalloc((void **)&d_soa, size);
    for (int i=0; i<N; i++) h_soa[i] = 1.0f;
    cudaMemcpy(d_soa, h_soa, size, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    SOA<<<blocks, threadsPerblock>>>(d_soa, N);
    cudaEventRecord(stop);
    cudaEventElapsedTime(&ms, start, stop);

    printf("SoA Kernel Time: %f ms\n", ms);

    // bandwidth
    double bytes = (double)N * 2.0 * sizeof(float); 
    double bandwidth = (bytes / (ms / 1000.0)) / 1e9;
    printf("SoA Effective Bandwidth: %f GB/s\n", bandwidth);

    // Cleanup
    cudaFree(d_aos);
    cudaFree(d_soa);
    free(h_aos);
    free(h_soa);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}