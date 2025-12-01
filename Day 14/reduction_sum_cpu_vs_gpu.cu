#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

__inline__ __device__ float warpReduceSum(float val){
    for (int offset = warpSize/2; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduceSumGPU(const float *i, float *o, int n){
    extern __shared__ float sdata[];
    const float4 *in_vec = reinterpret_cast<const float4*>(i);
    int num_vec = n / 4;
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + tid;
    int stride = blockDim.x * 2 * gridDim.x;
    float sum = 0.0f;

    // Grid stride loop
    while (idx < num_vec){
        float4 v1 = in_vec[idx];
        sum += v1.x + v1.y + v1.z + v1.w;
        if (idx + blockDim.x < num_vec){
            float4 v2 = in_vec[idx + blockDim.x];
            sum += v2.x + v2.y + v2.z + v2.w;
        }
        idx += stride;
    }
    // handle tail elements (not multiple of 4)
    int tail_start = num_vec * 4;
    for (int k= tail_start + tid; k<n; k+=blockDim.x){
        sum += i[k];
    }
    sdata[tid] = sum;
    __syncthreads();

    // do reduction in shared mem
    for (int s= blockDim.x/2; s>=32; s>>=1){
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float val = sdata[tid];
    if (tid < 32){
        val = warpReduceSum(val);
    }

    if (tid == 0){
        o[blockDim.x] = val;
    }
}

float reduceSumCPU(const float *A, int N){
    float sum = 0.0f;
    for (int i=0; i<N; i++){
        sum += A[i];
    }
    return sum;
}

int main(){
    const int N = 1 << 20;
    size_t size_in_bytes = N * sizeof(float);
    const int blockSizes[] = {128, 256, 512};
    const int numtests = 3;

    float *h_a = (float *)malloc(size_in_bytes);
    for (int i=0; i<N; i++){
        h_a[i] = 1.0f;
    }
    float *d_a;
    cudaMalloc(&d_a, size_in_bytes);
    cudaMemcpy(d_a, h_a, size_in_bytes, cudaMemcpyHostToDevice);

    for (int t=0; t<numtests; t++){
        int threadsPerBlock = blockSizes[t];
        int blocks = (N + threadsPerBlock*8 - 1) / (threadsPerBlock * 8);

        float *d_b;
        cudaMalloc(&d_b, blocks * sizeof(float));
        printf("\nTesting block size: %d (grid: %d, shared mem: %zu bytes)\n", threadsPerBlock, blocks);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        reduceSumGPU<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_a, d_b, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess){
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        float gpu_time = 0;
        cudaEventElapsedTime(&gpu_time, start, stop);
        printf("GPU execution time: %.3f ms\n", gpu_time);
        float *h_b = (float *)malloc(blocks * sizeof(float));
        cudaMemcpy(h_b, d_b, blocks * sizeof(float), cudaMemcpyDeviceToHost);
        float gpu_sum = 0.0f;
        for (int i=0; i< blocks; i++){
            gpu_sum += h_b[i];
        }
        printf("Final Sum = %.0f (expected %d)\n", gpu_sum, N);

        auto cpu_start = std::chrono::high_resolution_clock::now();
        float cpu_sum_check = reduceSumCPU(h_a, N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        float cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
        printf("CPU execution time: %.3f ms\n", cpu_time);
		printf("CPU sum: %.2f\n", cpu_sum_check);

		if (fabs(gpu_sum - cpu_sum_check) < 1e-3) {
			printf("Reduction sum completed successfully\n");
			printf("speedup: %.2fX\n", cpu_time / gpu_time);
		}
        free(h_b);
        cudaFree(d_b);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    cudaFree(d_a);
    free(h_a);
    return 0;
}