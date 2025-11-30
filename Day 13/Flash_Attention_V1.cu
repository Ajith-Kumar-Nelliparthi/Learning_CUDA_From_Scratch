#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel - 1: Compute QK^T
__global__ void matmul_qk(const float *Q, const float *K, float *S, int B, int N, int dk){
    int b = blockIdx.z; // Batch index
    int query = threadIdx.y; // Query index
    int key = threadIdx.x; // Key index

    float sum = 0.0f;
    for (int t=0; t<dk; t++){
        float q = Q[b*N*dk + query*dk + t];
        float k = K[b*N*dk + key*dk + t];
        sum += q * k;
    }
    S[b*N*N + query*N + key] = sum / sqrtf((float)dk);
}

// Kernel - 2: Apply softmax to S
__global__ void softmax(float *S, int B, int N){
    int b = blockIdx.y;
    int i = blockIdx.x;

    float max_val = -INFINITY;
    for (int j=0; j<N; j++){
        max_val = fmaxf(max_val, S[b*N*N + i*N + j]);
    }

    float sum = 0.0f;
    for (int j=0; j<N; j++){
        float e = expf(S[b*N*N + i*N + j] - max_val);
        S[b*N*N + i*N + j] = e;
        sum += e;
    }
    for (int j=0; j<N; j++){
        S[b*N*N + i*N + j] /= sum;
    }
}

// Kernel - 3: Compute S*V
__global__ void matmul_sv(const float *S, const float *V, float *O, int B, int N, int dv){
    int b = blockIdx.z ; // Batch index
    int i = blockIdx.y;
    int t = blockIdx.x;

    float sum = 0.0f;
    for (int j=0; j<N; j++){
        float s = S[b*N*N + i*N + j];
        float v = V[b*N*dv + j*dv + t];
        sum += s * v;
    }
    O[b*N*dv + i*dv + t] = sum;
}

int main()
{
    const int B = 1;   // batch size
    const int N = 4;   // sequence length
    const int dk = 4;  // key/query dims
    const int dv = 4;  // value dims

    int sizeQ = B * N * dk * sizeof(float);
    int sizeK = B * N * dk * sizeof(float);
    int sizeV = B * N * dv * sizeof(float);
    int sizeS = B * N * N * sizeof(float);
    int sizeO = B * N * dv * sizeof(float);

    float h_Q[B*N*dk], h_K[B*N*dk], h_V[B*N*dv], h_O[B*N*dv];

    // Initialize toy values
    for (int i = 0; i < B*N*dk; i++) {
        h_Q[i] = 1.0f;
        h_K[i] = 1.0f;
    }
    for (int i = 0; i < B*N*dv; i++)
        h_V[i] = 1.0f;

    // Device allocations
    float *d_Q, *d_K, *d_V, *d_S, *d_O;
    cudaMalloc(&d_Q, sizeQ);
    cudaMalloc(&d_K, sizeK);
    cudaMalloc(&d_V, sizeV);
    cudaMalloc(&d_S, sizeS);
    cudaMalloc(&d_O, sizeO);

    // Upload
    cudaMemcpy(d_Q, h_Q, sizeQ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, sizeK, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, sizeV, cudaMemcpyHostToDevice);

    // Launch QK^T kernel
    dim3 grid_qk(N, N, B);
    matmul_qk<<<grid_qk, 1>>>(d_Q, d_K, d_S, B, N, dk);

    // Softmax kernel
    dim3 grid_sm(N, B);
    softmax<<<grid_sm, 1>>>(d_S, B, N);

    // Multiply softmax × V
    dim3 grid_sv(dv, N, B);
    matmul_sv<<<grid_sv, 1>>>(d_S, d_V, d_O, B, N, dv);

    // Download result
    cudaMemcpy(h_O, d_O, sizeO, cudaMemcpyDeviceToHost);

    printf("\nOutput Attention Matrix O:\n");
    for (int i = 0; i < N; i++) {
        for (int t = 0; t < dv; t++)
            printf("%.4f ", h_O[i*dv + t]);
        printf("\n");
    }

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_S); cudaFree(d_O);

    return 0;
}