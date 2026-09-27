#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void softmax(const float *s, float *p, int seq_l) {
    int tid = threadIdx.x;
    int row = blockIdx.x * blockDim.x + tid;
    
    float exp_sum = 0.0f;
#pragma unroll
    for (int j = 0; j < seq_l; j++) {
        exp_sum += expf(s[row * seq_l + j]);
    }
#pragma unroll
    for (int j = 0; j < seq_l; j++) {
        p[row * seq_l + j] = expf(s[row * seq_l + j]) / exp_sum;
    }
}

__global__ void self_attn_naive(float *q, float *k, float *v, float *s, float *p, int seq_l, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;    // query index
    int col = blockIdx.x * blockDim.x + threadIdx.x;    // key index
    float scale = (1 / sqrtf(d));

    if (row >= seq_l || col >= seq_l) return;
    float sum = 0.0f;

#pragma unroll
    for (int t = 0; t < d; t++) {
        sum += q[row * d + t] * k[col * d + t];
    }
    s[row * seq_l + col] = sum * scale;
}

__global__ void output(float *p, float *v, float *O, int seq_l, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= seq_l || col >= seq_l) return;
    float sum = 0.0f;

#pragma unroll
    for (int t = 0; t < seq_l; t++) {
        sum += p[row * seq_l + t] * v[t * d + col];
    }
    O[row * d + col] = sum;
}

int main() {
    int seq_l = 4;   // sequence length
    int d     = 3;   // head dimension

    float *h_q = (float*)malloc(seq_l * d * sizeof(float));
    float *h_k = (float*)malloc(seq_l * d * sizeof(float));
    float *h_v = (float*)malloc(seq_l * d * sizeof(float));
    float *h_s = (float*)malloc(seq_l * seq_l * sizeof(float));
    float *h_p = (float*)malloc(seq_l * seq_l * sizeof(float));
    float *h_O = (float*)malloc(seq_l * d * sizeof(float));

    for (int i = 0; i < seq_l * d; i++) {
        h_q[i] = (float)(rand() % 10) / 10.0f;
        h_k[i] = (float)(rand() % 10) / 10.0f;
        h_v[i] = (float)(rand() % 10) / 10.0f;
    }

    float *d_q, *d_k, *d_v, *d_s, *d_p, *d_O;
    cudaMalloc(&d_q, seq_l * d * sizeof(float));
    cudaMalloc(&d_k, seq_l * d * sizeof(float));
    cudaMalloc(&d_v, seq_l * d * sizeof(float));
    cudaMalloc(&d_s, seq_l * seq_l * sizeof(float));
    cudaMalloc(&d_p, seq_l * seq_l * sizeof(float));
    cudaMalloc(&d_O, seq_l * d * sizeof(float));

    cudaMemcpy(d_q, h_q, seq_l * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, seq_l * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, seq_l * d * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block1(16,16);
    dim3 grid1((seq_l+15)/16, (seq_l+15)/16);
    self_attn_naive<<<grid1, block1>>>(d_q, d_k, d_v, d_s, d_p, seq_l, d);

    dim3 block2(128);
    dim3 grid2((seq_l+127)/128);
    softmax<<<grid2, block2>>>(d_s, d_p, seq_l);

    dim3 block3(16,16);
    dim3 grid3((d+15)/16, (seq_l+15)/16);
    output<<<grid3, block3>>>(d_p, d_v, d_O, seq_l, d);

    cudaMemcpy(h_O, d_O, seq_l * d * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Output O (seq_l x d):\n");
    for (int i = 0; i < seq_l; i++) {
        for (int j = 0; j < d; j++) {
            printf("% .4f ", h_O[i*d + j]);
        }
        printf("\n");
    }

    free(h_q); free(h_k); free(h_v); free(h_s); free(h_p); free(h_O);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_s); cudaFree(d_p); cudaFree(d_O);

    return 0;
}