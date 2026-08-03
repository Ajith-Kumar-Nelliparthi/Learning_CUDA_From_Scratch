#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <float.h>
#include <math.h>
#include <torch/extension.h>
#include <torch/types.h>

#define FLOAT4(valu) (reinterpret_cast<float4 *>(&(value))[0])
#define INT4(valu) (reinterpret_cast<int4 *>(&(value))[0])
#define WARP_SIZE 32

__global__ void sgemm_naive_f32(float *a, float *b, float *c, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N) {
        float sum = 0.0f;
#pragma unroll
        for (int k=0; k<K; k++) {
            sum += a[m * K + k] * b[k * N + n];
        }
        c[m * N + n] = sum;
    }
}

// define slices of BM,BN,BK
template <const int BM=32, const int BN=32, const int BK=32>
__global__ void sgemm_sliced_k_f32_kernel(float *a, float *b, float *c, int M, int N, int K) {
    __shared__ float s_a[BM][BK], s_b[BK][BN];

    // define thread & block indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    // tid within the block upto 1024 (31 * 32 + 31 = 1023)
    int tid = ty * blockDim.y + tx; 
    // shared memory indices (to write values)
    int load_smem_a_m = tid / 32;
    int load_smem_a_k = tid % 32;
    int load_smem_b_k = tid / 32;
    int load_smem_b_n = tid % 32;
    // global memory indices (read values from A and B)
    int load_gmem_a_m = by * BM + load_smem_a_m;     // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n;     // global col of a and c

    float sum = 0.0f;
    for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;                        // calculate k index in a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;           // calculate linear index address of A in 1D      (r * col + c)
        s_a[load_smem_a_m][load_smem_a_k] = load_gmem_a_addr;

        int load_gmem_b_k = bk * BK + load_smem_b_k;                        // calculate k index in b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;           // calculate linear index address of B in 1D
        s_b[load_smem_b_k][load_smem_b_n] = load_gmem_b_addr;
        __syncthreads();
        
#pragma unroll
        for (int k=0; k<BK; ++k) {
            int comp_smem_a_m = load_smem_a_m;
            int comp_smem_b_n = load_smem_b_n;
            sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
        __syncthreads();
    }

    int store_gmem_c_m = load_gmem_a_m;
    int store_gmem_c_n = load_gmem_b_n;
    int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
    c[store_gmem_c_addr] = sum;
}

template <const int BM = 128, const int BN = 128, const int BK = 8,
    const int TM = 8, const int TN = 8>
__global__ void sgemm_t_8x8_sliced_k_f32x4_kernel(float *a, float *b, float *c, int M, int N, int K) {
    __shared__ float s_a[BM][BK], s_b[BK][BM];    // initalize shared memory of 128x8 and 8x128 tiles

    // indexing
    const int tx = threadIdx.x;   // 0 to 127
    const int ty = threadIdx.y;   // 0 to 7
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    int tid = ty * blockDim.x + tx; // thread within the block

    int load_smem_a_m = tid / 2; // (128 / 8) * (128 / 8) = 256 threads per block
                                // tid / 2 = 256 / 2 = 128 -> (0-127)
    int load_smem_a_k = (tid % 2 == 0) ? 0:4; // a_k loads or one thread reads 0-4 & the other reads 4-7 indices at once 
    int load_smem_b_k = tid / 32;                   // 0-7 threads in y-dir
    int load_smem_b_n =  (tid % 32) * 4;           // each thread loads 4 elements in x-dir
    // 1D flat index calculation m and n
    int load_gmem_a_m = by * BM + load_smem_a_m;    // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n;    // global col of b and c

    float r_c[TM][TN] - {0.0};
// #pragma unroll
    for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;   // global memory address
        FLOAT4(s_a[load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_gmem_a_addr]);

        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;   // global memory address
        FLOAT4(s_b[load_smem_b_k][load_smem_b_n]) = FLOAT4(b[load_gmem_b_addr]);
        __syncthreads();

#pragma unroll
        for (int k=0; k<BK; ++k) {
#pragma unroll
            for (int m=0; m<TM; ++m) {
#pragma unroll
                for (int n=0; n<TN; ++n) {
                    int comp_smem_a_m = ty * TM + m;
                    int comp_smem_b_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int m=0; m<TM; ++m) {
        int store_gmem_c_m = by * BM + ty * TM + m;
#pragma unroll
        for (int n=0; n<TN; ++n) {
            int store_gmem_c_n = bx * BN + tx * TN + n;
            int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;    // global memory address
            FLOAT4(c[store_gmem_c_addr]) = FLOAT4(r_c[m][n]);
        }
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 8,
    const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void sgemm_t_8x8_sliced_k_f32x4_bcf_kernel(float *a, float *b, float *c, int M, int N, int K) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    int tid = ty * bx + tx;     // thread within the block

    // shared memory declaration
    __shared__ float s_a[BK][BM + OFFSET], s_b[BK][BN + OFFSET];

    // registers declaration
    float r_load_a[TM / 2];
    float r_load_b[TN / 2];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0f};

    // thread mapping to shared mem
    int load_smem_a_m = tid / 2;   // 0,1,,,127
    int load_smem_a_k = (tid & 1) << 2;     // (0,4)
    int load_smem_b_k = tid / 32;   // 0-7
    int load_smem_b_n = (tid & 31) << 2;     // 0,4,,,124

    // 1D flat index for GM
    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;

    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_gmem_a_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_gmem_b_addr]);

        // write to shared mem
        s_a[load_smem_a_k][load_smem_a_m] = r_load_a[0];
        s_a[load_smem_a_k + 1][load_smem_a_m] = r_load_a[1];
        s_a[load_smem_a_k + 2][load_smem_a_m] = r_load_a[2];
        s_a[load_smem_a_k + 3][load_smem_a_m] = r_load_a[3];
        FLOAT4(s_b[load_smem_b_k][load_smem_b_n]) = FLOAT4(r_load_b[0]);
        __syncthreads();

        
#pragma unroll
        for (int tk = 0; tk < BK; ++tk){
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][ty * TM / 2 + BM / 2]);

            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[tk][tx * TN / 2 + BN / 2]);
#pragma unroll
            for (int tm = 0; tm < TM; ++tm){
#pragma unroll
                for (int tn = 0; tn < TN; ++tn){
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
  }
#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
  }
}


template <const int BM = 128, const int BN = 128, const int BK = 8,
    const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel(float *a, float *b, float *c, int M, int N, int K) {
    // indexing
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    int tid = ty * bx + tx;     // thread within the block

    // shared memory declaration
    __shared__ float s_a[2][BK][BM + OFFSET], s_b[2][BK][BN + OFFSET];

    // registers declaration
    float r_load_a[TM / 2];
    float r_load_b[TN / 2];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0f};

    // thread mapping to shared mem
    int load_smem_a_m = tid / 2;  // 0,1,,,127
    int load_smem_a_k = (tid & 1) << 2;    // check if tid is even or odd with last value in bits, then multiply by 4 to get 0 or 4
    int load_smem_b_k = tid / 32; // 0-7
    int load_smem_b_n = (tid & 31) << 2;   // check if tid is even or odd, then multiply by 4 to get 0 or 4 (0,4,,,124)

    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;    // 1D flat index for GM

    // bk = 0 loading here, buffer 0
    {
        int load_gmem_a_k = load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        // load from global memory to registers
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_gmem_a_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_gmem_b_addr]);

        // write to shared memory buffer 0
        s_a[0][load_smem_a_k][load_smem_a_m] = r_load_a[0];
        s_a[0][load_smem_a_k + 1][load_smem_a_m] = r_load_a[1];
        s_a[0][load_smem_a_k + 2][load_smem_a_m] = r_load_a[2];
        s_a[0][load_smem_a_k + 3][load_smem_a_m] = r_load_a[3];
        FLOAT4(s_b[0][load_smem_b_k][load_smem_b_n])  = FLOAT4(r_load_b[0]);
    }
    __syncthreads();

    for (int bk=1; bk < (K + BK - 1) / BK; ++bk) {
        int smem_sel = (bk - 1) & 1; // buffer index for current iteration
        int smem_sel_next = bk & 1; // buffer index for next iteration

        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_gmem_a_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_gmem_b_addr]);

#pragma unroll
        for (int tk=0; tk<BK; ++tk) {
            // load from shared memory buffer
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
            for (int tm=0; tm<TM; ++tm){
            #pragma unroll
                for (int tn=0; tn<TN; ++tn) {
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }
        // write to shared memory buffer for next iteration
        s_a[smem_sel_next][load_smem_a_k][load_smem_a_m] = r_load_a[0];
        s_a[smem_sel_next][load_smem_a_k + 1][load_smem_a_m] = r_load_a[1];
        s_a[smem_sel_next][load_smem_a_k + 2][load_smem_a_m] = r_load_a[2];
        s_a[smem_sel_next][load_smem_a_k + 3][load_smem_a_m] = r_load_a[3];
        FLOAT4(s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]) = FLOAT4(r_load_b[0]);
        __syncthreads();

    #pragma unroll
        for (int tk=0; tk<BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel_next][tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel_next][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel_next][tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel_next][tk][tx * TN / 2 + BN / 2]);

        #pragma unroll
            for (int tm=0; tm<TM; tm++) {
            #pragma unroll
                for (int tn=0; tn<TN; tn++) {
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }

    }
#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
  }
#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
  }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

// SGEMM NAIVE
void sgemm_naive_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32);
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32);
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32);
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 32;
    constexpr int BN = 32;

    dim3 block(BM, BN);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_slice_k_f32_kernel<<<grid, block>>>(
        reinterpret_cast<float *>(a.data_ptr()),
        reinterpret_cast<float *>(b.data_ptr()),
        reinterpret_cast<float *>(c.data_ptr()),
        M, N, K
    );
}

void sgemm_sliced_k_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;

  dim3 block(BN, BM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_sliced_k_f32_kernel<BM, BN, BK>
      <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                        reinterpret_cast<float *>(b.data_ptr()),
                        reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void sgemm_t_8x8_sliced_k_f32x4(torch::Tensor a, torch::Tensor b,
                                torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x8_sliced_k_f32x4_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                        reinterpret_cast<float *>(b.data_ptr()),
                        reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void sgemm_t_8x8_sliced_k_f32x4_bcf(torch::Tensor a, torch::Tensor b,
                                    torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x8_sliced_k_f32x4_bcf_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                        reinterpret_cast<float *>(b.data_ptr()),
                        reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void sgemm_t_8x8_sliced_k_f32x4_bcf_offset(torch::Tensor a, torch::Tensor b,
                                           torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;
  constexpr int OFFSET = 4;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x8_sliced_k_f32x4_bcf_kernel<BM, BN, BK, TM, TN, OFFSET>
      <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                        reinterpret_cast<float *>(b.data_ptr()),
                        reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf(torch::Tensor a, torch::Tensor b,
                                         torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                        reinterpret_cast<float *>(b.data_ptr()),
                        reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset(torch::Tensor a,
                                                torch::Tensor b,
                                                torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;
  constexpr int OFFSET = 4;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel<BM, BN, BK, TM, TN, OFFSET>
      <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                        reinterpret_cast<float *>(b.data_ptr()),
                        reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // CUDA Cores
  TORCH_BINDING_COMMON_EXTENSION(sgemm_naive_f32)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_sliced_k_f32)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4_bcf)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4_bcf_offset)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset)
}