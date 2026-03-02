#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cusparse.h>

// Error checking macro
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::cerr << "cuSPARSE Error: " << status << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

int main() {
    const int numRows = 4;
    const int numCols = 4;
    const int nnz     = 8;

    std::vector<int> h_csrRowPtr = {0, 1, 3, 5, 8};
    std::vector<int> h_csrColInd = {0, 0, 1, 1, 2, 0, 2, 3};
    std::vector<float> h_csrVal  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 1.0f, 6.0f};
    
    // Right-hand side b and solution x
    std::vector<float> h_b = {1.0f, 5.0f, 14.0f, 11.0f}; 
    std::vector<float> h_x(numRows, 0.0f);

    // --- 2. Allocate Device Memory ---
    int *d_csrRowPtr, *d_csrColInd;
    float *d_csrVal, *d_b, *d_x;

    CHECK_CUDA(cudaMalloc((void**)&d_csrRowPtr, (numRows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_csrColInd, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_csrVal,    nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_b,         numRows * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_x,         numRows * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_csrRowPtr, h_csrRowPtr.data(), (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrColInd, h_csrColInd.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrVal,    h_csrVal.data(),    nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b,         h_b.data(),         numRows * sizeof(float), cudaMemcpyHostToDevice));

    // --- 3. cuSPARSE Setup ---
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Create Sparse Matrix Descriptor
    cusparseSpMatDescr_t matL;
    CHECK_CUSPARSE(cusparseCreateCsr(&matL, numRows, numCols, nnz,
                                     d_csrRowPtr, d_csrColInd, d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Create Dense Vector Descriptors
    cusparseDnVecDescr_t vecB, vecX;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecB, numRows, d_b, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, numRows, d_x, CUDA_R_32F));

    // Set Fill Mode (Lower) and Diagonal Type (Non-unit)
    cusparseFillMode_t fillMode = CUSPARSE_FILL_MODE_LOWER;
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode)));
    cusparseDiagType_t diagType = CUSPARSE_DIAG_TYPE_NON_UNIT;
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType)));

    // --- 4. Prepare for Solve (Analysis) ---
    cusparseSpSVDescr_t spsvDescr;
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescr));

    float alpha = 1.0f;
    size_t bufferSize = 0;
    
    // Query buffer size for analysis and solve
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matL, vecB, vecX, CUDA_R_32F,
                                           CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, &bufferSize));
    
    void* d_buffer = nullptr;
    CHECK_CUDA(cudaMalloc(&d_buffer, bufferSize));

    // Analysis Phase: Finds dependencies and levels
    CHECK_CUSPARSE(cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha, matL, vecB, vecX, CUDA_R_32F,
                                         CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, d_buffer));

    // --- 5. Execution (Solve) ---
    // Lx = b
    CHECK_CUSPARSE(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha, matL, vecB, vecX, CUDA_R_32F,
                                      CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr));

    // --- 6. Results and Cleanup ---
    CHECK_CUDA(cudaMemcpy(h_x.data(), d_x, numRows * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Solution x: " << std::endl;
    for (int i = 0; i < numRows; i++) {
        std::cout << "x[" << i << "] = " << h_x[i] << std::endl;
    }

    // Cleanup
    CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsvDescr));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecB));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroySpMat(matL));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColInd);
    cudaFree(d_csrVal);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_buffer);

    return 0;
}