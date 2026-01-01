# Level - 3: Memory Patterns
**Goal**:
Master GPU Memory Heirarchy.

**Completed Work**
Solved a total of 11 Memory Pattern problems, progressing from naive implementations to highly optimized kernels.

## Implemented Memory Patterns
### 1. [Matrix Transpose (Naive)](https://github.com/Ajith-Kumar-Nelliparthi/Learning_CUDA_From_Scratch/blob/main/Problems/L3-Memory%20Patterns/transpose%20(naive).cu)
**Key learnings:**
- Memory Coalscening issues.

### 2. [Transpose (Optimized with shared memory)](https://github.com/Ajith-Kumar-Nelliparthi/Learning_CUDA_From_Scratch/blob/main/Problems/L3-Memory%20Patterns/matrix%20multiplication%20optimized.cu)
**Key learnings:**
- Shared Memory Tiling, Bank conflicts.

### 3. [Strided Access Analysis](https://github.com/Ajith-Kumar-Nelliparthi/Learning_CUDA_From_Scratch/blob/main/Problems/L3-Memory%20Patterns/strided_access_analysis.cu)
**Key learnings:**
- Impact of Non-Coalsced access.

### 4. [Array of Structure vs Structure of Array](https://github.com/Ajith-Kumar-Nelliparthi/Learning_CUDA_From_Scratch/blob/main/Problems/L3-Memory%20Patterns/SOA_VS_AOS.cu)
**Key learnings:**
- Data Layout Impact.

### 5. [Constant Memory Test](https://github.com/Ajith-Kumar-Nelliparthi/Learning_CUDA_From_Scratch/blob/main/Problems/L3-Memory%20Patterns/constat%20memory%20test.cu)
**Key learnings:**
- When to use constant memory.

### 6. [Shared Memory Bank COnflicts](https://github.com/Ajith-Kumar-Nelliparthi/Learning_CUDA_From_Scratch/blob/main/Problems/L3-Memory%20Patterns/shared%20memory%20bank%20conflicts.cu)
**Key learnings:**
- What causes conflicts, how to avoid them.

### 7. [Shared Memory Padding](https://github.com/Ajith-Kumar-Nelliparthi/Learning_CUDA_From_Scratch/blob/main/Problems/L3-Memory%20Patterns/matrix%20transpose(optimized%20with%20padding).cu)
**Key learnings:**
- Adding Padding to avoid Conflicts.

### 8. [1D Convolution with Shared Memory](https://github.com/Ajith-Kumar-Nelliparthi/Learning_CUDA_From_Scratch/blob/main/Problems/L3-Memory%20Patterns/1D%20Convolution.cu)
**Key learnings:**
- Halo cells, shared memory reuse.

### 9. [Shared Memory Double Buffering](https://github.com/Ajith-Kumar-Nelliparthi/Learning_CUDA_From_Scratch/blob/main/Problems/L3-Memory%20Patterns/shared%20memory%20double%20buffering.cu)
**Key learnings:**
- Learned about Double Buffering.

### 10. [Matrix Multiply (Naive)](https://github.com/Ajith-Kumar-Nelliparthi/Learning_CUDA_From_Scratch/blob/main/Problems/L3-Memory%20Patterns/matrix_multiply(naive).cu)
**Key learnings:**
- Triple Nested loop.

### 11. [Matrix Multiply (Tiled with Shared memory)](https://github.com/Ajith-Kumar-Nelliparthi/Learning_CUDA_From_Scratch/blob/main/Problems/L3-Memory%20Patterns/matrix%20multiplication%20optimized.cu)
**Key learnings:**
- Loading tiles into shared memory and doing computation.

### Summary
This Level focused on Mastering Memory Patterns in CUDA, starting from basic concepts and gradually building toward optimized, production-style kernels.