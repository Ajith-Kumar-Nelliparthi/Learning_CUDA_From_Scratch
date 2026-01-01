# Level - 3: Memory Patterns
**Goal**:
Master GPU Memory Heirarchy.

**Completed Work**
Solved a total of 11 Memory Pattern problems, progressing from naive implementations to highly optimized kernels.

## Implemented Memory Patterns
### 1. Matrix Transpose (Naive)
**Key learnings:**
- Memory Coalscening issues.

### 2. Transpose (Optimized with shared memory)
**Key learnings:**
- Shared Memory Tiling, Bank conflicts.

### 3. Strided Access Analysis
**Key learnings:**
- Impact of Non-Coalsced access.

### 4. Array of Structure vs Structure of Array
**Key learnings:**
- Data Layout Impact.

### 5. Constant Memory Test
**Key learnings:**
- When to use constant memory.

### 6. Shared Memory Bank COnflicts
**Key learnings:**
- What causes conflicts, how to avoid them.

### 7. Shared Memory Padding
**Key learnings:**
- Adding Padding to avoid Conflicts.

### 8. 1D Convolution with Shared Memory
**Key learnings:**
- Halo cells, shared memory reuse.

### 9. Shared Memory Double Buffering
**Key learnings:**
- Learned about Double Buffering.

### 10. Matrix Multiply (Naive)
**Key learnings:**
- Triple Nested loop.

### 11. Matrix Multiply (Tiled with Shared memory)
**Key learnings:**
- Loading tiles into shared memory and doing computation.

### Summary
This Level focused on Mastering Memory Patterns in CUDA, starting from basic concepts and gradually building toward optimized, production-style kernels.