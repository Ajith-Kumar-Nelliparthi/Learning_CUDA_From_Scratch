# Level - 2: Reductions - Combining Data

**Goal**:
Learn how to efficiently combine multiple inputs into a single output using CUDA reduction patterns.

**Completed Work**
Solved a total of 12 reduction problems, progressing from naive implementations to highly optimized kernels.

## Implemented Reductions
### 1. Sum Reduction (Naive → Most Optimized)
**Key learnings:**
- Shared memory usage
- Tree-based reduction
- Warp-level reduction using shuffle instructions
- Vectorized memory access with warp shuffle
- Multi-phase (hierarchical) reduction

### 2. Max and Min Reduction (Naive → Optimized)
- Implemented and optimized both maximum and minimum reductions
- Focused on reducing global memory access and improving parallel efficiency

### 3. Maximum Value with Index (Optimized)
**Key learnings:**
- Reducing compound data types (value + index)
- Preserving index information during reduction

### 4. Count Non-Zero Elements, Dot Product, and L2 Norm
**Key learnings:**
- Conditional reductions
- Two-array reductions
- Vector magnitude computation

### 5. Advanced Reductions
- Implemented the following advanced patterns:
- Mean
- Variance
- Histogram (simple)
- Reduce-by-key (simple)

**Key learnings:**
- Two-pass algorithms
- Atomic operations
- Handling race conditions
- Key–value based reductions

### Summary
This level focused on mastering reduction patterns in CUDA, starting from basic concepts and gradually building toward optimized, production-style kernels. Each problem strengthened understanding of performance-critical ideas such as memory hierarchy, warp-level primitives, and synchronization.

Building block by block, one reduction at a time