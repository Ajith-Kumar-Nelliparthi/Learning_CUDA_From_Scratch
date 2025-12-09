## Day 19 of Learning CUDA:
- Solved Relu  and LeakyRelu in LeetGPU
- Implemented Max Reduction kernels from Naive to Optimized versions.

## 1. Project Overview
Goal: Efficiently calculated the maximum value in a large array (e.g., 16 million elements) using CUDA.
Methodology: Implements and compares 6 different kernels, starting from a naive approach and evolving into a fully optimized, vectorized implementation.
Outcome: Demonstrates a massive performance jump from the CPU baseline and the naive GPU kernel to the final optimized kernel.

### 2. Kernel Evolution (The Optimization Steps)
- Naive Kernel: Uses global memory writes for every thread. Extremely slow due to memory latency and lack of reduction logic.
- Kernel 1 (Interleaved Addressing): Introduces Shared Memory to reduce global memory access but suffers from Warp Divergence (threads are inactive within warps).
- Kernel 2 (Interleaved w/o Divergence): Fixes divergence issues but creates Shared Memory Bank Conflicts (multiple threads accessing the same memory bank).
- Kernel 3 (Sequential Addressing): changes access pattern to eliminate bank conflicts.
- Kernel 4 (Thread-Level / Grid-Stride):
    - Grid-Stride Loop: Decouples grid size from array size (can process arrays larger than the grid).
    - ILP (Instruction Level Parallelism): Each thread reduces multiple elements in a local register before writing to shared memory.
- Kernel 5 (Vectorized & Warp Shuffle):
    - float4 Access: Loads 128 bits (4 floats) per instruction to maximize memory bandwidth.
    - Warp Shuffle (__shfl_down_sync): Performs reduction at the register level within a warp, avoiding shared memory latency for the final steps.

### 3. Key Concepts Conered
- Global vs. Shared Memory latency.
- Warp Divergence optimization.
- Shared Memory Bank Conflicts.
- Grid-Stride Loops (Persistent Threads).
- Vectorized Memory Access (float4).
- Warp primitives (Shuffle Down).

### 4. Performance Results
- **Hardware**: Colab T4
- **Data Size**: 16,777,216 elements (64MB).
- **Benchmarks**:
    - CPU: ~42ms
    - Naive GPU: ~19,850ms (Bottlenecked by global memory)
    - Shared Memory (Kernel 3): 0.68ms
    - Optimized (Kernel 5): 0.28ms
- **Speedup**: The final kernel is ~150x faster than CPU and saturates the GPU memory bandwidth (~228 GB/s effective throughput).

## 5. How to Compile and Run
code
```Bash
# Compile in Colab
nvcc -arch=sm_70 main.cu -o main
```
### Run
```
./main
```

### 6. File Structure
- main.cu: Contains the host code (memory management, timing logic) and all CUDA device kernels.
```
| Kernel Strategy           | Time (ms)   | Speedup vs CPU | Key Bottleneck Solved            |
|---------------------------|-------------|-----------------|----------------------------------|
| CPU                       | 42.24       | 1x              | N/A                              |
| Naive                     | 19,850.49   | <1x             | Global Memory Latency            |
| Interleaved (Mod)         | 1.11        | ~38x            | Global Memory Bandwidth          |
| Interleaved (Strided)     | 0.84        | ~50x            | Warp Divergence                  |
| Sequential                | 0.68        | ~62x            | Bank Conflicts                   |
| Grid-Stride Loop          | 0.29        | ~145x           | Instruction Overhead             |
| Vectorized (float4)       | 0.28        | ~150x           | Memory Instruction Throughput    |
```
