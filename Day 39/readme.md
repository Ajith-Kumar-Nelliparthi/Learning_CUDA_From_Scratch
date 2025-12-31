## Day 39 of Learning CUDA:
![alt text](<Screenshot 2025-12-31 062502.png>)
- Implemented a CUDA kernel to benchmark Constant Memory performance against Global Memory.
- Observed that Global Memory actually outperformed Constant Memory at small scales (N=1024). This is because the data set was small enough to fit entirely within the L2 cache, making the specialized constant cache overhead unnecessary.
- Corrected a memory management error: I initially tried to call free() on stack-allocated host arrays. I’ve since reinforced my understanding that stack memory is automatically deallocated, while free() is reserved for heap memory (allocated via malloc).
- Implemented 1D Convolution.
- Implemented Shared Memory Bank Conflicts with2 kernels specifically (With No Conflicts and With Conflicts (2, 26, 32)). And Here's the results:
    - No Bank Conflict Time:    0.733 ms
    - Stride 2 Conflict Time:   0.728 ms
    - Stride 16 Conflict Time:   0.700 ms
    - Stride 32 Conflict Time:   0.700 ms
- Solved Matrix multiplication form naive to Shared memory and noted the differences btw them.