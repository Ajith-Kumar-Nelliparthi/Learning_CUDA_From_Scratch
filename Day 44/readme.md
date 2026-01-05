## Day 44 of learning CUDA:
- Solved Segmented Scan in naive method.
- Solved Segmented scan in block level wiht shared memory.

### Report: Performance Analysis of Segmented CUDA Scan
- Dataset Size:  (1,048,576) int32 elements
- Algorithm: Hillis-Steele Segmented Scan (Shared Memory Implementation)
- Hardware Profile: Measured via NVIDIA Nsight Systems (nsys)

1. **Executive Summary**
At a scale of 1 million elements, the GPU execution is IO-Bound. While the compute kernel is highly efficient (finishing in under 150 microseconds), the total execution time is dominated by memory transfers between the Host (CPU) and Device (GPU). The use of CUDA Streams and a "Warmup" kernel provides a more stable profiling environment and prepares the code for advanced latency-hiding techniques.

2. **Key Performance Metrics**
| Metric                | Synchronous (Normal) | Asynchronous (Streams) | Variance    |
|-----------------------|----------------------|------------------------|-------------|
| Kernel Execution (Scan) | 144,252 ns           | 144,029 ns             | Negligible  |
| Warmup Kernel         | N/A                  | 75,806 ns              | -           |
| Host-to-Device (8MB)  | 1.55 ms              | 1.62 ms                | +4.5%       |
| Device-to-Host (4MB)  | 1.56 ms              | 2.13 ms                | +36.5%      |
| Compute-to-IO Ratio   | ~1:21                | ~1:25                  | IO Dominant |
