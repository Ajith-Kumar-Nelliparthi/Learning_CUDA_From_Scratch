## Profiling of Sum Reduction
### Reduction Kernel Overview
The first reduction kernel (v1) has the following characteristics:
- Purpose: Reduce N values (all 1.0f) into a single sum
- Initial Load Optimization:
    - Each thread loads 2 elements from global memory (reduces total memory transactions)
- Shared Memory Usage:
    - Intermediate values are stored in shared memory for fast intra-block communication
- Reduction Strategy:
    - Tree-based reduction, halving the number of active threads at each iteration
This design minimizes global memory accesses and leverages shared memory efficiently.

### Nsight Systems Results — reduction_sum Kernel v1
1. **Kernel Execution Time**
| Time (%) | Total Time (ns) | Instances | Avg (ns) |
| -------- | --------------- | --------- | -------- |
| 100.0    | 169,979         | 3         | 56,659.7 |
**Interpretation**:
- Kernel executed 3 times (one per block-size configuration)
- Average execution time: ~56.7 µs
- Total GPU compute time: ~170 µs
**Analysis**:
- For reducing 1 million elements, this is excellent performance.
- The kernel is compute-efficient and well-optimized.

2. **Memory Transfer Time (The Real Bottleneck!)**
| Time (%) | Total Time (ns) | Operation     |
| -------- | --------------- | ------------- |
| 99.7     | 2,306,159       | Host → Device |
| 0.3      | 7,648           | Device → Host |

**Key Insight:**
- Host → Device copy: ~2.3 ms
- Kernel execution: ~0.17 ms
- Device → Host copy: ~0.008 ms
👉 Memory transfer is ~13.6× slower than computationThis is classic "memory-bound" behavior.
```yaml
Memory Transfer:  ████████████████████████████  (2.3 ms)
Kernel Execution: ██                            (0.17 ms)
```
**Conclusion**:
This workload is memory-bound, not compute-bound. The GPU spends far more time waiting on PCIe transfers than performing computation.

3. **CUDA API Overhead**
| Time (%) | Total Time (ns) | API Call   |
| -------- | --------------- | ---------- |
| 95.8     | 90,344,360      | cudaMalloc |
| 3.1      | 2,915,209       | cudaMemcpy |
| 0.5      | 475,069         | cudaFree   |

- **Shocking Discovery**: cudaMalloc took 90 milliseconds! This is because:
    - First allocation initializes GPU context
    - Memory allocation is expensive
    - Multiple allocations inside loops amplify this overhead


