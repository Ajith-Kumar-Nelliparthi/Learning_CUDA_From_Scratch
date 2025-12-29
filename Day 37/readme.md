## Day 37 of Learning CUDA:
- Solved reduce by key problem in naive kernel.
- Completed Level-2: Reduction Patterns - Combining Data. Moving on to Level-3.
- Solved Naive matrix transpose and observed the perfomance with nsight systems
**Performance Breakdown**
```
Operation:        Matrix Transpose (1024×1024) Naive
Kernel Time:      192 μs
Achieved BW:      43.7 GB/s
Efficiency:       14.7% of peak
Bottleneck:       UNCOALESCED WRITES

Breakdown:
  Read efficiency:   ~90% (coalesced)
  Write efficiency:  ~3% (strided by 1024)
  Average:           ~15% ✗
```
- solved matrix transpose with shared memory achieved 3X speedup than naive.