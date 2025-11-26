## Day-9 of Learning CUDA:
- Started reading Chapter 5 of PMPP and focusing on the Reduction Sum Kernel — it's a large topic, so progressing steadily.
- [links](https://kathsucurry.github.io/cuda/2025/10/27/reduction_sum_part2.html)

1) Implemented Naive Interleaved Addressing (Reduction) though it's taken time to understand kernel logic.
- Here's the performance results:
```
Testing block size: 128 (blocks: 4096)
Time taken for v1: 0.24 ms
Final Sum = 1048576 (expected 1048576)

Testing block size: 256 (blocks: 2048)
Time taken for v1: 0.07 ms
Final Sum = 1048576 (expected 1048576)

Testing block size: 512 (blocks: 1024)
Time taken for v1: 0.07 ms
Final Sum = 1048576 (expected 1048576)
```
- Observed that 128 threads slower than 256 / 512 because we are taking 2 elements per thread. so no.of blocks "blocks = N / (threads / 2)" . More threads per block = fewer blocks = less kernel launch overhead.

2) Implemented Interleaved version 2 but avoids shared memory bank conflicts.
- Output:
```
128 threads → 0.1378 ms
256 threads → 0.0773 ms
512 threads → 0.0880 ms
```
- observed v2 is significantly faster than v1 cause v2 solves bank conflicts and v2 introduces more divergent control flow.

3) Implemented Sequential Adressing
- observed v3 is faster than v2 and v1, because loads 2 elements per thread, shared memory tree reduction and no divergence inside reduction loop
- output:
```
128 threads → 0.1103 ms
256 threads → 0.0661 ms
512 threads → 0.0695 ms
```
4) Implemented synchronization kernel where removed the  synchronization and added after for loop.
- Output:
```
128 threads → 0.1389 ms
256 threads → 0.0652 ms
512 threads → 0.0694 ms
```