## Day-4 of Learning CUDA:
- Completed Chapter 2 of PMPP.
- Learned about Warps, how they function, and their role in GPU execution.
- Compared CPU vs GPU vector addition performance.
- Observed that the GPU ran ~31.6× faster than the CPU.
- Implemented array multiplication with different block sizes: 64, 128, 256.
- Analyzed how execution time changes with block size.
- Found that larger block sizes (more threads per block) reduce the number of blocks, improving execution time due to better occupancy and less scheduling overhead.

### Experiment Results
- Block Size: 64
- Blocks: 157
- GPU Time: 0.141 ms
  First 5 results:

c[0] = 0.33 (0.84 * 0.39)
c[1] = 0.63 (0.78 * 0.80)
c[2] = 0.18 (0.91 * 0.20)
c[3] = 0.26 (0.34 * 0.77)
c[4] = 0.15 (0.28 * 0.55)

- Block Size: 128
- Blocks: 79
- GPU Time: 0.014 ms
  First 5 results:

c[0] = 0.33 (0.84 * 0.39)
c[1] = 0.63 (0.78 * 0.80)
c[2] = 0.18 (0.91 * 0.20)
c[3] = 0.26 (0.34 * 0.77)
c[4] = 0.15 (0.28 * 0.55)

- Block Size: 256
- Blocks: 40
- GPU Time: 0.010 ms
  First 5 results:

c[0] = 0.33 (0.84 * 0.39)
c[1] = 0.63 (0.78 * 0.80)
c[2] = 0.18 (0.91 * 0.20)
c[3] = 0.26 (0.34 * 0.77)
c[4] = 0.15 (0.28 * 0.55)
