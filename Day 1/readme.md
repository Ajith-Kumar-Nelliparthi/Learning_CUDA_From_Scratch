## Day-1 of Learning CUDA:
- Installed the CUDA Toolkit on my laptop even without GPU access, to gain hands-on familiarity with the tools and environment.
- Completed reading Chapter 1 of Programming Massively Parallel Processors (PMPP).
- Understood the design philosophy of CPU vs GPU, why GPUs were created, and how parallelization enables massive speedups for data-parallel workloads.
- Learned Amdahl’s Law and how it limits the performance gain from parallelization.
- Explored the three major hardware components of a GPU:
	- SMs (Streaming Multiprocessors)
	- CUDA Cores
	- L1 & L2 Cache Hierarchy
- Learned the difference between Shared Memory (fast, on-chip, per-block) and Global Memory / DRAM (slower, off-chip, high latency).