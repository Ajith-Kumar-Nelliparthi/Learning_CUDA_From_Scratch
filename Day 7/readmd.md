## Day-7 of Learning CUDA:
- Completed Chapter 3 of PMPP.
- Understood why 2D matrices are stored in 1D arrays in CUDA — because GPU memory is linear, and flattening the matrix ensures efficient coalesced memory access.
- Implemented Matrix Addition (CPU vs GPU).
- Observed that for block size (16, 16), the CPU ran faster than the GPU.
- Understood the reason: CPUs are highly optimized for sequential, low-latency tasks, whereas GPUs excel at high-throughput parallel workloads.
- This reinforced the classic Latency vs Throughput architectural difference between CPUs and GPUs.

# Why the CPU Has Lower Latency Than the GPU for Small Problem Sizes
When operating on small data sizes (like a 16×16 matrix), the CPU often runs faster than the GPU. This is because the overhead of launching work on the GPU is larger than the amount of actual computation being performed.

Below is the detailed explaniation through an example:

1. “Bus vs Ferrari” Analogy
- CPU = Ferrari
    - Designed for low-latency, high-speed sequential tasks
    - Very high clock speeds (3.0 GHz – 5.0 GHz)
    - Can complete small tasks extremely quickly

- GPU = City Bus
    - Designed for high-throughput, massively parallel workloads
    - Lower clock speeds (1.0 GHz – 1.5 GHz)
    - Slow to start but can move thousands of threads at once

Small workload (e.g., 256 elements):
- The Ferrari (CPU) finishes before the Bus (GPU) even starts moving.
Large workload (e.g., millions of elements):
- The Ferrari must make too many trips
- The Bus transports thousands at once and wins easily

2. Kernel Launch Overhead
Before the GPU begins calculations, the following must happen:
- CPU prepares the kernel launch configuration
- Commands are sent across the PCIe bus
- GPU driver schedules the kernel
- Streaming Multiprocessors (SMs) wake up and start executing

This overhead is fixed (usually a few microseconds), regardless of workload size.
- For small arrays, the actual computation is so tiny that:
    - It finishes faster on the CPU than the time it takes to even start the GPU.

3. Clock Speed Differences
- CPU core: ~4.0 GHz
- GPU core: ~1.4 GHz

For tiny workloads:
- A CPU can brute-force a loop extremely fast
- GPU relies on parallelism to compensate for lower clock speed
- But parallelism doesn’t help when the workload itself is too small
So the CPU simply computes the result faster due to raw per-core speed.

4. GPU Memory Latency Hiding Doesn't Work on Small Grids
- GPUs hide memory latency by switching between thousands of active threads.
- If one block stalls, GPU instantly switches to another block.
- But with a small workload:
    - Few threads = few warps
    - Few warps = nothing to switch to
    - Memory latency is fully exposed
    - Many GPU cores remain idle
Thus, the GPU never reaches its designed throughput.

### Summary
```
| Problem Size	| CPU	| GPU |
|Small N    | Wins (low latency, fast clock, zero overhead)	| Loses (launch + scheduling overhead dominate) |
|Large N	| Loses (limited cores)	 | Wins (thousands of threads, massive throughput) |
```
### Final takeaway:
- CPU is optimized for low-latency sequential work.
- GPU is optimized for high-throughput parallel work.
- For small workloads, latency dominates → CPU wins.
- For large workloads, throughput dominates → GPU wins.
