## Day 38 of Learning CUDA:
![alt text](<Screenshot 2025-12-30 082901.png>)
- SOlved Strided access analysis with bunch of strides.
- This benchmark demonstrates the impact of strided global memory access on effective bandwidth. With unit stride, memory accesses are fully coalesced and achieve near-peak bidirectional DRAM throughput. As stride increases, each warp touches more cache lines, increasing memory transactions and reducing effective bandwidth approximately inversely with stride.
- I also learned that Vectorised memory access is also a type of **coalesced**.
- Because:
    - Each thread loads 16 bytes
    - Warp loads 512 bytes
    - Served in 4 transactions

