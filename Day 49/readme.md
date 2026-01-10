## Day 49 of CUDA:
- Solved 1D wave Equation using Triple Buffer Pointers.
- ### **Report**:
    - Kernel time: ~898.7 µs per iteration (Total: ~898.7 ms for 1,000 steps).
    - Host→Device copy: ~28.75 ms (134.2 MB total for prev and curr).
    - Device→Host copy: ~13.86 ms (67.11 MB for the final state).
- ### **Observation**: 
    The system is Memory Bound at the kernel level but highly efficient at the application level due to:
    - Effective Triple Buffering: By rotating pointers on the GPU, avoided 1,000 rounds of PCIe transfers. If copied data back to the host every step, the simulation would have taken ~42 seconds; instead, it took less than 1 second.
    - High Memory Pressure: Each thread must load from two different global memory arrays (curr and prev) and write to a third (next).  This results in a very low "Arithmetic Intensity"—meaning the GPU spends most of its time waiting for the memory controllers to fetch data from VRAM.
    - Shared Memory Utilization: use of __shared__ memory for the curr array successfully reduced global memory reads for the spatial neighbors (the Laplacian), but the prev array still requires a unique global fetch per thread, which is the primary bottleneck.
    - Scaling Advantage: Unlike earlier single-pass kernels, here the Kernel Execution (900ms) finally outweighs the PCIe Overhead (42ms). This is the "sweet spot" for GPU acceleration: keeping the data on the device for as many operations as possible.