## Day 58 of CUDA: Sobel Edge Detection
Today done a high-performance implementation of the Sobel Operator for edge detection, optimized for NVIDIA GPUs using CUDA.
### Technical Highlights
- **Dual-Gradient Kernel:** Calculated both Horizontal (Gx) and Vertical (Gy) gradients in a single kernel pass.
- **Shared Memory Tiling:** Implemented a collaborative loading strategy to bring 18x18 pixel tiles into Shared Memory for 16x16 thread blocks. This reduced global memory traffic by nearly 90%.
- **Constant Memory Optimization:** Sobel weights are stored in __constant__ memory, utilizing the GPU's constant cache for broadcasted access across warps.
- **Fused Magnitude Calculation:** The gradient magnitude (Gx2+Gy2) is computed on-the-fly within the registers to minimize global memory writes.

### Performance Report (Nsight Systems)
- **Grid Size:** 1024 x 1024 (1M pixels)
- **Kernel Execution Time:** ~129 µs
- **Observation:** The kernel is highly efficient. Computing two simultaneous convolutions (X and Y directions) only added a negligible 6 µs compared to a standard single-pass 2D convolution (~123 µs), proving the efficiency of reusing shared memory tiles for multiple operations.
- **Bottleneck Analysis:** The execution remains IO Bound. Memory transfers (HtoD/DtoH) take ~2.4 ms total, which is ~18x longer than the kernel computation itself.