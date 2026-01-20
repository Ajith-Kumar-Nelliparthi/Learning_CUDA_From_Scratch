## Day 56 of CUDA
Today implemented a high-performance 2D Heat Equation solver using CUDA. It translates the continuous Laplacian diffusion model into a discrete 5-point stencil operation optimized for GPU architecture.

### **Implementation Details**
- Stencil Operation: 5-point stencil (Center, North, South, East, West) to approximate the 2D Laplacian.
- Memory Optimization: Used 2D Shared Memory Tiling. Each 16x16 thread block loads an 18x18 "tile" into shared memory to handle the spatial halos, reducing global memory pressure.
- Boundary Conditions: Dirichlet boundaries (fixed temperature at edges) implemented via hardware-efficient ternary checks.
- Stability Control: Adhered to the CFL condition for 2D diffusion, maintaining α≤0.25 to ensure numerical stability and prevent divergence.

### **Performance Analysis (Nsight Systems)**
![alt text](<Screenshot 2026-01-20 084552.png>)
Based on a 1024×1024 (1M elements) simulation:
- Kernel Execution: Each iteration of the heat_2d_kernel takes approximately 103 µs.
- Efficiency: Shared memory tiling reduced global memory dependencies from 5 reads per pixel to effectively ~1.1 reads per pixel.
- Bottlenecks: The profiling report identifies the system as Memory Bound. The time taken for cudaMalloc and PCIe data transfers outweighs the 100 iterations of computation, suggesting that larger iteration counts or larger grids would maximize GPU throughput.

### **Visualization**
![alt text](<Screenshot 2026-01-20 084911.png>)
- Data Export: Results are exported as raw binary (.bin) to maintain floating-point precision.
- Python Integration: Utilizes Matplotlib’s hot colormap for thermal heatmap generation, allowing for clear observation of the diffusion patterns.