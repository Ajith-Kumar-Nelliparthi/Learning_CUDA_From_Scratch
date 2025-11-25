## Day-8 of Learning CUDA:
- Completed Chapter 4 of PMPP.
- Learned about Shared Memory and its two types: Static and Dynamic Shared Memory.
- Learned the concept of Tiling (Blocking) and why it is used to improve memory access patterns in matrix multiplication.
- Started implementing matrix multiplication using both tiling and shared memory. The logic was initially challenging, especially understanding how tiles map to shared memory.
- Misunderstood the relationship between tiles and shared memory and reached out to ChatGPT for clarification.
- Finally wrote a working matrix multiplication kernel using tiling + shared memory.
- Realized I need more practice to understand tiling deeply, but this foundation is important because most future CUDA programs involve 2D and 3D data (images, matrices, volumes, etc.).