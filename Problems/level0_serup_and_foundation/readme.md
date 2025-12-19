## L0.2: Hello World Kernels
This section contains three introductory CUDA problems designed to build intuition about kernel launches, thread indexing, and memory management.

### **Problem 0.1: Hello from GPU**
**Goal**:
Print "Hello from GPU!" from each thread.
Key Concepts Learned:
- How to write and launch a CUDA kernel.
- Understanding the basic structure of GPU programs.
- Seeing parallel execution in action (multiple threads printing simultaneously).
Time Taken: ~5 mins

### **Problem 0.2: Thread Index Explorer**
**Goal**:
Print blockIdx, threadIdx, and global index for the first 32 threads.
Key Concepts Learned:
- Thread hierarchy in CUDA (grid → block → thread).
- How global thread IDs are computed using blockIdx.x * blockDim.x + threadIdx.x.
- Visualizing how threads are distributed across blocks.
Time Taken: ~5 mins

### **Problem 0.3: Memory Copy Test**
**Goal**:
Copy an array from host to device and back.
Key Concepts Learned:
- Using cudaMalloc to allocate memory on the GPU.
- Using cudaMemcpy to transfer data between host and device.
- Freeing memory with cudaFree and free.
- Importance of synchronization (cudaDeviceSynchronize) to ensure kernel completion.
Time Taken: ~3 mins

### Overall Learnings from Level 0.2
- How to launch kernels and control thread execution.
- How to index threads correctly to map them to data.
- How to manage memory between host (CPU) and device (GPU).
- Gained confidence in debugging simple CUDA programs and understanding output ordering.
- Learned that GPU printf output can appear interleaved due to parallel execution.
