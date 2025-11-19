## Day-2 of Learning CUDA

- Completed subtopics 2.1–2.3 from Chapter 2 of Programming Massively Parallel Processors (PMPP).
- Learned about various CUDA function qualifiers (__global__, __device__, __host__, etc.).
- Deepened understanding of the concepts of Kernel, Grid, Block, and Thread in the CUDA programming model.
- Resolved a confusion I had: previously I wondered why we need to calculate a global index for each thread. Today I understood that computing a unique global thread index is essential so each thread processes its own portion of data—otherwise, all threads would operate on the same memory location.
- Learned and practiced the formulas for calculating global thread indices in 1D, 2D, and 3D grid/block configurations.
- Written the vector addition in cuda from scratch
- Ran the vector addition kernel in colab and verified the numeric output.