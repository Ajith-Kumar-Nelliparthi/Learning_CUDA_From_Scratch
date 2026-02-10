## Day 70 of CUDA
- Solved Tree Reduction vs Atomic Reduction
- Observed that for 1M elements atomic condition is faster than tree reduction but larger arrays it's tree reduction scales better.
- Memory allocation and transfers dominate runtime.