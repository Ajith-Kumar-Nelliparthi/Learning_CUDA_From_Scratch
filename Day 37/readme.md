## Day 37 of Learning CUDA:
- Solved reduce by key problem in naive kernel.
- Completed Level-2: Reduction Patterns - Combining Data. Moving on to Level-3.
- solved Matrix transpose from naive to optimized versions and recorded their perfomances.
### Performance Achieved
- V1 (Naive): 43.7 GB/s (15% efficiency)
- V2 (Shared): 240 GB/s (81% efficiency) → **5.5× speedup**
- V3 (Padded): 271 GB/s (91% efficiency) → **6.2× speedup**

### Key Learnings
1. Shared memory eliminates uncoalesced writes
2. Block coordinate swapping is the key trick
3. Padding avoids bank conflicts
4. Proper benchmarking requires multiple run