## SGEMV
## 0x00 Explanation

Includes the following：

- [X] sgemv_k32_f32_kernel
- [X] sgemv_k128_f32x4_kernel
- [X] PyTorch bindings

## Testing

```bash
export TORCH_CUDA_ARCH_LIST=Ada
python3 sgemv.py
```

Output:

```bash
--------------------------------------------------------------------------------
   out_k32f32: [7.99294567, -9.91914749, 0.60257053], time:0.00563622ms
out_k128f32x4: [7.99294567, -9.91915035, 0.60256958], time:0.00571609ms
   out_f32_th: [7.99294567, -9.91914749, 0.60256952], time:0.01708508ms
--------------------------------------------------------------------------------
```

### PTX
```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z23sgemv_k128_f32x4_kernelPfS_S_ii' for 'sm_75'
ptxas info    : Function properties for _Z23sgemv_k128_f32x4_kernelPfS_S_ii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 48 registers, used 0 barriers, 384 bytes cmem[0]
ptxas info    : Compile time = 11.032 ms
ptxas info    : Compiling entry function '_Z20sgemv_k32_f32_kernelPfS_S_ii' for 'sm_75'
ptxas info    : Function properties for _Z20sgemv_k32_f32_kernelPfS_S_ii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers, 384 bytes cmem[0]
ptxas info    : Compile time = 8.847 ms
```