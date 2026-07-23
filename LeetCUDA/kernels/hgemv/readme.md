## HGEMV
## 0x00 Explanation

Includes the following：

- [X] hgemv_k32_f16_kernel
- [X] hgemv_k128_f16x4_kernel
- [X] PyTorch bindings

## Testing

```bash
export TORCH_CUDA_ARCH_LIST=Ada
python3 hgemv.py
```

Output:

```bash
--------------------------------------------------------------------------------
   out_k32f16: [-0.921875, 0.24414062, -12.03125], time:0.00540614ms
out_k128f16x4: [-0.921875, 0.25390625, -12.015625], time:0.00550508ms
   out_f16_th: [-0.91552734, 0.2454834, -12.015625], time:0.01600504ms
--------------------------------------------------------------------------------
```

### PTX
```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z23hgemv_k128_f16x4_kernelP6__halfS0_S0_ii' for 'sm_75'
ptxas info    : Function properties for _Z23hgemv_k128_f16x4_kernelP6__halfS0_S0_ii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 31 registers, used 0 barriers, 384 bytes cmem[0]
ptxas info    : Compile time = 81.217 ms
ptxas info    : Compiling entry function '_Z20hgemv_k32_f16_kernelP6__halfS0_S0_ii' for 'sm_75'
ptxas info    : Function properties for _Z20hgemv_k32_f16_kernelP6__halfS0_S0_ii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers, 384 bytes cmem[0]
ptxas info    : Compile time = 5.893 ms
```