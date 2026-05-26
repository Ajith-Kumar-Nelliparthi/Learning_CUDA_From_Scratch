# Histogram Statistics
### 0x00 Explanation
Includes the following:
→ histogram_i32_kernel \
→ histogram_i32x4_kernel (int4 vectorized version) \
→ PyTorch bindings

### Testing
```bash
python histogram.py
```

### Output
```
--------------------------------------------------------------------------------
h_i32   0: 103
h_i32   1: 98
h_i32   2: 93
h_i32   3: 99
h_i32   4: 93
h_i32   5: 104
h_i32   6: 106
h_i32   7: 91
h_i32   8: 100
h_i32   9: 116
--------------------------------------------------------------------------------
h_i32x4 0: 103
h_i32x4 1: 98
h_i32x4 2: 93
h_i32x4 3: 99
h_i32x4 4: 93
h_i32x4 5: 104
h_i32x4 6: 106
h_i32x4 7: 91
h_i32x4 8: 100
h_i32x4 9: 116
--------------------------------------------------------------------------------
```