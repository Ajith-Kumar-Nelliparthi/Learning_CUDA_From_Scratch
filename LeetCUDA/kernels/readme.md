<div align="center">
  <img src="assets/banner.svg" alt="CUDA Kernels from Scratch" width="100%">
</div>

<h1 align="center">CUDA Kernels from Scratch</h1>

<p align="center">
  Hand-written CUDA kernels, ordered easy → hard. Each one implemented and validated against a reference implementation.
</p>

---

## Kernels

| Kernel | Description | Difficulty |
|--------|--------------|:----------:|
| [`elementwise`](./elementwise) | Generic element-wise operation template (add/mul/etc.) applied across an array | ⭐ |
| [`relu`](./relu) | Rectified Linear Unit — `f(x) = max(0, x)` | ⭐ |
| [`sigmoid`](./sigmoid) | Logistic function — `f(x) = 1 / (1 + e⁻ˣ)` | ⭐ |
| [`elu`](./elu) | Exponential Linear Unit — smooth negative saturation for x < 0 | ⭐⭐ |
| [`gelu`](./gelu) | Gaussian Error Linear Unit — smooth, tanh-approximated activation used in transformers | ⭐⭐ |
| [`swish`](./swish) | Self-gated activation — `f(x) = x · sigmoid(x)` | ⭐⭐ |
| [`hardswish`](./hardswish) | Piecewise-linear, hardware-friendly approximation of Swish | ⭐⭐ |
| [`embedding`](./embedding) | Lookup-table gather kernel — maps token indices to embedding vectors | ⭐⭐ |
| [`dot_product`](./dot_product) | Two-vector reduction — `Σ Aᵢ · Bᵢ`, using shared memory | ⭐⭐ |
| [`reduce`](./reduce) | Generic parallel reduction (sum/max/min) with tree reduction + warp shuffle | ⭐⭐ |
| [`histogram`](./histogram) | Binned counting kernel using atomic operations | ⭐⭐ |
| [`mat_transpose`](./mat_transpose) | Tiled matrix transpose — coalesced access, bank-conflict-free shared memory | ⭐⭐ |
| [`layer_norm`](./layer_norm) | Layer Normalization — per-row mean/variance + affine transform | ⭐⭐ |
| [`rms_norm`](./rms_norm) | Root-Mean-Square Normalization (LLaMA-style, no mean subtraction) | ⭐⭐ |
| [`softmax`](./softmax) | Numerically-stable softmax with max-subtraction trick | ⭐⭐ |
| [`rope`](./rope) | Rotary Positional Embedding — rotates query/key vectors by position-dependent angles | ⭐⭐⭐ |

---