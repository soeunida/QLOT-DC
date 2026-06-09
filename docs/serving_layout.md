# StaticScale serving layout (packed FP/INT prototype)

This describes the static per-layer layout a serving kernel would consume — the
**packed FP/INT prototype**. **Experimental / reference-only:** the default
`torch_reference` backend is fake-quantized (dequant + FP matmul); `custom_packed`
is a stub. No backend-independent speedup is claimed and no real kernel is
implemented; any throughput figures are **prototype diagnostics only**.

## `[FP block | INT block]`

For each routed projection (gate_proj, up_proj), the input-channel dimension is
permuted into two contiguous blocks:

```
columns:  [  FP block (K_F)  |        INT block (C_int)        ]
          ^ original order      ^ packing-aware order
W_F = W[:, fp_indices]   (FP16, [O, K_F])
W_I = W[:, int_indices]  (INT8 W8-G128, [O, C_int])
```

`K_F = k_fp` is a static split offset (per layer; under the global budget it can
differ across layers). The FP and INT partial outputs are summed in the same
output space; **no inverse permutation** is needed (channels are pre-permuted).

## Packing-aware INT permutation

INT channels are reordered so each contiguous group of `w8_group_size` (128)
input channels has a similar per-channel activation scale. W8-G128 uses one
symmetric scale per (output, group); making within-group activation/weight scales
more uniform reduces quantization error. The order is frozen in
`int_indices`; the activation scales `act_scales` are aligned to it.

## Tensors (per routed projection, one layer)

| tensor | shape | dtype | source |
|---|---|---|---|
| `y_F` | `[T, K_F]` | fp16 | `u_F·γ_F + β_F` |
| `y_I` / `q` | `[T, C_int]` | fp16 / int8 | `u_I·γ_I + β_I`, then per-channel quant |
| `act_scales` | `[C_int]` | fp32 | frozen, per INT channel |
| `W_F` | `[O, K_F]` | fp16 | FP weight slice |
| `W_I` (codes + scales) | `[O, C_int]` int8 + `[O, n_groups]` fp16 | INT8 W8-G128 |
| `bias` | `[O]` or None | fp16 | projection bias |
| output `z` | `[T, O]` | fp16 | FP32-accumulated sum |

`T` = tokens (prefill), `C = K_F + C_int` (LN2 hidden), `O` = intermediate_size,
`n_groups = ceil(C_int / 128)`.

## Backend status

- `torch_reference` (default): correctness oracle — dequantizes the W8-G128
  weight and runs FP32 matmuls. **Slower / higher-memory than FP16 by design.**
- `custom_packed`: **stub** (`NotImplementedError`). A real CUDA/Triton fused
  branched kernel (FP16 over the FP block + INT8 W8-G128 with INT32 accumulation
  over the INT block, summed in FP32 → FP16) must pass correctness tests vs the
  reference before being enabled. Until then, **no speedup is claimed**.

Export static artifacts with `python -m staticscale.serving_export` (writes
`indices.pt`, `scales.pt`, `gate.pt`, `up.pt`, `meta.json`, `manifest.json`).
Validate readiness with `eval/check_custom_packed_readiness.py`.
