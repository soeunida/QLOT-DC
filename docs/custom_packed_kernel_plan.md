# Q-LOT-RMS `custom_packed` backend — roadmap & minimal prototype plan

Status: **NOT implemented.** `qlot_rms/projection.py::CustomPackedBackend` raises
`NotImplementedError`. This document specifies exactly what a real kernel must do
so it is a drop-in replacement for `TorchReferenceBackend`, verifiable against the
reference within tolerance. **No speedup is claimed until a real kernel is
implemented and benchmarked here.**

The `torch_reference` backend is correctness-only (fake quant + dequant + FP32
matmul + extra branch) and is ~2.5× slower than FP16; it is near its practical
floor. Real throughput requires the branched FP16+INT8 kernel below.

---

## 1. Where it plugs in

Per routed decoder layer (`mlp_only`), the FFN input is the pre-affine
LN2/RMSNorm activation `u` (original channel order). The static permutation
`P = [FP, INT]` is applied, then two projections (`gate_proj`, `up_proj`) consume
the **same** branch inputs. The kernel replaces the per-projection matmul:

```
z = FP_branch(y_F, W_F) + INT_branch(quant(y_I), W_I_packed) [+ bias]   # FP16 out
```

All routing/scales/permutation are **frozen at calibration time** — the kernel
does **no** runtime routing, top-k, or sort.

---

## 2. Target API

A drop-in `ProjectionBackend` (same signatures as `TorchReferenceBackend`):

```python
class CustomPackedBackend(ProjectionBackend):
    def fp_matmul(self, y_F, W_F) -> z_F          # FP16 GEMM
    def int_matmul(self, y_I, W_I, act_scales, w_group_size, qmax) -> z_I
```

OR a single fused entry point (preferred for performance), consumed by
`PackedProjection`/`QLotRmsFFN`:

```python
packed_matmul(
    y_F: fp16[B, S, K_F],          # FP branch input (u_F * gamma_F + beta_F)
    yq:  int8[B, S, C_int],        # pre-quantized INT branch activation
    a_scales: fp32[C_int],         # per-input-channel activation scales
    W_F: fp16[O, K_F],             # FP weight slice
    Wq:  int8[O, C_int],           # INT8 W8-G128 weight codes
    s_w: fp16[O, n_groups],        # per-(out, group) weight scales
    bias: fp16[O] | None,
    group_size: int = 128, qmax: int = 127,
) -> z: fp16[B, S, O]
```

GroupRMS + affine remain an elementwise **pre-pass** producing `y_F` and `y_I`
(then `yq = round(clamp(y_I / a_scales, -qmax, qmax))`), or may be fused into the
preceding norm epilogue in a later stage.

---

## 3. Tensor shapes

`B` batch, `S` seq, `C = K_F + C_int` (LN2 hidden), `O` output
(= intermediate_size for gate/up). Split is along the **input** dim; `O`
unchanged; partial outputs summed in the same output space; **no inverse
permutation** (channels pre-permuted `[FP, INT]`).

| tensor | shape | dtype |
|---|---|---|
| `y_F` | `[B, S, K_F]` | fp16 |
| `y_I` / `yq` | `[B, S, C_int]` | fp16 / int8 |
| `a_scales` | `[C_int]` | fp32 |
| `W_F` | `[O, K_F]` | fp16 |
| `Wq` (INT codes) | `[O, C_int]` | int8 |
| `s_w` (group scales) | `[O, n_groups]`, `n_groups = ceil(C_int/128)` | fp16 |
| `z` (output) | `[B, S, O]` | fp16 |

---

## 4. Packed FP/INT layout

- `K_F = floor(fp_ratio * C)` is a **static split offset** baked at pack time.
- `W_F` = FP16 columns for FP channels; `W_I` = INT channels.
- Recommended packing: a single contiguous buffer per projection holding
  `[W_F (fp16) | Wq (int8) + s_w (fp16) + group layout]`, with `K_F` and the
  per-group sizes (last group may be smaller) in a small header. Pad `C_int` to a
  multiple of `group_size` (128) for aligned group loads; padded weight codes = 0.

---

## 5. W8-G128 scale layout

Symmetric, per-group, along the **input** dim (matches `quant.py`):
`s_w[o, g] = max(|W_I[o, group_g]|) / qmax`; `Wq[o, c] = round(W_I[o, c] / s_w[o, g(c)])`.
- Mean-scale compensation (`mu_g`, per group) is **folded into the weight at pack
  time** (`W_I[:, group_g] *= mu_g[g]`) only when GroupRMS is enabled for the
  layer; the kernel then needs no `mu_g`. On TinyLlama GroupRMS is gated off, so
  no folding and no GroupRMS pre-pass is needed (routing-only path).

---

## 6. Activation quantization path

Per **input channel**, symmetric, frozen scales `a_scales[c]` (calibrated):
`yq[.., c] = round(clamp(y_I[.., c] / a_scales[c], -qmax, qmax))` (int8).
Quantize **once per layer** (gate & up share `y_I`/`a_scales`). In a fused kernel
this is the INT prologue; in a staged kernel it is a separate elementwise op.

---

## 7. FP branch path

Standard FP16 GEMM over the first `K_F` input channels:
`z_F[.., o] = sum_{c in FP} y_F[.., c] * W_F[o, c]`. Tensor-core fp16, fp32 accum.

---

## 8. INT branch path

Per output row `o`, per group `g` of 128 input channels:
```
acc_int32[o] = sum_g ( s_w[o, g] * sum_{c in group g} yq[.., c] * Wq[o, c] )
```
Inner products in **INT32**, scaled by the per-group fp16 `s_w`, accumulated into
the fp32 output accumulator. Per-channel `a_scales` are applied either by
pre-scaling `yq` back (`yq * a_scales`, the fake-quant reference) or, for a true
INT kernel, by folding `a_scales` into the dequant of the partial sums (note:
per-channel act scale × per-group weight scale cannot be a single factor outside
the group sum, so apply `a_scales` per channel inside or before the INT MAC).

---

## 9. Accumulation dtype

- INT8×INT8 products: **INT32** accumulation within a group.
- Group/branch partial sums and the FP+INT sum: **FP32** accumulator.
- Final store: **FP16** (matches the reference, which casts the fp32 result to fp16).

---

## 10. CUDA / Triton implementation stages

1. **Stage 0 (host packing):** build `W_F`, `Wq`, `s_w`, headers (`K_F`,
   group sizes) from `LayerRouting` + the projection weights. Pure Python/torch.
2. **Stage 1 (Triton, unfused):** Triton `int_matmul` (INT8 × W8-G128, INT32→FP32)
   + reuse torch FP16 GEMM for the FP branch; sum in fp32. Validate vs reference.
3. **Stage 2 (Triton, branched fused):** one kernel that tiles `O`, runs the FP16
   partial over `[0:K_F)` and the INT8 partial over `[K_F:C)`, sums into one fp32
   accumulator, writes fp16. Activation quant in the prologue.
4. **Stage 3 (epilogue fusion):** fold GroupRMS + affine + activation-quant into
   the preceding norm epilogue (only needed when GroupRMS is enabled).
5. **Stage 4 (CUDA/CUTLASS, optional):** replace the Triton GEMMs with CUTLASS
   mixed-input/INT8 kernels for peak throughput.

Wire `CustomPackedBackend` to call the packed kernel; keep `TorchReferenceBackend`
as the default and the correctness oracle.

---

## 11. Correctness tests needed (before enabling)

- `custom_packed` `int_matmul` vs `simulated_int8_matmul` within tolerance
  (per-element rel-err and output L2; INT path may differ from the fp32
  fake-quant reference at the ULP level — define an explicit tolerance, e.g.
  rel-L2 ≤ 1e-2, and document it).
- Packed-vs-reference **per-layer projection output** parity within tolerance.
- Patched-model forward parity within tolerance on a tiny model.
- End-to-end WikiText-2 PPL parity within a small absolute tolerance (e.g.
  ≤ 0.02 PPL) vs the `torch_reference` selected config.
- No runtime top-k/sort introduced (monkeypatch guard, as in
  `tests/test_perf_equivalence.py`).
- Shapes/dtypes: output `[B, S, O]` fp16; `K_F + C_int == C`.

Run `python -m eval.check_custom_packed_readiness.py --plan <routing.pt>` first
to confirm the calibration artifacts are kernel-ready.

---

## 12. Benchmark protocol

- Same `eval/benchmark.py` harness, **same** model, fp_ratio, branch shapes,
  routed layers, batch, seq_len, and kernel path.
- Report `timing_backend="custom_packed"` vs `fp16` and vs `torch_reference`.
- A speedup may be reported **only** when measured with the custom kernel against
  FP16 under identical conditions; never for `torch_reference`.
- Include peak memory; the packed INT layout should reduce weight memory vs the
  fp32-dequant reference cache.

---

## 13. Limitations

- Mixed per-channel activation scale × per-group weight scale prevents a single
  global dequant factor; handle scales per channel/group inside the kernel.
- A real INT8 kernel uses INT32 accumulation, which is **not bit-identical** to
  the fp32 fake-quant reference; equivalence is "within tolerance", not exact.
- `mlp_attn` is out of scope (still `NotImplementedError`).
- Model family: Llama/Mistral/Qwen2-style Pre-LN (MLP fed by
  `post_attention_layernorm`).
- On models like TinyLlama where INT8 is already near-lossless and GroupRMS is
  gated off, the kernel's value is throughput/memory, not quality.
