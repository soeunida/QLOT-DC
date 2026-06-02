# Q-LOT-DC serving layout & kernel plan

Serving-oriented plan for the `custom_packed` backend. **Experimental.** The
default `torch_reference` backend is fake-quantized and correctness-only; it is
**not** a fast path and **no speedup is claimed** here. This document specifies
the static layout the serving kernel consumes; the kernel API is in
`docs/custom_packed_kernel_plan.md`.

## 1. Target inference scope

- **Prefill first**, **batch=1 first** (a 2-D `[tokens, C]` activation). Decode
  and batched serving are later stages.
- Routed projections: **`gate_proj` and `up_proj`** only (the Pre-LN
  `LN2 -> FFN` interface). `down_proj` and attention are untouched.
- Static everything: FP/INT split, permutation, `alpha_c`, activation scales,
  weight scales — all frozen at calibration. **No runtime top-k / sort.**

## 2. FP branch layout

- Channels: the `K_F = floor(fp_ratio * C)` highest-distortion channels (per
  layer; `fp_ratio` may be per-layer under the error-bounded budget).
- Weight `W_F`: `[O, K_F]` FP16, columns = FP input channels (original order).
- Activation `y_F = u_F * gamma_F + beta_F` (pre-affine `u`, LN2 affine applied).
- Compute: FP16 GEMM `z_F = y_F @ W_F^T`, FP32 accumulation.

## 3. INT branch layout

- Channels: the remaining `C_int = C - K_F` channels (original order).
- Activation `y_I = u_I * gamma_I + beta_I`, then (Q-LOT-DC) `y_I *= alpha_c`.
- Per-input-channel INT8 quantization with frozen `act_scales` `a_c`:
  `q = round(clamp(y_I / a_c, -qmax, qmax))`.
- Weight: W8-G128 along the input dim — int8 codes `Wq [O, C_int]` + per-group
  fp16 scales `s_w [O, n_groups]`, `n_groups = ceil(C_int/128)`. The Q-LOT-DC
  inverse scale (`/alpha_c`) and any mean-comp are **folded into the weight at
  pack time**, so the kernel needs no alpha at runtime.
- Compute: per group, INT8×INT8 with **INT32 accumulation**, scaled by `s_w` (and
  `a_c`), summed into the FP32 accumulator.

## 4. Packed weight format

Per routed projection, one contiguous buffer recommended:
`[ W_F (fp16, O×K_F) | Wq (int8, O×C_int) | s_w (fp16, O×n_groups) ]` with a
small header `{K_F, C_int, n_groups, group_size, qmax}`. Pad `C_int` to a
multiple of `group_size`; padded codes = 0. Split offset `K_F` is static.

The exporter (`qlot_rms/serving_export.py`) currently writes the *reference*
form per layer: `W_F`, effective `W_I` (alpha-inversed / mean-comp'd), the
fake-quant `W_I_dq`, indices, scales, `alpha`, and optional bias/bias_corr. A
production packer would additionally emit int8 codes + `s_w`.

## 5. Activation scale format

- `act_scales`: fp32 `[C_int]`, per INT input channel, frozen. Applied in the
  kernel prologue (activation quant). Shared by `gate_proj` and `up_proj` (same
  `y_I`).

## 6. `alpha_c` handling (Q-LOT-DC)

- `alpha_c`: fp32 `[C_int]`, per INT input channel, static.
- Activation side `y_I *= alpha_c` (elementwise pre-pass or kernel prologue).
- Weight side `W[:, c] /= alpha_c` is **folded at pack time** — the kernel sees
  the already-inversed weight and does not handle alpha at runtime. The transform
  is exact pre-quant: `(alpha·y)·(W/alpha) == y·W`.

## 7. Projection bias correction (optional)

- `bias_corr_gate`, `bias_corr_up`: fp32 `[O]`, static, off by default. Added to
  the projection output in the kernel epilogue: `z += bias + bias_corr`.

## 8. Expected tensor shapes (per routed projection, one layer)

| tensor | shape | dtype |
|---|---|---|
| `y_F` | `[T, K_F]` | fp16 |
| `y_I` / `q` | `[T, C_int]` | fp16 / int8 |
| `act_scales` | `[C_int]` | fp32 |
| `alpha` | `[C_int]` | fp32 |
| `W_F` | `[O, K_F]` | fp16 |
| `Wq` / `W_I_dq` | `[O, C_int]` | int8 / fp16 |
| `s_w` | `[O, n_groups]` | fp16 |
| `bias`, `bias_corr` | `[O]` | fp16 / fp32 |
| output `z` | `[T, O]` | fp16 |

`T` = tokens (prefill seq for batch=1), `C = K_F + C_int`, `O` = intermediate_size.

## 9. Memory layout assumptions

- Row-major; weights stored `[O, ·]` (output-major) so each output tile reads
  contiguous input columns. Pad `C_int` to 128 for aligned group loads.
- FP and INT slices contiguous; `K_F` split offset static. No inverse permutation
  after the projection (channels are pre-permuted `[FP, INT]`).
- The reference cache stores a fp32 dequant weight (≈2× the int8 footprint) — a
  packed kernel would store int8 codes + fp16 scales instead, reducing memory.

## 10. Limitations

- Mixed per-channel activation scale × per-group weight scale: apply `a_c` per
  channel inside/around the INT MAC (no single global dequant factor).
- A true INT8 (INT32-accum) kernel is **not bit-identical** to the fp32
  fake-quant reference — equivalence is "within tolerance", with an explicit
  bound.
- `custom_packed` is experimental and **not wired into the full model forward**;
  only the one-projection `packed_forward` prototype is correctness-tested.
- `mlp_attn` out of scope; Llama/Mistral/Qwen2-style Pre-LN only.

## 11. Why `torch_reference` cannot show speedup

`torch_reference` is a *correctness oracle*: it dequantizes the weight and runs
**FP32 GEMMs** for the INT branch, plus a separate FP16 GEMM for the FP branch,
plus elementwise quant — strictly more work than a single FP16 GEMM. Profiling
shows the FP32 GEMM dominates (~50%+). It is therefore slower and higher-memory
than FP16 by construction. A real speedup requires the packed FP16+INT8 branched
kernel above, benchmarked against FP16 under identical fp_ratio / shapes /
routed layers / kernel path. **No speedup is claimed for the reference backend.**
