# SADND-CAP — method

SADND-CAP = **S**ensitivity-**A**ware **D**istortion-**N**ormalized **D**ecision +
**C**alibration-time **A**daptive **P**acking. It produces a static FP16/INT8
layout for the Pre-LN `LN2 → FFN` interface (gate_proj, up_proj) of a transformer
MLP. All decisions are frozen at calibration; inference does no runtime
top-k/sort/dynamic routing and applies no correction modules.

## Pipeline

Calibration runs on WikiText-2 chunks (default 128 × 512, 5 subsets × 32).

### 1. Pre-affine capture
For each routed layer we capture the **pre-affine** LN2/RMSNorm activation
`u = x · rsqrt(mean(x²)+ε)` (the FFN input before the elementwise γ/β).

### 2. SADND distortion
Per channel `c`, the relative INT8 proxy distortion
`δ_c = E[(u_c − û_c)²] / (E[u_c²] + ε)` where `û_c` is `u_c` quantized with a
high-quantile proxy scale (`p_proxy=0.9995`, `qmax=127`). Aggregated across
subsets as `mean + std`.

### 3. Routing score (`routing_score`)
- `sadnd`: `δ_c`
- `output_aware_sadnd`: `δ_c · (||W_gate[:,c]||₂ + ||W_up[:,c]||₂)` — favors
  channels that are both hard to quantize and heavily used.
- `magnitude`: `E[|u_c|]` (baseline)

### 4. FP/INT selection + FP budget (`fp_budget_mode`)
- `fixed`: each layer keeps `⌊fp_ratio · C⌋` highest-score channels in FP16.
- `global`: one budget `⌊fp_ratio · Σ_l C_l⌋` is allocated by globally ranking
  all (layer, channel) pairs by score; the **total** is preserved
  (`same_global_fp_budget`), but layers with more high-distortion channels get
  more FP. INT = the remaining channels.

### 5. Packing-aware INT permutation (`int_permutation_mode`)
INT channels are reordered so each contiguous W8-G128 group has a more uniform
per-channel activation scale (`scale_sorted` / `scale_clustered` /
`packing_aware`), lowering within-group quantization error. The FP block stays
first; `perm = [FP (orig order), INT (packed order)]`; no inverse permutation at
inference.

### 6. Activation scales
After the permutation, per-INT-channel symmetric INT8 scales
`a_c = max(quantile(|y_I,c|, p_act)/qmax, ε_s)` are frozen (`p_act=0.999`).

### 7. Equal-budget accept-only selection
`eval/select_sadnd_cap.py` compares candidates at the same `fp_ratio` and accepts
one only if it beats SADND@same-fp by `accept_only_margin`; else falls back to
SADND (`clear_improvement=false`). No method is credited for a larger FP budget.

## Inference (reference path)
Per routed layer: compute `u`; split by the frozen permutation into FP/INT;
`y_F = u_F·γ_F + β_F` (FP16), `y_I = u_I·γ_I + β_I` then per-channel INT8 quant;
`z = y_F @ W_F^T + quant(y_I) @ W_I(W8-G128)^T + bias`, FP32 accumulation → FP16.
gate_proj and up_proj share the branch inputs. `down_proj` is unrouted. There are
**no** correction modules.

## What was removed (negative findings)
Q-LOT-DC (static diagonal compensation), Q-LOT-DC+ (output-aware + adaptive FP +
bias/low-rank), and Q-LOT-OBC (block-output bias/affine/low-rank correction) were
implemented and evaluated. At equal FP budget they did not robustly beat SADND in
the INT8-near-lossless regime; block low-rank overfit badly. They were removed
from the active code (git tag `backup-before-final-sadnd-cap-cleanup`).

No speedup is claimed; `torch_reference` is correctness-only.
