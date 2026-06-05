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

## SADND-CAP+: Cascade-aware FP Budget Allocation

Standard layerwise allocation treats layers independently. But Transformer
**residual streams propagate quantization error across layers**, so FP budget
should also consider downstream error propagation. SADND-CAP+ adds two optional
budget policies (both off by default; they only re-allocate the SAME total FP
budget — `same_global_fp_budget`):

- **Cascade-aware budget** (`use_cascade_aware_budget`): measure each layer's
  relative residual-stream quantization error of a baseline (fixed-fp) plan vs
  FP16, accumulate it, and weight the budget toward layers that are both locally
  sensitive and contribute to downstream error:

  ```
  e_l            = ||h_l^q - h_l^fp|| / ||h_l^fp||        (cascade_metric: hidden_l2)
  cascade_l      = beta * cascade_{l-1} + e_l             (beta = cascade_beta)
  budget_score_l = local_sensitivity_l + gamma * cascade_l   (+ amp_weight * amp_l)
  ```
  The global FP budget is allocated across layers by `budget_score` (largest
  remainder, capped per layer), preserving the total.

- **Marginal-gain allocation** (`use_marginal_gain_allocation`): greedily spend
  each FP increment where it removes the most quantization error. The proxy uses
  per-channel SADND distortion (optionally scaled by the layer's cascade score),
  i.e. it gives FP to the globally highest-distortion channels; with cascade
  weighting, high-cascade layers are prioritized. A `marginal_gain_table` records
  per-layer per-candidate gains.

These are **policy/layout** choices, not correction modules. The equal-budget
accept-only rule is unchanged: a policy is accepted only if it beats clean SADND
at the same FP budget by `accept_only_margin`. No speedup is claimed.

### Summary

SADND-CAP+ = SADND-CAP + cascade-aware FP budget allocation + marginal-gain FP
allocation. The motivation:

- Clean SADND treats layers **mostly independently** (per-layer or single-ranking
  FP/INT selection).
- SADND-CAP+ estimates each layer's **local quantization error** and the
  **cross-layer cascade error** it propagates through the residual stream.
- It **reallocates the same global FP budget** toward layers that contribute more
  to accumulated residual-stream error (no extra FP is spent — `same_global_fp_budget`).
- **Marginal-gain allocation** spends FP channels where they reduce error most
  (greedy global, optionally cascade-weighted).

**Measured (Qwen2.5-7B, multi-seed):**
- At **fp_ratio=0.06**, base SADND-CAP shows only a weak, sub-margin trend over
  clean SADND (mean Δ ≈ −0.0006 PPL, 0/3 seeds clear the 0.001 margin).
- At **fp_ratio=0.20** (seeds 0/1/2, 64 chunks), **SADND-CAP+ robustly clears the
  equal-budget margin: mean Δ = −0.00185 PPL, std 0.00019, 3/3 seeds clear**
  (`robust_better = True`); base SADND-CAP: mean Δ = −0.00129, 2/3 clearing. This
  is the first method here to satisfy the multi-seed robustness criterion.

The improvement is **small but consistent** (~0.03% PPL) and **budget-dependent**:
robustness is shown at fp_ratio=0.20, not yet at fp_ratio=0.06. Whether it clears
the margin must still be verified per model under the same equal-budget
accept-only rule — it is not assumed to. **No speedup is claimed** because
`torch_reference` is correctness-only. See `docs/results_summary.md` for the table.

## What was removed (negative findings)
Q-LOT-DC (static diagonal compensation), Q-LOT-DC+ (output-aware + adaptive FP +
bias/low-rank), and Q-LOT-OBC (block-output bias/affine/low-rank correction) were
implemented and evaluated. At equal FP budget they did not robustly beat SADND in
the INT8-near-lossless regime; block low-rank overfit badly. They were removed
from the active code (git tag `backup-before-final-sadnd-cap-cleanup`).

No speedup is claimed; `torch_reference` is correctness-only.
