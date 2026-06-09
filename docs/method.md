# StaticScale — method

StaticScale is a training-free, **calibration-time static policy search** for INT8
Transformer FFN inference. It produces a static FP16 / INT8(W8-G128) layout for the
Pre-LN `LN2 → FFN` interface (gate_proj, up_proj). All decisions are frozen at
calibration; inference does no runtime top-k / sort / search / activation
normalization and applies no correction modules. `down_proj` is left unrouted.

The default backend (`torch_reference`) is fake-quantized and **correctness-only**.
**No backend-independent speedup is claimed**; packed-layout throughput is a
**prototype diagnostic** only.

## Pipeline

Calibration runs on WikiText-2 chunks (default 128 × 512, 5 subsets × 32).

### 1. Pre-affine capture
For each routed layer capture the **pre-affine** RMSNorm activation
`u = x · rsqrt(mean(x²)+ε)` (the FFN input before the elementwise γ/β).

### 2. SADND distortion
Per channel `c`, the relative INT8 proxy distortion
`δ_c = E[(u_c − û_c)²] / (E[u_c²] + ε)`, where `û_c` is `u_c` quantized with a
high-quantile proxy scale (`p_proxy=0.9995`, `qmax=127`), aggregated across subsets
as `mean + std`.

### 3. Output-aware SADND routing (`routing_score`)
- `sadnd`: `δ_c`
- `output_aware_sadnd` (default): `δ_c · (||W_gate[:,c]||₂ + ||W_up[:,c]||₂)` — favors
  channels that are both hard to quantize and heavily used.
- `magnitude`: `E[|u_c|]` (baseline)

The `fp_ratio·C` highest-score channels are protected in FP16; the rest go to INT8.

### 4. Cascade-aware & marginal-gain FP budget (`fp_budget_mode`)
One fixed global FP budget `⌊fp_ratio · Σ_l C_l⌋` is distributed across layers:
- `fixed`: each layer keeps `⌊fp_ratio · C⌋`.
- `global`: globally rank all (layer, channel) pairs by score; total preserved.
- `cascade` / `marginal`: residual streams propagate quantization error across layers,
  so budget is weighted by local sensitivity **and** accumulated cascade error:

  ```
  e_l            = ||h_l^q − h_l^fp|| / (||h_l^fp|| + ε)
  cascade_l      = beta · cascade_{l-1} + e_l
  budget_score_l = local_sensitivity_l + gamma · cascade_l
  ```
  The budget is allocated by `budget_score` (largest remainder, capped) or greedily
  by per-channel marginal gain. The **total is always preserved**.

### 5. Equal-budget FP mask refinement (`use_fp_mask_refinement`)
Keep each layer's FP count `k_fp` fixed and refine *which* channels are FP. The
routing score is a cheap heuristic; near the FP/INT boundary it can disagree with the
actual MLP-output error. Greedy boundary swaps (a low-score FP channel out, a high-score
INT channel in) are accepted only if a measured MLP-output proxy improves by
`fp_refine_margin`. Each swap preserves `k_fp` exactly; the packing-aware permutation
is rebuilt afterwards.

### 6. Static groupwise clip-gain tuning (`use_groupwise_clip_gain_tuning`)
Tune the INT branch statically (a dynamic runtime activation-normalization variant was
tried and removed — see `docs/negative_findings.md`):

1. **Group-wise clip multiplier** `tau_g` on the INT8 activation scale:
   ```
   s_g' = tau_g · s_g
   ```
   chosen from `gt_clip_candidates` to minimize the weight-normalized INT activation
   quantization error of group `g`. `tau` folds into the frozen `act_scales`.
2. **INT output gain** `eta` so the quantized INT output matches the exact FP
   INT-branch output (least squares, clamped to `[gt_gain_clip_min, gt_gain_clip_max]`):
   ```
   z = z_FP + eta · z_INT ,   minimize ||z_FP16 − z||²
   eta = ⟨target, z_INT⟩ / ⟨z_INT, z_INT⟩ ,  target = z_FP16 − z_FP = y_I @ W_I^T
   ```
   `eta` is fit per projection (gate, up); layer-wise `eta` is one scalar, group-wise
   `eta` is an independent per-group fit (experimental). `eta` folds into the INT
   weight columns at patch time.

Per-layer accept-only: GT is enabled on a layer only if its proxy (`gt_metric`)
improves by a relative `gt_accept_margin`; otherwise that layer falls back.

### 7. Packing-aware static FP/INT layout (`int_permutation_mode`)
INT channels are reordered so each contiguous W8-G128 group has a uniform per-channel
activation scale (`scale_sorted` / `scale_clustered` / `packing_aware`). The FP block
stays first; `perm = [FP (orig order), INT (packed order)]`; no inverse permutation at
inference.

### 8. Equal-budget accept-only selection
`eval/run_staticscale_select.py` compares candidates at the same `fp_ratio` and accepts
one only if it beats clean SADND by `accept_only_margin`; a refined / clip-gain
candidate must also beat its non-tuned counterpart. Otherwise it falls back. Failed
candidates are recorded, never fabricated. No method is credited for a larger FP budget.

## Inference (reference path)
Per routed layer: compute `u`; split by the frozen permutation into FP/INT;
`y_F = u_F·γ_F + β_F` (FP16), `y_I = u_I·γ_I + β_I` then per-channel INT8 quant with the
(clip-tuned) `act_scales`; `z = y_F @ W_F^T + quant(y_I) @ W_I(W8-G128)^T + bias`, where
the INT weight columns already carry the static output gain. FP32 accumulation → FP16.
gate_proj and up_proj share the branch inputs. There are **no** correction modules and
**no** runtime search.

**Calibration approximation.** The `eta` least-squares fit and the GT accept-only proxy
use the FP INT-branch weights during calibration, while final selection and inference use
the actual quantized W8-G128 path. In our current results this approximation has
negligible effect because `eta` remains close to 1.0, and accept-only selection is
ultimately checked through the real quantized path.

## Measured (Qwen2.5-7B, multi-seed; fp_ratio=0.20, 64 chunks, seeds 0/1/2)
The improvement is **small and clip-driven** — static clip-scale tuning is the dominant
driver; the structural stages are supporting mechanisms.
- **StaticScale vs clean SADND:** mean Δ = −0.00293 PPL, std = 0.00069, 3/3 seeds clear
  the −0.001 margin.
- **StaticScale vs CAP+ (cascade+marginal budget):** mean Δ = −0.00108 PPL, 2/3 seeds
  clear — small and **borderline**.
- `tau` mean ≈ 1.24–1.25; output gain `eta ≈ 1.0` contributes little; group-wise `eta`
  is experimental and rejected by accept-only.

**We do not claim a strong full-pipeline improvement over `CAP+ + clip`.** Once clip
tuning is applied, the structural stages (mask refinement, joint coupling) give
diminishing returns: `CAP+ + clip` is already close to the full pipeline, and a joint
mask-scale search found no additional gain (see *Why is the additional gain small?* and
`docs/negative_findings.md`). **No speedup is claimed.** See `docs/results_summary.md`
and `results/sadnd_cap_gt_multiseed_qwen25_7b/`.

## Why is the additional gain over `CAP+ + clip` small?
Our diagnostics show that, under fp_ratio=0.20, most recoverable error is **scale-driven**.
Clean SADND and CAP+ already protect the highest-risk channels, leaving only marginal
boundary channels for equal-budget mask refinement. Consequently, the structural stages
provide diminishing returns once static clip-scale tuning is applied. We further tested
**joint mask-scale search**, where candidate FP/INT swaps are followed by local retuning
of INT group scales. This search did **not** improve the calibration proxy over the
additive `CAP+ + mask refinement + clip-tuning` pipeline, suggesting that the residual
error in this setting is **not** primarily caused by a mismatch between the refined mask
and the tuned INT scales. We therefore interpret StaticScale primarily as a **static
clip-scale calibration method** for FP/INT-routed activations, with routing and budget
allocation serving as supporting mechanisms that make the clip tuning stable under an
equal FP budget.

Contributing factors:
- SADND already protects the most fragile channels.
- At fp_ratio=0.20, the FP budget is close to saturation for high-risk channels.
- Most recoverable error is scale-driven (recovered by clip tuning, not by mask choice).
- Mask refinement operates on boundary channels with small marginal differences.
- Calibration-proxy improvements do not always translate to PPL gains.
- Joint retune/swap search found no additional proxy gain over the additive pipeline.
