# Q-LOT-RMS

**From Quantization Sensitivity to Static FP/INT Layouts for INT8 Transformer Inference.**

A faithful, testable, modular reference implementation, added as a *separate*
feature path. It does **not** import, edit, or remove any existing `PQ*`/SFPA
script. Q-LOT-RMS changes a model only when `enable_qlot_rms=True` **and** the
model is explicitly patched.

---

## 1. What is implemented (and what is reference-only)

### Implemented and tested (`qlot_scope="mlp_only"`)
- **Calibration-time SADND routing** at the Pre-LN `LN2 → FFN` interface
  (`post_attention_layernorm → gate_proj/up_proj`), using pre-affine activations.
  - High-quantile proxy scale (`p_proxy=0.9995`, `qmax=127`), proxy INT8
    reconstruction, relative proxy distortion
    `δ_c = E[(u_c−û_c)²]/(E[u_c²]+ε)`.
  - Cross-subset aggregation `δ̃_c = mean_s(δ_c) + λ·std_s(δ_c)` (`λ=1.0`).
  - Assignment `K_F=⌊ρ_F·C⌋` (`ρ_F=0.06`); **INT = BottomK(δ̃, C−K_F)**,
    **FP = remaining** high-distortion channels.
- **Static permutation** `P = [FP (orig order), INT (orig order)]`. No runtime
  top-k / sort / input-dependent routing (test-enforced).
- **INT-branch GroupRMS** (gate_proj & up_proj only; never FP / down_proj /
  attention). Contiguous groups of `grms=128`; the final group may be smaller.
- **Mean-scale compensation**: scalar `μ_g` (mean RMS over calibration) folded
  into the INT **input-channel weight columns** `W[:, int_indices] *= μ_g` at
  packing (orientation-correct for PyTorch `nn.Linear` `[out, in]`).
- **Frozen activation-scale calibration**:
  `a_c = max(quantile(|y_I,c|, p_act)/qmax, ε_s)`, `p_act=0.999`. One per-layer
  vector, shared by gate_proj & up_proj (both consume the same `y_I`).
- **Static packed FP/INT layout**: split along the input dim; FP slice in FP16,
  INT slice in simulated W8-G128; partial outputs summed in the same output
  space; no inverse permutation.
- **Reference PyTorch inference path** (`backend="torch_reference"`, default):
  correct without any custom kernel.
- **Reversible integration**: `patch_model` / `unpatch_model` restore the
  original `post_attention_layernorm` and `mlp` exactly.

### Reference-only / stubs
- **`backend="custom_packed"`** — a clean stub. Calling it raises
  `NotImplementedError` with a precise CUDA/Triton TODO list (see
  `qlot_rms/projection.py::CustomPackedBackend`). The reference backend is the
  default correctness path; nothing silently falls through.
- **`qlot_scope="mlp_attn"`** — explicit stub. Selecting it raises
  `NotImplementedError`; attention routing is *not* silently ignored.
- The INT matmul is *fake-quantized* (quantize→dequantize→FP32 matmul). This is
  numerically faithful for quality but is **not** a fast integer kernel.

---

## 2. Configuration

`qlot_rms/config.py::QLotRmsConfig` (defaults match the paper):

| field | default | meaning |
|---|---|---|
| `enable_qlot_rms` | `False` | master switch |
| `qlot_scope` | `"mlp_only"` | `"mlp_only"` works; `"mlp_attn"` raises |
| `fp_ratio` | `0.06` | ρ_F, FP channel fraction |
| `grms_group_size` | `128` | INT-branch GroupRMS group size |
| `lambda_agg` | `1.0` | mean + λ·std aggregation |
| `p_proxy` | `0.9995` | proxy INT8 scale quantile |
| `p_act` | `0.999` | activation-scale quantile |
| `qmax` | `127` | INT8 symmetric max |
| `routed_layers` | `"all"` | `"all"` \| list \| `{start,stop,step}` |
| `calibration_samples` | `128` | WikiText-2 chunks |
| `calibration_seq_len` | `512` | tokens per chunk |
| `num_calib_subsets` | `5` | subsets for mean+std |
| `subset_size` | `32` | sequences per subset |
| `backend` | `"torch_reference"` | `"custom_packed"` is a stub |
| `w8_group_size` | `128` | simulated W8-G128 group |
| `use_grms` | `True` | ablation: INT-branch GroupRMS on/off |
| `use_mean_comp` | `True` | ablation: fold μ_g into INT weight columns |
| `act_scale_max_tokens` | `16384` | per-layer token cap for scale estimation |
| `cache_dequant_weight` | `True` | **reference-only** speed flag: fake-quantize the static W8-G128 weight once at packing instead of every forward (numerically identical) |

---

## 3. How to run calibration

```python
from qlot_rms.config import QLotRmsConfig
from qlot_rms.calibration import calibrate
from qlot_rms.model_integration import patch_model, unpatch_model

cfg = QLotRmsConfig(enable_qlot_rms=True, qlot_scope="mlp_only", fp_ratio=0.06)
plan = calibrate(model, tokenizer, cfg, device="cuda:0",
                 routing_method="sadnd")        # "sadnd" | "random" | "magnitude"
plan.save("out_dir")                            # writes .pt + .json + config.json

handle = patch_model(model, plan, cfg)          # enable Q-LOT-RMS
# ... run inference ...
unpatch_model(handle)                           # restore original model exactly
```

Re-load a saved plan with `RoutingPlan.load("out_dir/qlot_rms_routing.pt")`.

---

## 4. How to run evaluation

WikiText-2 perplexity across the controlled variant matrix:

```bash
python eval/eval_perplexity.py \
    --model meta-llama/Llama-2-7b-hf --device cuda:0 \
    --fp_ratio 0.06 --seq_len 2048 --max_chunks 64 \
    --out_dir eval/results_llama2_7b
```

Variants: `fp16`, `int8_ptq` (fp_ratio=0, no GroupRMS), `random`, `magnitude`,
`sadnd`, `sadnd_grms`, `sadnd_grms_meancomp`. Results saved as JSON + CSV.

Optional throughput / memory (batch=1, seq_len=1024) — raw wall-clock only:

```bash
python eval/benchmark.py --model meta-llama/Llama-2-7b-hf --device cuda:0 \
    --seq_len 1024 --variants fp16 sadnd_grms_meancomp
```

Tiny offline sanity (no download, CPU):

```bash
python eval/run_tiny_sanity.py
```

Unit tests:

```bash
python -m pytest tests/ -q
```

---

## 5. Enabling / disabling Q-LOT-RMS

- **Disabled by default.** With `enable_qlot_rms=False` (or simply not calling
  `patch_model`), the model is untouched and behaves exactly as before.
- **Enable**: `cfg.enable_qlot_rms=True`, calibrate, then `patch_model`.
- **Disable at runtime**: `unpatch_model(handle)` restores the original modules.
- Existing PQ/SFPA quantization paths are independent and unaffected.

---

## 6. Observed TinyLlama Behavior

These are the **measured** results in this repository on **TinyLlama-1.1B-Chat-v1.0**
(WikiText-2 test, `seq_len=2048`, `torch_reference` backend, fp_ratio=0.06,
grms_group_size=128, all layers routed). They are reported honestly and are
**not** a reproduction of the paper's headline numbers — see the interpretation
below.

### Perplexity (lower is better)
| variant | PPL | Δ vs FP16 |
|---|---|---|
| FP16 baseline | **8.0267** | — |
| INT8 PTQ (FFN input, no routing/GroupRMS) | 8.0306 | +0.0039 |
| random routing | 8.0306 | +0.0039 |
| magnitude routing | 8.0299 | +0.0032 |
| **SADND routing (routing-only)** | 8.0303 | +0.0036 |
| SADND + GroupRMS (no mean-comp) | 9.7581 | +1.7314 |
| SADND + mean-comp GroupRMS (full) | 9.1376 | +1.1109 |

### Throughput / memory (reference-backend overhead only)
batch=1, `seq_len=1024`, prefill, `torch_reference` (fake-quantized):
| variant | prefill latency | peak GPU mem |
|---|---|---|
| FP16 | 38.2 ms | 2280 MB |
| INT8 PTQ | 138.8 ms | 3452 MB |
| SADND (routing-only) | 148.4 ms | 3451 MB |
| SADND + mean-comp GroupRMS | 169.2 ms | 3452 MB |

> These timings are **reference-backend overhead**, NOT a speedup. The
> `torch_reference` path dequantizes and runs FP matmuls, so it is *slower* and
> uses *more* memory than FP16. **No speedup is claimed.**

### Interpretation (honest)
1. **TinyLlama-1.1B does not reproduce the full Q-LOT-RMS quality gain.**
2. **Plain INT8 PTQ is already near-lossless** here: FP16 8.0267 → INT8 PTQ
   8.0306 (**+0.0039 PPL**). There is essentially no quantization headroom to
   recover.
3. **SADND routing is also near-lossless**: 8.0303 (**+0.0036 PPL** vs FP16),
   i.e. routing-only preserves quality.
4. **GroupRMS introduces a function shift**: SADND + GroupRMS = 9.7581;
   SADND + mean-comp GroupRMS = 9.1376. GroupRMS divides the INT activation by a
   per-(token, group) RMS that **cannot** be statically inverted, so it changes
   the function (it is not function-preserving).
5. **Mean compensation helps substantially** (9.7581 → 9.1376) — confirming the
   per-group μ_g compensation works — **but does not recover FP16 quality** on
   this model, because there was no quantization error for GroupRMS to offset,
   only added per-token noise.
6. **Recommended measured configuration for TinyLlama + torch_reference is
   routing-only** (`use_grms=false`), which is near-lossless; full GroupRMS is
   not recommended here.
7. The paper's speedup requires a **custom packed branched kernel** (FP16 + INT8
   W8-G128 fused). This implementation currently uses **`torch_reference` fake
   quantization**, which is a correctness reference, not a fast path. **No
   backend-independent INT8 acceleration is claimed**, and **paper-level
   reproduction is not claimed.**
8. **Llama-2-7B custom-kernel reproduction remains future work** (see §9 for the
   kernel API contract; larger models with real INT8 quantization error are
   where routing + GroupRMS is expected to help, but that has not been measured
   here).

## 6a. Practical recommendation after TinyLlama evaluation

### Frozen conclusion (TinyLlama-1.1B)

- **Plain INT8 PTQ is already near-lossless on TinyLlama** (FP16 8.0267 → INT8
  PTQ 8.0306, +0.0039 PPL): there is essentially no quantization headroom.
- **SADND routing-only is the best practical TinyLlama setting** (8.0303,
  +0.0036 PPL). Recommended preset:
  `configs/qlot_rms_tinyllama_sadnd_only.json`.
- **GroupRMS is disabled by per-layer gating** on TinyLlama (0/22 layers),
  because the calibration proxy shows it *worsens* INT-branch reconstruction
  error (≈0.018 → 0.166) and PPL (always-on GroupRMS = 9.14–9.76).
- **`torch_reference` is correctness-only and is not expected to beat FP16.**
  It uses fake quantization, dequantization, FP32 matmul, and an extra branch;
  it is ~2.5× slower than FP16 and is near its practical floor. Further
  reference-backend speed tuning has been stopped.
- **No speedup and no paper-level reproduction is claimed.** Real throughput
  requires a `custom_packed` FP16+INT8 branched kernel — see
  `docs/custom_packed_kernel_plan.md` and §9.

### Variant guidance

- **Routing-only (SADND, `use_grms=false`) is the best measured variant for
  TinyLlama on the `torch_reference` backend** — near-lossless (+0.0036 PPL),
  the recommended safe default. Preset:
  `configs/qlot_rms_tinyllama_sadnd_only.json` (or `qlot_rms_tiny_sadnd_only.json`).
- **GroupRMS should be treated as conditional, not always-on.** Use per-layer
  gating (`grms_gating=true`): each routed layer enables GroupRMS only if a
  calibration-time proxy shows it reduces the INT-branch output reconstruction
  error vs plain INT8 (recorded in `grms_enabled` / `grms_gate_reason` /
  `grms_proxy_err_*` per layer). On models where INT8 is already near-lossless,
  gating disables GroupRMS on most/all layers and recovers routing-only quality.
- **`torch_reference` is correctness-only and is not expected to be fast.** It
  is fake-quantized (dequant + FP matmul) and is slower / higher-memory than
  FP16.
- **A `custom_packed` backend is required for any real speedup claim** (see §9).
  No backend-independent INT8 acceleration is claimed.

**Auto-selection & sweeping** (small validation split, PPL only):
```bash
# pick the lowest-PPL variant on a small validation split
python -m eval.select_variant --config configs/qlot_rms_full.json \
    --device cuda:0 --val_chunks 16 --out_dir results/qlot_rms_auto \
    --config_out configs/auto      # writes variant_selection.json + qlot_rms_auto_selected.json

# sweep fp_ratio / grms_group_size / p_act / use_grms / use_mean_comp
python -m eval.sweep_qlot --device cuda:0 --val_chunks 16 \
    --out_dir results/qlot_rms_sweep   # writes sweep_results.{json,csv} + best_config.json

# routing-only recommended default
python -m eval.eval_perplexity --config configs/qlot_rms_tiny_sadnd_only.json \
    --out_dir results/qlot_rms_routing_only
```

## 6b. Q-LOT-DC: Static Diagonal Compensation

GroupRMS divides the INT activation by a per-(token, group) RMS, which is
**token-dependent** and cannot be statically inverted — it introduces a function
shift that hurts quality on models like TinyLlama where INT8 PTQ is already
near-lossless. **Q-LOT-DC** replaces GroupRMS with a **static** correction:

- During calibration, compute one per-INT-channel scale `alpha_c` (no token
  dependence). Modes:
  - `median_scale`: `alpha_c = clamp(median(s)/(s_c+eps), αmin, αmax)` where
    `s_c = quantile(|y_c|, p_act)` — flattens INT activation scales.
  - `smoothquant_like`: `alpha_c = clamp((a_c^β)/((w_c+eps)^(1-β)), αmin, αmax)`
    (with `w_c = max|W_gate[:,c]| + max|W_up[:,c]|`), normalized by its median.
- Apply as a **diagonal similarity transform** that preserves the projection
  function *before* quantization:
  - activation: `y_c -> alpha_c * y_c`
  - weight (nn.Linear `[out, in]`): `W[:, c] -> W[:, c] / alpha_c`  (INT columns only)
  - so `(alpha_c·y_c)·(W[:,c]/alpha_c) == y_c·W[:,c]` exactly in full precision;
    the only change is that the subsequent per-channel INT8 quantization sees
    flatter, more uniform scales.

Because the transform is exact pre-quant and static, Q-LOT-DC has **no
token-dependent normalization** and is safer than GroupRMS for near-lossless
models. It is selected by `method="qlot_dc"`, `use_static_diag_comp=true`,
`use_grms=false` (DC replaces GroupRMS). Preset: `configs/qlot_dc_tinyllama.json`.

Companion features:
- **Error-bounded FP budget** (`fp_budget_mode="error_bounded"`): per-layer
  selection of the smallest `fp_ratio` candidate whose estimated error
  (`activation_mse` = mean relative proxy distortion over INT channels, or
  `output_mse` = INT-branch output reconstruction error) is ≤ `error_bound`;
  otherwise the largest candidate. Selected ratio + per-candidate errors are
  saved per layer (`selected_fp_ratio`, `fp_budget_errors`).
- **Projection bias correction** (`use_projection_bias_correction=true`, off by
  default): static per-output `b = mean_t(z_fp_ref − z_quant)` for gate/up,
  added at inference. Saved as `bias_corr_gate` / `bias_corr_up`.
- **Routing score** (`routing_score`): `sadnd` (default) or `magnitude`;
  `output_aware_sadnd` is a config stub that raises `NotImplementedError`.

Q-LOT-DC is **not a speedup claim**; `torch_reference` remains correctness-only.
On TinyLlama (where INT8 PTQ is already near-lossless) Q-LOT-DC is expected to
match SADND routing-only rather than beat it — it is a *safer* GroupRMS
replacement, not a universal quality gain. See measured results in
`results/qlot_dc_tinyllama_*`.

## 7. Limitations

- **Not primarily memory-saving.** Q-LOT-RMS keeps a small fraction of channels
  in FP16 and stores INT weights plus per-group scales and FP slices. Its goal
  is quality-preserving INT8 *compute layout*, not footprint reduction.
- **GroupRMS is not function-preserving.** It removes per-(token, group) scale
  information; mean-scale compensation only restores the *average* scale.
  For LayerNorm (β≠0) the compensation is approximate (β is not rescaled).
- **Throughput gains require custom packed branched execution.** The default
  reference backend is fake-quantized (dequant + FP matmul) and is for
  correctness, not speed. A real speedup needs the `custom_packed` CUDA/Triton
  kernel (not yet implemented).
- **SADND alone is a quality-preserving routing rule, not a universal speedup
  method.** Routing decides *where* INT8 is safe; it does not by itself make
  inference faster.
- **No backend-independent speedup is claimed.** Speedup may only be reported
  when measured in the *same* backend, with the *same* FP/INT ratio, branch
  shapes, routed layers, and kernel path. `benchmark.py` reports raw wall-clock
  and refuses to compute a speedup ratio for the reference backend.
- **Model family.** End-to-end integration targets Llama-family Pre-LN models
  whose MLP is fed by `post_attention_layernorm` (Llama / Mistral / Qwen2). Other
  norm placements (e.g. Gemma-2's `pre_feedforward_layernorm`) are not wired up.

---

## 8. Reference-backend performance (optimizations)

The `torch_reference` backend has been optimized **without changing numerics**
(locked by `tests/test_perf_equivalence.py`):

- **Cached fake-quantized weight** (`cache_dequant_weight=True`): the static
  W8-G128 weight is fake-quantized once at packing rather than on every forward.
  Bit-identical to the per-forward path (same fake-quant of a static weight).
  Reference-only; the `custom_packed` backend ignores it.
- **Vectorized GroupRMS**: single padded reduction instead of a Python loop over
  groups; verified equal to the loop reference.
- **Precomputed affine slices + device-resident indices**: `gamma_F/gamma_I/
  beta_F/beta_I` and FP/INT index tensors are computed once at packing, so the
  forward does no `index_select` on gamma/beta and no host→device transfers.
- gate_proj & up_proj **share** the branch inputs (`y_F`, `y_I`) computed once.

These are reference-backend-only speedups (the reference is still fake-quantized
and is **not** a fast INT8 path). See §7 limitations.

### Measured outcome (TinyLlama-1.1B, batch=1, seq_len=1024, prefill)

The optimized `torch_reference` backend improves TinyLlama prefill latency by
approximately 2% for INT8 PTQ and SADND routing, while preserving bit-identical
outputs in projection-level equivalence tests. However, the reference backend
remains slower than FP16 because it uses fake quantization, dequantization, FP32
matmul, and an additional branched path. Therefore, these timings should be
interpreted as correctness-reference overhead rather than an INT8 acceleration
result. Real throughput improvement requires a custom packed FP16+INT8 branched
kernel.

| variant | before | after |
|---|---|---|
| fp16 | 37.84 ms | 37.79 ms |
| int8_ptq | 95.57 ms | 93.55 ms |
| sadnd | 95.93 ms | 94.01 ms |

- This is the **`torch_reference` backend only**.
- **No speedup over FP16 is claimed** (the reference is ~2.5× slower than FP16 by design).
- **No paper-level throughput reproduction is claimed.**
- **Real throughput improvement requires a `custom_packed` FP16+INT8 branched kernel** (see §9).

## 9. Custom kernel API contract (for `custom_packed`)

`custom_packed` MUST remain `NotImplementedError` until a real kernel exists.
A future CUDA/Triton branched packed kernel should consume exactly these frozen,
per-layer artifacts (all produced by calibration / packing):

**Inputs per routed projection (gate_proj, up_proj), one decoder layer:**

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `y_F` | `[B, S, K_F]` | fp16 | FP-branch input (`u_F * gamma_F + beta_F`), shared by gate & up |
| `y_I` | `[B, S, C_int]` | fp16/fp32 | INT-branch input after GroupRMS + affine, shared by gate & up |
| `act_scales` | `[C_int]` | fp32 | frozen per-input-channel activation scales `a_c` |
| `W_F` | `[O, K_F]` | fp16 | FP weight slice (input columns = FP channels) |
| `W_I` packed | `[O, C_int]` codes + `[O, n_groups]` scales | int8 + fp16 | W8-G128 weight slice; **mean-comp (per-group μ_g) already folded into the codes/scales at packing** |
| `bias` | `[O]` or None | fp16 | projection bias |

**Static metadata (no runtime routing):**
`K_F` (FP split offset), `C_int`, `w8_group_size=128`, `qmax=127`,
`grms_group_size=128`, per-group sizes (last may be smaller), and `mu_g`
(`[n_groups]`, already folded into `W_I` — pass only if the kernel folds it
itself instead).

**Computation (per output row o):**
```
z[o] = sum_{c in FP}  y_F[c] * W_F[o, c]                      # FP16 partial
     + sum_{c in INT} q(y_I[c], a_c) * q(W_I[o, c], s_w[o, g(c)])   # INT8 W8-G128 partial
     ( INT8 products accumulated in INT32 per group, then scaled )
```
- **Accumulation dtype:** FP32 accumulator for the summed partial outputs; final
  cast to FP16. (The reference uses FP32 matmul accumulation.)
- **Shapes:** `B`=batch, `S`=seq, hidden `C = K_F + C_int`, output `O`
  (= intermediate_size for gate/up). Split is along the input dim; output dim
  `O` unchanged; FP and INT partials summed in the same output space; **no
  inverse permutation** (channels are pre-permuted `[FP, INT]`).
- GroupRMS + affine remain an elementwise pre-pass producing `y_F`, `y_I`
  (or may be fused into the preceding norm epilogue).

**Integration:** implement `fp_matmul` / `int_matmul` (or a single fused
`packed_matmul`) on `CustomPackedBackend` with the same signature as
`TorchReferenceBackend`, then verify against the reference within tolerance
before enabling. See `qlot_rms/projection.py::CustomPackedBackend`.
```
