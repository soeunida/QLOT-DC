# SADND-CAP: Calibration-Time Adaptive FP/INT Routing and Packing for Low-Bit LLM Inference

SADND-CAP is a calibration-time method that lays out the Pre-LN `LN2 → FFN`
interface of a transformer as a **static FP16 / INT8 (W8-G128)** mixture. It is
the single active method in this repository; earlier correction attempts
(Q-LOT-DC, Q-LOT-DC+, Q-LOT-OBC, GroupRMS, diagonal/bias/low-rank/block
corrections) were evaluated, did **not** robustly beat SADND at equal FP budget,
and have been removed from the active code (recoverable via git tag
`backup-before-final-sadnd-cap-cleanup`).

> The default backend (`torch_reference`) is fake-quantized and **correctness-only**.
> **No speedup is claimed**; a real custom kernel is not implemented.

## 1. What is SADND-CAP?

For each routed MLP layer, SADND-CAP decides — once, at calibration time —
which input channels stay FP16 and which become INT8, how the FP budget is
shared across layers, and how INT channels are ordered for W8-G128 packing.
Everything is frozen; inference does **no** runtime top-k / sort / dynamic
routing.

## 2. Method overview

1. **Output-aware SADND routing** — score channels by quantization sensitivity.
2. **Global layer-wise FP budget allocation** — share one FP budget across layers.
3. **Packing-aware INT permutation** — order INT channels for uniform W8-G128 groups.
4. **Equal-budget accept-only selection** — keep a choice only if it beats SADND
   at the *same* FP budget.

## 3. Output-aware SADND routing

`score_c = δ_c · (||W_gate[:,c]||₂ + ||W_up[:,c]||₂)` where `δ_c` is the relative
INT8 proxy distortion of channel `c` (`E[(u−û)²]/E[u²]`, aggregated mean+std over
calibration subsets) and the weight column norms (nn.Linear `[out,in]`) capture
how heavily the channel is used. The `fp_ratio·C` highest-score channels go to
FP16; the rest to INT8. `routing_score ∈ {sadnd, output_aware_sadnd, magnitude}`.

## 4. Global layer-wise FP budget allocation

With `fp_budget_mode="global"`, the total FP budget `⌊fp_ratio · Σ_l C_l⌋` is
allocated by globally ranking all (layer, channel) pairs by score: a layer with
more high-distortion channels receives more FP capacity, while the **total**
budget is preserved (`same_global_fp_budget`). `"fixed"` uses `fp_ratio` per layer.

## 5. Packing-aware INT permutation

INT channels are reordered so that each contiguous W8-G128 group has a more
uniform per-channel activation scale, reducing within-group quantization error.
Modes: `original`, `scale_sorted`, `scale_clustered`, `packing_aware` (default).
Only INT channels are reordered; the FP block stays first; no inverse permutation
is needed at inference.

## 6. Equal-budget accept-only selection

`eval/select_sadnd_cap.py` compares candidates at the **same `fp_ratio`** and
accepts one only if it beats SADND at that budget by `accept_only_margin`
(default 0.001); otherwise it falls back to the best SADND baseline and records
`clear_improvement=false`. No method is credited for using a larger FP budget.

## 6a. SADND-CAP+ (optional): cascade-aware & marginal-gain FP budget

Transformer residual streams propagate quantization error across layers, so FP
budget can be allocated by **accumulated cascade error**, not just local
sensitivity:

```
e_l = ||h_l^q - h_l^fp|| / ||h_l^fp||      cascade_l = beta·cascade_{l-1} + e_l
budget_score_l = local_sensitivity_l + gamma·cascade_l
```

`use_cascade_aware_budget` allocates the (same total) FP budget by `budget_score`;
`use_marginal_gain_allocation` greedily spends FP where it removes the most error.
Both are **off by default** and gated by the same equal-budget accept-only rule.
Configs: `configs/sadnd_cap_cascade_select.json`, `sadnd_cap_cascade_fp0{06,10,20}.json`.
These are policy/layout choices (no correction modules); on the near-lossless
models tested, SADND-CAP's equal-budget effect is below the robustness margin
(see §10), and SADND-CAP+ is not assumed to clear it — it must be verified per
model. **No speedup is claimed.**

## 7. Quick start

```python
from qlot_rms.config import QLotRmsConfig
from qlot_rms.calibration import calibrate
from qlot_rms.model_integration import patch_model, unpatch_model

cfg = QLotRmsConfig.load_json("configs/sadnd_cap_fp006.json")
plan = calibrate(model, tokenizer, cfg, device="cuda:0")
handle = patch_model(model, plan, cfg)     # enable SADND-CAP
# ... run inference ...
unpatch_model(handle)                      # restore original model exactly
```

## 8. Tests

```bash
python -m pytest tests/ -q
python eval/run_tiny_sanity.py             # offline, CPU, no download
```

## 9. Evaluation

```bash
# equal-budget selection on a small validation split
python eval/select_sadnd_cap.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --device cuda:0 --seq_len 1024 --max_chunks 32 \
    --config configs/sadnd_cap_select.json --out_dir results/sadnd_cap_select

# full WikiText-2 test (fp16 / int8_ptq / sadnd / output_aware / config)
python eval/eval_perplexity.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --device cuda:0 --seq_len 2048 \
    --config configs/sadnd_cap_fp006.json --out_dir results/sadnd_cap_full
```

## 10. Results summary

See `docs/results_summary.md`. In the INT8-near-lossless regime tested
(TinyLlama-1.1B, Qwen2.5-7B), INT8 PTQ and SADND are already near-lossless, and
the prior *correction* methods (DC/OBC) did not robustly beat SADND at equal FP
budget.

- At **fp_ratio=0.06**, a multi-seed equal-budget check on Qwen2.5-7B shows base
  SADND-CAP is consistently but only slightly better than clean SADND (mean
  Δ = −0.0006 PPL, **0/3** seeds clear the 0.001 margin, `robust_better = False`)
  — a weak, sub-margin trend.
- At **fp_ratio=0.20**, a multi-seed check (seeds 0/1/2, 64 chunks) shows
  **SADND-CAP+ (cascade-aware + marginal-gain FP budget) robustly beats clean
  SADND at equal FP budget: mean Δ = −0.00185 PPL, std 0.00019, 3/3 seeds clear
  the 0.001 margin, `robust_better = True`** (base SADND-CAP: mean Δ = −0.00129,
  2/3 clearing). This is the **first method here to satisfy the multi-seed
  equal-budget robustness criterion**.

The improvement is **small but consistent** (~0.0018 PPL on a 6.58 baseline,
~0.03%) and **budget-dependent** — robustness is shown at fp_ratio=0.20, not yet
at fp_ratio=0.06. SADND-CAP remains a **policy/layout framework** (routing +
budget + packing under an equal-budget accept-only rule); SADND-CAP+'s
cascade/marginal budget is a useful extension, **not a large quality win**.
**No speedup is claimed.**

## 11. Limitations

- `torch_reference` is **correctness-only** (fake quant + FP matmul); slower and
  higher-memory than FP16 by construction.
- **No speedup claim**; a custom packed FP16+INT8 kernel is **not** implemented
  (`custom_packed` is a stub; see `docs/serving_layout.md`).
- Deprecated DC/OBC/GroupRMS/correction methods were removed from the active
  code (git tag `backup-before-final-sadnd-cap-cleanup`).
- End-to-end integration targets Llama/Mistral/Qwen2-style Pre-LN models whose
  MLP is fed by `post_attention_layernorm`.
