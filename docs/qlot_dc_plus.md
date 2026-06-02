# Q-LOT-DC+ : validation-PPL-guided Q-LOT-DC

## Motivation

On TinyLlama, plain INT8 PTQ and SADND routing are already near-lossless, and
**Q-LOT-DC alone only *ties* them** (full-test PPL within ~0.001 of INT8/SADND).
GroupRMS is worse (function shift). To actually *beat* INT8/SADND we need (a)
stronger static corrections and (b) **direct validation-PPL selection** — pick
the variant that measurably lowers held-out PPL, rather than trusting a proxy.

**Q-LOT-DC+** = Q-LOT-DC + output-aware routing + adaptive FP budget + stronger
projection bias correction + optional low-rank residual correction, with the
final method chosen by validation PPL.

No speedup is claimed (`torch_reference` is correctness-only). Selection uses a
small validation split; the final test set is evaluated **once**, never tuned on.

## Why validation-PPL selection

Calibration proxies (proxy distortion, reconstruction MSE) correlate with PPL
but do not guarantee a PPL win in a near-lossless regime where differences are
~1e-3. `eval/select_qlot_dc_plus.py` therefore evaluates each candidate's actual
PPL on a small validation split and selects the best, with an explicit decision
rule:

```
threshold = min(ppl_int8_ptq, ppl_sadnd) - margin     # margin default 1e-3
winner    = argmin over DC+ candidates of val PPL
clear_improvement = (ppl[winner] < threshold)
```

If no candidate clears the margin, the best candidate is still selected but
flagged `clear_improvement=false` — we do **not** pretend a tie is a win.

## Components

- **Output-aware SADND routing** (`routing_score="output_aware_sadnd"`):
  `score_c = sadnd_delta_c · (||W_gate[:,c]||₂ + ||W_up[:,c]||₂)` (nn.Linear is
  `[out,in]`, so column norms). Channels that are both hard-to-quantize *and*
  heavily used by the projection are prioritized for the FP branch. Still static
  — no runtime top-k/sort.
- **Adaptive FP budget** (`fp_budget_mode="validation_search"`): the selection
  script sweeps `fp_ratio_candidates` (up to 0.20 — larger FP is allowed because
  the goal is PPL, not speed) and keeps the lowest-PPL ratio per candidate. The
  chosen ratio and `k_fp` are recorded.
- **Projection bias correction** (`use_projection_bias_correction`,
  `bias_corr_scope`): static per-output `b = mean_t(z_fp − z_quant)` for
  `gate_proj`/`up_proj` (`bias_corr_scope="gate_up"`, the DC+ default). `"none"`
  disables; `"ffn_input"` is reserved (SwiGLU has no gate+up sum, so it currently
  aliases `gate_up`).
- **Low-rank residual correction** (`use_lowrank_correction`, `lowrank_rank`):
  fit `X@A@B ≈ (z_fp − z_quant − b)` per projection via least-squares + truncated
  SVD (no gradient training); at inference `z += (y_I@A)@B`. Optional, **off by
  default** (extra small matmul per projection).

## How to run

```bash
# 1) selection on a small validation split (writes selected_config.json)
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python eval/select_qlot_dc_plus.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda:0 \
    --seq_len 1024 --max_chunks 32 \
    --config configs/qlot_dc_plus_select.json \
    --out_dir results/qlot_dc_plus_select_tinyllama

# 2) full test ONCE for fp16 / int8_ptq / sadnd / selected DC+ config
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python eval/eval_perplexity.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda:0 --seq_len 2048 \
    --config results/qlot_dc_plus_select_tinyllama/selected_config.json \
    --out_dir results/qlot_dc_plus_full_tinyllama
```

## Honesty constraints

- No fabricated results; no tuning on the final test set (small val for selection,
  full test once).
- No speedup claim; `torch_reference` is correctness-only; `custom_packed`
  remains experimental.
- Q-LOT-DC+ is accepted as a *clear improvement* only if it beats both INT8-PTQ
  and SADND on held-out validation by the margin; otherwise it is reported as a
  tie.
