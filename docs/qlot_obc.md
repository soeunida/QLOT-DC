# Q-LOT-OBC : Output-aware Block Correction

## Why this revision

On Qwen2.5-7B, Q-LOT-DC+ did **not** beat SADND at equal FP budget (within
±0.0007 PPL; the apparent gain over INT8 was just the FP budget). The reason:
**per-projection correction is too weak.** Q-LOT-DC and projection bias/low-rank
correct `gate_proj` and `up_proj` *before* the FFN nonlinearity. But the MLP block
output is

```
y = down_proj( SiLU(gate(n)) * up(n) )
```

so an error fixed at the gate/up outputs is then passed through **SiLU and the
gate∗up elementwise product** — a nonlinearity that the linear pre-correction
cannot account for. Correcting the **final block output** `y` directly is much
more closely tied to the model's actual error.

## Method

For each routed layer, on calibration data:

```
h    = MLP block input (pre-LN2 residual hidden, == what QLotRmsFFN sees)
y_fp = FP16 MLP block output         (original mlp on LN2(h))
y_q  = quantized routed MLP output   (QLotRmsFFN(h), before block correction)
E    = y_fp - y_q
```

Fit one **static** correction (chosen by `block_correction_mode`):

- **bias**: `y_corr = y_q + b`, `b = mean_t(E)`  (per hidden channel)
- **affine**: `y_corr = a·y_q + b`, per-channel least squares, `a` clamped to
  `[block_affine_a_min, block_affine_a_max]` (default [0.5, 2.0])
- **lowrank**: `y_corr = y_q + h @ A @ B`, `E ≈ h·A·B` via least-squares +
  truncated SVD, rank `block_lowrank_rank` (default 4)

Applied at inference **after `down_proj`** (elementwise / one small matmul). Shape
and dtype unchanged; no runtime top-k/sort; fully static.

## Accept-only gating (per layer)

A layer enables its correction **only if it lowers the MLP-output MSE**:

```
after_mse < before_mse · (1 − block_correction_margin)      # default margin 5e-4
```

Otherwise the correction is rejected for that layer (no-op). This prevents a
harmful correction from being forced anywhere. Per-layer
`block_mse_before/after`, `block_corr_enabled`, and the reason are saved.

## Equal-budget selection (the key rule)

`eval/select_qlot_obc.py` compares every candidate **at the same `fp_ratio`** and
accepts an improvement only if it beats **SADND at that same `fp_ratio`** by a
margin (default 0.001):

```
selected_ppl < sadnd_same_budget_ppl − margin
```

**No method is selected merely for using a larger FP budget.** Candidates:
`fp16`, `int8_ptq`, `sadnd`, `sadnd_block_{bias,affine,lowrank_r2,r4}`,
`qlot_dc_biascorr`, `qlot_obc_{bias,affine,lowrank_r2,r4}`, swept over
`fp_ratio ∈ {0.04, 0.06, 0.10, 0.20}`.

## How to run

```bash
# TinyLlama small validation
python eval/select_qlot_obc.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --device cuda:0 --seq_len 1024 --max_chunks 32 \
    --config configs/qlot_obc_select.json --out_dir results/qlot_obc_select_tinyllama

# Qwen2.5-7B small validation
python eval/select_qlot_obc.py --model Qwen/Qwen2.5-7B \
    --device cuda:0 --seq_len 2048 --max_chunks 64 \
    --config configs/qlot_obc_select.json --out_dir results/qlot_obc_select_qwen25_7b

# full test ONLY if a candidate beat SADND at equal budget by margin
python eval/eval_perplexity.py --model Qwen/Qwen2.5-7B --device cuda:0 --seq_len 2048 \
    --config results/qlot_obc_select_qwen25_7b/selected_config.json \
    --out_dir results/qlot_obc_full_qwen25_7b
```

## Honesty constraints

- No speedup claim; `torch_reference` is fake-quantized / correctness-only;
  `custom_packed` remains experimental.
- A correction is accepted only if it measurably helps (per-layer MSE gate +
  equal-budget PPL margin). Harmful corrections are disabled.
- No method is credited for a larger FP budget — comparisons are equal-budget.
