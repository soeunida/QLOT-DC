# Q-LOT-DC

**Q-LOT-DC = SADND routing + Static Diagonal Compensation + Error-bounded FP
budget + (optional) projection bias correction** — a reference implementation
for INT8 transformer FFN inference, derived from the Q-LOT-RMS line of work.

This is a faithful, modular, **flag-guarded** PyTorch reference. The default
backend (`torch_reference`) is fake-quantized and **correctness-only**; it is
not a fast INT8 path. No speedup and no paper-level reproduction is claimed.

## Why Q-LOT-DC (vs GroupRMS)

GroupRMS divides the INT activation by a per-(token, group) RMS — a
**token-dependent** normalization that cannot be statically inverted and that
introduces a function shift. On TinyLlama (where INT8 PTQ is already
near-lossless) GroupRMS *worsens* quality (per-layer gating disables it 0/22).

**Static Diagonal Compensation (Q-LOT-DC)** replaces it with a static,
calibration-time, per-INT-channel scale `alpha_c` applied as a diagonal
similarity transform that preserves the projection function *before*
quantization:

- activation: `y_c -> alpha_c * y_c`
- weight (PyTorch `nn.Linear` is `[out, in]`): `W[:, c] -> W[:, c] / alpha_c` (INT columns only)
- so `(alpha_c·y_c)·(W[:,c]/alpha_c) == y_c·W[:,c]` exactly in full precision; only
  the subsequent per-channel INT8 quantization sees flatter, more uniform scales.

Companion features: an **error-bounded FP budget** (per-layer `fp_ratio` chosen
by an activation/output MSE bound) and an optional **projection bias correction**
(static per-output `mean(z_fp − z_quant)`, off by default).

## Method components

- **SADND routing** — relative INT8 proxy distortion per channel, aggregated
  `mean + λ·std` over calibration subsets; highest-distortion channels go to a
  small FP16 branch, the rest to INT8 (W8-G128). Static `[FP, INT]` permutation;
  no runtime top-k / sort.
- **Static Diagonal Compensation** — `diag_comp_mode ∈ {median_scale, smoothquant_like}`.
- **Error-bounded FP budget** — `fp_budget_mode ∈ {fixed, error_bounded}`.
- **Projection bias correction** — `use_projection_bias_correction` (default off).
- **Backends** — `torch_reference` (default, correctness-only) and a
  `custom_packed` **stub** (`NotImplementedError`) with a documented kernel API
  (`docs/custom_packed_kernel_plan.md`). **Real throughput requires the custom
  packed FP16+INT8 branched kernel.**

`mlp_only` scope is implemented; `mlp_attn` is an explicit `NotImplementedError`
stub. See `docs/qlot_rms.md` for full details and limitations.

## Install

```bash
pip install -r requirements.txt
```

## Run tests

```bash
python -m pytest tests/ -q
python eval/run_tiny_sanity.py        # offline, CPU, no download
```

## TinyLlama small validation (Q-LOT-DC)

```bash
HF_HUB_OFFLINE=1 python eval/eval_perplexity.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda:0 \
    --seq_len 1024 --max_chunks 32 \
    --config configs/qlot_dc_tinyllama.json \
    --out_dir results/qlot_dc_tinyllama_smallval
```

This evaluates `fp16`, `int8_ptq`, `sadnd` (routing-only), and the Q-LOT-DC
variants (`qlot_dc_median`, `qlot_dc_error_bounded`, `qlot_dc_biascorr`) and
writes `ppl_results.json/.csv` + `metadata_summary.json`. Full test: drop
`--max_chunks` and use `--seq_len 2048`.

## Scope & honesty

- `torch_reference` is fake-quantized (dequant + FP32 matmul + extra branch) and
  is **slower than FP16** by design — correctness-only.
- **No INT8 speedup is claimed.** Real throughput requires a custom packed
  FP16+INT8 branched kernel (see `docs/custom_packed_kernel_plan.md`).
- **No paper-level reproduction is claimed.** On TinyLlama, INT8 PTQ and SADND
  routing are already near-lossless; Q-LOT-DC is a *safer* GroupRMS replacement
  that preserves (and marginally improves) quality, not a universal gain.
