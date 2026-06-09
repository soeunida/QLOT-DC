# StaticScale

**Calibration-Time FP/INT Routing and Static INT Scale Tuning for Transformer Inference**

StaticScale is a **training-free, calibration-time static policy search** for INT8
Transformer FFN inference. It keeps the FP budget fixed, improves *where* FP
protection is assigned, and tunes the remaining INT branch with **static** group-wise
scale multipliers. Every decision is frozen before serving — inference does **no**
runtime top-k / sort / search / activation normalization.

> The current implementation is a **correctness/reference backend** (`torch_reference`:
> fake-quantized, FP matmul). **No backend-independent speedup is claimed.** Any
> throughput numbers from the packed FP/INT prototype are **prototype diagnostics only**.

## Components

1. **Output-aware SADND routing** — score each input channel by relative INT8 proxy
   distortion weighted by its gate/up weight-column norms; protect the
   highest-scoring channels in FP16, quantize the rest to INT8 (W8-G128).
2. **Cascade-aware & marginal-gain FP budget allocation** — distribute one fixed
   global FP budget across layers by local sensitivity plus accumulated
   residual-stream (cascade) error, or greedily by per-channel marginal gain.
3. **Equal-budget FP mask refinement** — keep each layer's FP count fixed and refine
   *which* channels are FP via greedy boundary swaps, accepted only if a measured
   MLP-output proxy improves.
4. **Static groupwise clip-gain tuning** — tune a per-INT-group activation clip
   multiplier `tau_g` (folds into the frozen activation scales) and a layer/group INT
   output gain `eta` (folds into the INT weight columns). Static metadata only.
5. **Packing-aware static FP/INT layout** — order INT channels so each contiguous
   W8-G128 group has a uniform activation scale; the FP block stays first.

All components are gated by an **equal-budget accept-only** rule: a choice is kept
only if it beats the baseline at the *same* FP budget by a margin; otherwise it
falls back. No method is credited for spending more FP.

## Contribution (what actually drives the result)

The **primary contribution is static groupwise clip-scale tuning** of the INT branch.
Output-aware SADND routing, cascade/marginal FP budget allocation (CAP+), and
equal-budget FP mask refinement are **supporting mechanisms** that keep the clip tuning
stable under a fixed FP budget. In the regime tested, the structural stages give
**diminishing returns** once clip tuning is applied, and **joint mask-scale optimization
did not improve over the additive pipeline** (a negative finding; see
`docs/negative_findings.md`).

## Main result

**Qwen2.5-7B, WikiText-2, seq_len=2048, 64 chunks, fp_ratio=0.20, seeds 0/1/2**
(equal-budget multi-seed check; robust iff ≥2/3 seeds clear −0.001 PPL **and** mean
Δ < −0.001):

| comparison | mean ΔPPL | std | seeds clearing | robust |
|---|---|---|---|---|
| StaticScale vs clean SADND | −0.00293 | 0.00069 | 3/3 | True |
| StaticScale vs CAP+ (cascade+marginal budget) | −0.00108 | 0.00068 | 2/3 | borderline |

- The gain is **small and clip-driven** (clip multiplier `tau ≈ 1.24–1.25`); the output
  gain `eta ≈ 1.0` contributes little, and group-wise `eta` is experimental (rejected by
  accept-only).
- **`CAP+ + clip` is already close to the full pipeline.** Adding equal-budget mask
  refinement or joint mask-scale search yields **no further measurable gain** at
  fp_ratio=0.20 (proxy diagnostics; see `docs/negative_findings.md`). We therefore do
  **not** claim a strong full-pipeline improvement over `CAP+ + clip`.
- The effect is budget-dependent (~0.04% PPL). **No speedup is claimed.**

Per-seed numbers and the summary CSV:
`results/sadnd_cap_gt_multiseed_qwen25_7b/` and `docs/results_summary.md`.

## Install

```bash
pip install -e .
```

## Quickstart

```bash
pip install -e .
python -m pytest tests/ -q
python eval/run_tiny_sanity.py
# validate the full StaticScale pipeline config without any GPU work:
python eval/run_staticscale_smoke.py --dry_run
```

Library usage:

```python
from staticscale import StaticScaleConfig, calibrate, patch_model, unpatch_model

cfg = StaticScaleConfig.load_json("configs/staticscale_qwen25_7b_fp020.json")
plan = calibrate(model, tokenizer, cfg, device="cuda:0")   # static policy search
handle = patch_model(model, plan, cfg)                      # enable StaticScale
# ... run inference ...
unpatch_model(handle)                                       # restore the model exactly
```

## Evaluation

```bash
# equal-budget candidate selection (per-fp accept-only)
python eval/run_staticscale_select.py --config configs/staticscale_select.json \
    --model Qwen/Qwen2.5-7B --device cuda:0 --seq_len 2048 --max_chunks 64 \
    --out_dir results/staticscale_qwen25_7b

# multi-seed equal-budget robustness (clean SADND vs CAP+ vs StaticScale)
python eval/run_staticscale_multiseed.py --model Qwen/Qwen2.5-7B \
    --fp_ratio 0.20 --seeds 0,1,2 --max_chunks 64 \
    --out_dir results/staticscale_multiseed_qwen25_7b

# 7B model-zoo availability (no downloads) + fp-ratio matrix
python eval/check_model_zoo_availability.py --zoo configs/model_zoo_7b.json
python eval/run_7b_matrix.py --config configs/staticscale_7b_matrix.json

# compact summary of a result dir
python eval/summarize_staticscale_results.py --result_dir results/staticscale_qwen25_7b
```

## Documentation

- `docs/method.md` — method and formulas
- `docs/results_summary.md` — measured results
- `docs/reproducibility.md` — how to reproduce
- `docs/negative_findings.md` — what did not work (removed correction methods; joint
  mask-scale search; why the gain over `CAP+ + clip` is small)
- `docs/api.md` — public API
- `docs/figures.md` — figure scripts and assets

## Limitations

- `torch_reference` is **correctness-only** (fake-quant + FP matmul); it is slower and
  higher-memory than FP16 by construction.
- **No backend-independent speedup is claimed**; a real packed FP16+INT8 kernel is not
  implemented (`custom_packed` is a stub). Packed-layout throughput numbers are
  **prototype diagnostics only**.
- Tested in the INT8-near-lossless regime (TinyLlama-1.1B, Qwen2.5-7B), where the
  equal-budget effect is small; a decisive test would need a regime where INT8
  materially degrades (e.g. lower-bit).
- End-to-end integration targets Llama / Mistral / Qwen2-style Pre-LN models whose MLP
  is fed by `post_attention_layernorm`.

## Package layout

`staticscale/` is the public package. The implementation currently lives in a
legacy/internal package and is re-exported under stable StaticScale names; the legacy
import path also works during the transition.
