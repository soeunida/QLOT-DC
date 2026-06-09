# Reproducibility

All results use the `torch_reference` backend (fake-quantized, correctness-only). No
backend-independent speedup is claimed. Calibration and evaluation are deterministic
given a fixed seed and model.

## Environment

```bash
pip install -e .            # installs staticscale + core deps (torch, transformers, datasets, numpy, tqdm)
pip install -e ".[dev]"     # + pytest
```

Tested with Python ≥ 3.10. Models are loaded from the local Hugging Face cache; set
`HF_HUB_OFFLINE=1` to forbid downloads. Choose a GPU with `CUDA_VISIBLE_DEVICES`.

## Sanity (offline, CPU, no download)

```bash
python -m pytest tests/ -q
python eval/run_tiny_sanity.py
python eval/run_staticscale_smoke.py --dry_run      # validates the full-pipeline config
```

## Equal-budget selection

```bash
python eval/run_staticscale_select.py \
    --config configs/staticscale_select.json \
    --model Qwen/Qwen2.5-7B --device cuda:0 --seq_len 2048 --max_chunks 64 \
    --out_dir results/staticscale_qwen25_7b
```

Writes `sadnd_cap_selection.{json,csv}`, `selected_config.json`, and per-component
summaries (`groupwise_clip_gain_summary.json`, `fp_mask_refinement_summary.json`,
`cascade_budget_summary.json`, `layerwise_fp_budget.json`, `int_permutation_summary.json`).

## Multi-seed robustness (primary result)

```bash
python eval/run_staticscale_multiseed.py \
    --model Qwen/Qwen2.5-7B --device cuda:0 \
    --fp_ratio 0.20 --seeds 0,1,2 --seq_len 2048 --max_chunks 64 \
    --out_dir results/staticscale_multiseed_qwen25_7b
```

Loads the model once and compares clean SADND vs CAP+ vs StaticScale at the same FP
budget. Writes `multiseed_results.json` and `gt_multiseed_summary.csv`. Robust iff ≥2/3
seeds clear −0.001 PPL **and** mean Δ < −0.001.

The recorded multi-seed run is at `results/sadnd_cap_gt_multiseed_qwen25_7b/` (StaticScale
vs clean SADND: mean Δ = −0.00293, std 0.00069, 3/3 clear, robust=True; vs CAP+: mean
Δ = −0.00108, 2/3 clear, borderline).

## Model zoo + matrix (optional)

```bash
python eval/check_model_zoo_availability.py --zoo configs/model_zoo_7b.json   # no downloads
python eval/run_7b_matrix.py --config configs/staticscale_7b_matrix.json \
    --device cuda:0 --seq_len 2048 --max_chunks 64
```

Models not present in the local cache are skipped (logged), never downloaded.

## Summarizing

```bash
python eval/summarize_staticscale_results.py --result_dir results/staticscale_qwen25_7b
```

## Determinism notes

- Calibration is seeded by `cfg.seed`; the same seed + model reproduces the same plan.
- `eval/run_staticscale_multiseed.py` loops seeds 0/1/2 by default.
- Only what is measured is reported; failed/skipped candidates are recorded, never
  fabricated.
