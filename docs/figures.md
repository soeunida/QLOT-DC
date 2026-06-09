# Figures

Figure scripts live in `figures/` and write PNG + PDF next to themselves. They use a
headless matplotlib backend and run both as scripts and interactively.

```bash
python figures/draw_staticscale_overview.py
python figures/draw_staticscale_diagnostics.py
```

## `draw_staticscale_overview.py`
Pipeline block diagram: **Offline Calibration & Static Policy Search** (output-aware
SADND routing → cascade-aware FP budget allocation → equal-budget FP mask refinement →
static groupwise clip-gain tuning → packing-aware static FP/INT layout) feeding the
**Serving Phase: Static Tuned FP/INT Layout** (tuned scale multiplier `tau_g` per INT
group). Outputs `staticscale_overview.{png,pdf}`.

## `draw_staticscale_diagnostics.py`
Two measured-only diagnostic figures (numbers read from `results/` when present, else the
recorded Qwen2.5-7B fp=0.20 multi-seed values):

- `routing_tuning_diagnostics_staticscale.{png,pdf}` — per-seed equal-budget ΔPPL
  (StaticScale vs clean SADND and vs CAP+) and the clip-driven nature of the gain
  (`tau ≈ 1.24`, `eta ≈ 1.0`).
- `within_stack_attribution_staticscale.{png,pdf}` — cumulative attribution across the
  static stack (FP budget → mask refinement → clip-gain) at equal FP budget.

## Conventions
- Reference backend; all throughput-adjacent claims are **prototype diagnostic only —
  not a StaticScale speedup claim**.
- Figure text uses the public StaticScale vocabulary only (StaticScale, SADND routing,
  packed FP/INT prototype, static tuned layout, static scale grouping controls).
