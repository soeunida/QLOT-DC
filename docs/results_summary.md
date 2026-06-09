# Results summary

All numbers are measured in this repository with the `torch_reference` backend
(fake-quantized, **correctness-only**). **No backend-independent speedup is claimed.**
Selection uses a small validation split; multi-seed checks use a held set of seeds.
Packed-layout throughput, where mentioned, is **prototype diagnostic evidence only**.

StaticScale is the final method; SADND routing, the cascade/marginal FP budget (CAP+),
equal-budget FP mask refinement, and static groupwise clip-gain tuning are its internal
components.

## Primary result: static clip-scale tuning is the dominant driver (Qwen2.5-7B)

WikiText-2, seq_len=2048, 64 chunks, fp_ratio=0.20, seeds 0/1/2 (fp16 = 6.5811). Decision
rule: ≥2/3 seeds clear −0.001 PPL **and** mean Δ < −0.001.

| seed | clean SADND | CAP+ (cascade+marginal) | StaticScale | Δ SS−SADND | Δ SS−CAP+ |
|---|---|---|---|---|---|
| 0 | 6.5850 | 6.5834 | 6.5815 | −0.0035 | −0.0018 |
| 1 | 6.5850 | 6.5829 | 6.5817 | −0.0033 | −0.0012 |
| 2 | 6.5845 | 6.5827 | 6.5826 | −0.0020 | −0.0002 |

| comparison | mean Δ | std Δ | seeds clearing | robust |
|---|---|---|---|---|
| StaticScale vs clean SADND | −0.00293 | 0.00069 | 3/3 | True |
| StaticScale vs CAP+ | −0.00108 | 0.00068 | 2/3 | borderline |

**Interpretation (honest):**
- **Clip-scale tuning is the dominant driver.** Static groupwise clip tuning carries the
  improvement; the structural stages (routing, budget, mask refinement) are supporting
  mechanisms that keep clip tuning stable under a fixed FP budget.
- The gain over the budget-only CAP+ baseline is **small and borderline** (mean −0.00108,
  2/3 clearing, seed 2 ≈ tie). We do **not** claim a strong full-pipeline improvement.
- **`CAP+ + clip` is already close to full StaticScale**; adding equal-budget mask
  refinement or joint mask-scale search adds no measurable gain (see the joint-search and
  ablation notes below and `docs/negative_findings.md`).
- `tau` mean ≈ 1.24–1.25; **output gain `eta ≈ 1.0` contributes little**; **group-wise
  `eta` is experimental and rejected by accept-only**.
- The effect is **small and budget-dependent** (~0.003 PPL, ~0.04%). **No speedup is
  claimed.**

> **Table note (component attribution / Table 9).** Static clip-scale tuning is the
> dominant driver of StaticScale's equal-budget improvement; `CAP+ + clip` is close to
> the full pipeline. **Joint mask-scale search did not improve over the additive full
> pipeline in the fp0.20 diagnostic** (proxy gate failed; see `docs/negative_findings.md`).

Data: `results/sadnd_cap_gt_multiseed_qwen25_7b/multiseed_results.json` and
`.../staticscale_summary.csv`.

## Single-seed equal-budget selection (Qwen2.5-7B, 64 chunks)

fp16 6.5811, INT8 PTQ 6.5877. StaticScale (clip-gain) clears both margins at both
budgets:

| fp | clean SADND | CAP+ | StaticScale | Δ vs SADND | Δ vs CAP+ |
|---|---|---|---|---|---|
| 0.06 | 6.5862 | 6.5855 | 6.5824 | −0.0038 | −0.0031 |
| 0.20 | 6.5850 | 6.5830 | 6.5818 | −0.0032 | −0.0012 |

The clip-gain projection-MSE proxy is cut ~2× on all 28 layers (e.g. 2.0e-4 → 1.1e-4 at
fp=0.20). Data: `results/sadnd_cap_gt_qwen25_7b/` (selection + `groupwise_clip_gain_summary.json`).

## Component-level multi-seed history (Qwen2.5-7B, 64 chunks)

- **FP budget (CAP+).** At fp=0.06 the cascade/marginal budget is a weak sub-margin trend
  over clean SADND (mean Δ ≈ −0.0006, 0/3 clearing). At fp=0.20 it robustly beats clean
  SADND (mean Δ = −0.00185, std 0.00019, 3/3 clearing) — the first component to clear the
  criterion versus clean SADND. Data: `results/sadnd_cap_cascade_multiseed_qwen25_7b/`.
- **Equal-budget FP mask refinement.** On the near-lossless models tested it makes very
  few swaps (0–1 per layer) and is essentially quality-neutral versus CAP+ at equal
  budget — it never regresses (accept-only) but the routing score is already near-optimal
  at the boundary here. Retained as a safe static refinement. Data:
  `results/sadnd_cap_refine_tinyllama/`, `results/sadnd_cap_gt_qwen25_7b/`.
- **Static groupwise clip-gain tuning.** The dominant component (see primary result). Layer-wise
  gain accepted on all 28 layers; group-wise gain experimental and rejected.

## TinyLlama-1.1B smoke (1024 seq, 32 chunks, single seed)

fp16 8.8232. StaticScale (clip-gain) at fp=0.20 = 8.8241 vs clean SADND 8.8308
(Δ −0.0067) and CAP+ 8.8274 (Δ −0.0033). Consistent with the Qwen finding: clip-driven,
`tau ≈ 1.24`, `eta ≈ 1.0`, group-gain rejected. Data: `results/sadnd_cap_gt_tinyllama/`.

## Regime note

In the INT8-near-lossless regime tested (FP16→INT8 PTQ is only a few ×1e-3 PPL), all
equal-budget effects are small. The removed static-correction methods did not robustly
beat SADND at equal budget; see `docs/negative_findings.md`. A decisive test would need a
regime where INT8 materially degrades (e.g. lower-bit), which is out of scope here.

No backend-independent speedup is claimed; `torch_reference` is correctness-only; a real
packed FP16+INT8 kernel is not implemented.
