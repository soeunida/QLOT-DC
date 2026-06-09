# Negative findings (removed from the active method)

StaticScale deliberately contains **no correction modules** and **no dynamic runtime
normalization**. Several such ideas were implemented and evaluated earlier and did
**not** robustly beat SADND at equal FP budget in the INT8-near-lossless regime; they
were removed from the active code (recoverable via the git tag
`backup-before-final-sadnd-cap-cleanup`). This page records them honestly so they are not
re-attempted blindly.

## Static output corrections (removed)
- **Static diagonal compensation (DC)** and its output-aware / adaptive-FP / projection
  bias / low-rank extensions: on TinyLlama-1.1B and Qwen2.5-7B, differences versus SADND
  at equal FP budget were within ~1e-3 PPL (noise); any gain over INT8 PTQ tracked the FP
  budget, which SADND captures equally.
- **Block-output correction (OBC)** (block bias / affine / low-rank on the FFN output):
  block-bias looked better on a small validation split but the advantage did **not**
  generalize to the full test (Qwen2.5-7B: slightly worse at equal fp; margin not
  cleared). The block **low-rank** variant badly **overfit** the validation split
  (validation PPL collapsed while test PPL rose) and was caught by a validation-PPL gate.

**Conclusion:** in this regime, static *output corrections* do not provide a robust
equal-budget quality gain over SADND. StaticScale therefore keeps only routing, budget,
mask refinement, packing, and static INT-scale tuning — all gated by the equal-budget
accept-only rule.

## Dynamic runtime INT normalization (removed)
A variant that **normalized INT activations dynamically at runtime** (a per-group
runtime rescale) was tried. It introduced a function shift and runtime work, and was
removed. Its lesson motivated StaticScale's **static** groupwise clip-gain tuning: the
useful part (better INT activation scaling) is captured **once at calibration** as
static metadata (`tau_g` folded into the frozen activation scales, `eta` folded into the
INT weight columns), with **nothing dynamic at inference**.

## Joint mask-scale optimization (no gain over the additive pipeline)
We tested whether *jointly* optimizing the FP mask and the static INT clip-scales beats
the additive `CAP+ + mask refinement + clip-tuning` pipeline (the reviewer concern that
the additive full pipeline adds little over `CAP+ + clip`). The joint search interleaves
candidate FP/INT swaps with local retuning of INT group scales (orderings
`joint_retune_then_swap`, `joint_swap_then_retune`), gated cheap-first on a per-layer
MLP-output rel-L2 calibration proxy.

**Result — Qwen2.5-7B, fp_ratio=0.20, seed 0 (`quick` stage, 12/28 layers sampled):**

| family | mean proxy (rel-L2) | Δ vs additive full |
|---|---|---|
| capplus | 0.00747636 | +0.00196553 |
| capplus_clip (`CAP+ + clip`) | 0.00573511 | +0.00022428 |
| capplus_refine_clip (additive full) | 0.00551082 | 0.00000000 |
| joint_retune_then_swap | 0.00551082 | **+0.00000000** |
| joint_swap_then_retune | 0.00555283 | **+0.00004201 (worse)** |

- The **proxy gate FAILED**: no joint ordering improved the calibration proxy over the
  additive full pipeline by the threshold (0.0001).
- `joint_retune_then_swap` **matched** the additive full pipeline but did **not** improve
  it (0 jointness gain); `joint_swap_then_retune` was **worse**.
- Per the cheap-first policy, **no `D_sel` / `D_eval` PPL run was performed** (so no
  `candidates_sel.csv`, `candidates_eval.csv`, or `best_candidate.json` were produced);
  spending GPU on the full/multiseed joint search was not justified.
- **Conclusion: no evidence for a strong joint mask-scale gain.** The residual error here
  is not primarily a mask/scale mismatch; the recoverable error is **scale-driven** and is
  already captured by static clip tuning. This is why `CAP+ + clip` is close to the full
  pipeline and why StaticScale is framed as a static clip-scale calibration method.

Data: `results/staticscale_joint_qwen25_7b_fp020_seed0/{candidates_proxy.csv,summary.txt}`.

## Components that are safe but near-neutral here
- **Equal-budget FP mask refinement** makes very few swaps on the near-lossless models
  tested and is essentially quality-neutral versus the budget baseline (never regresses,
  by accept-only). It is retained as a safe static refinement, not a reliable win in this
  regime.
- **Group-wise output gain** (vs layer-wise) is experimental: the independent per-group
  fit tended to worsen the combined proxy and is rejected by accept-only; it falls back
  to the layer-wise gain / budget baseline.

## Backend / throughput
The packed FP/INT prototype's throughput numbers are **prototype diagnostic evidence
only**. The active backend is correctness/reference-oriented; **no backend-independent
speedup is claimed**, and a production packed FP16+INT8 kernel is not implemented.
