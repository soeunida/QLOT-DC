# Results summary

All numbers are measured in this repository with the `torch_reference` backend
(fake-quantized, **correctness-only**). **No speedup is claimed.** Selection uses
a small validation split; full WikiText-2 test is evaluated once.

## Final supported method: SADND-CAP

SADND-CAP = output-aware SADND routing + global layer-wise FP budget allocation +
packing-aware INT permutation + equal-budget accept-only selection. It keeps only
the routing/packing choices that are safe under the equal-budget rule (a choice
is kept only if it beats SADND at the *same* FP budget by margin).

Run the equal-budget selection (`eval/select_sadnd_cap.py`) and the full test
(`eval/eval_perplexity.py`) per the README. Outputs are written under
`results/sadnd_cap_*`.

### Multi-seed equal-budget robustness (Qwen2.5-7B, fp_ratio=0.06)

A single full-test run (145 chunks, calib 128) showed SADND-CAP at 6.8019 vs
clean SADND 6.8031 — Δ = −0.0013, barely clearing the 0.001 margin. To test
whether that is real or noise, we ran a multi-seed equal-budget check (seeds
0/1/2; screen at max_chunks=64, calib 64; same fp_ratio=0.06; SADND-CAP =
output-aware + global budget + packing-aware vs clean SADND = fixed + original):

| seed | SADND | output-aware | SADND-CAP | Δ (CAP − SADND) | clears −0.001? |
|---|---|---|---|---|---|
| 0 | 6.5863 | 6.5854 | 6.5854 | −0.0009 | no |
| 1 | 6.5860 | 6.5862 | 6.5856 | −0.0004 | no |
| 2 | 6.5855 | 6.5857 | 6.5849 | −0.0006 | no |

Aggregate: **mean Δ = −0.0006, std Δ = 0.0002, margin-clearing seeds = 0/3,
`robust_better = False`.**

**Interpretation (honest):**
- SADND-CAP is **consistently slightly better** than clean SADND at equal FP
  budget (Δ negative on all 3 seeds), but the gain is **below the predefined
  0.001 margin** and within the calibration/seed noise floor.
- The single-run full-test improvement of −0.0013 was **likely at the favorable
  end of seed/calibration noise**; the multi-seed mean (−0.0006) is the honest
  estimate.
- **No robust improvement is claimed.** SADND-CAP is best described as a
  **policy/layout framework with a weak positive trend**, not a decisive quality
  win, in this INT8-near-lossless regime.
- output-aware routing alone is neutral (Δ vs SADND mixed sign across seeds); the
  small trend comes from the global budget + packing-aware layout, not routing.
- No speedup claim; `torch_reference` is correctness-only.

Data: `results/sadnd_cap_multiseed_qwen25_7b/multiseed_results.json` and the
single full-test run in `results/sadnd_cap_full_qwen25_7b/`.

### Multi-seed equal-budget robustness — SADND-CAP+ (Qwen2.5-7B, fp_ratio=0.20)

SADND-CAP+ adds cascade-aware FP budget allocation + marginal-gain FP allocation
on top of SADND-CAP (see `docs/method.md`). We ran a multi-seed equal-budget
check on Qwen2.5-7B (WikiText-2 validation, seq_len=2048, 64 chunks, fp_ratio=0.20,
seeds 0/1/2) comparing clean SADND (fixed + original) vs SADND-CAP
(global+packing) vs SADND-CAP+ (cascade+marginal), all at the **same** FP budget:

| seed | clean SADND | SADND-CAP (global+packing) | Δ | SADND-CAP+ (cascade+marginal) | Δ |
|---|---|---|---|---|---|
| 0 | 6.5850 | 6.5833 | −0.0017 | 6.5834 | −0.0017 |
| 1 | 6.5850 | 6.5845 | −0.0005 | 6.5829 | −0.0021 |
| 2 | 6.5845 | 6.5829 | −0.0016 | 6.5827 | −0.0018 |

Aggregate (Δ = candidate − clean SADND; decision rule: ≥2/3 seeds clear −0.001
**and** mean Δ < −0.001):

| variant | mean Δ | std Δ | seeds clearing −0.001 | `robust_better` |
|---|---|---|---|---|
| **SADND-CAP+ (cascade+marginal)** | **−0.00185** | 0.00019 | **3/3** | **True** |
| SADND-CAP (global+packing) | −0.00129 | — | 2/3 | True |

**Interpretation (honest):**
- This is the **first method in this project to satisfy the multi-seed
  equal-budget robustness criterion** (SADND-CAP+: 3/3 seeds clear the margin,
  mean Δ = −0.00185, low spread std 0.00019).
- The improvement is **small but consistent** — ~0.0018 PPL on a 6.58 baseline
  (~0.03%), inside the INT8-near-lossless regime. **This is not a large quality
  win.**
- The result **supports cascade-aware + marginal-gain FP budget allocation as a
  useful extension** over clean SADND, under the equal-budget accept-only rule.
- The claim is **budget-dependent**: robustness is shown at **fp_ratio=0.20**, and
  is **not yet shown at fp_ratio=0.06** (where the earlier base SADND-CAP
  multi-seed gave mean Δ = −0.0006, 0/3 clearing — see the section above).
- **No speedup is claimed**; `torch_reference` is correctness-only.

Data: `results/sadnd_cap_cascade_multiseed_qwen25_7b/multiseed_results.json`.

## Prior correction attempts — negative findings (removed from active code)

Three correction methods were implemented and evaluated, then removed (git tag
`backup-before-final-sadnd-cap-cleanup`):

- **Q-LOT-DC** (static diagonal compensation) and **Q-LOT-DC+** (output-aware +
  adaptive FP + projection bias / low-rank): on TinyLlama-1.1B and Qwen2.5-7B,
  **did not robustly beat SADND at equal FP budget** (differences within ~1e-3,
  i.e. noise; any gain over INT8 PTQ tracked the FP budget, which SADND captures
  equally).
- **Q-LOT-OBC** (block-output bias/affine/low-rank correction): block-bias looked
  better on small validation but the advantage **did not generalize** to the full
  test (Qwen2.5-7B: DC+OBC 6.8014 vs SADND 6.8008 at equal fp=0.10 — slightly
  worse; margin not cleared). Block **low-rank** correction badly **overfit**
  (validation PPL 8.82 → 10–12) and was caught by a validation-PPL gate.

**Conclusion:** in the INT8-near-lossless regime tested (FP16→INT8 PTQ is only a
few ×1e-3 PPL), static *output corrections* do not provide a robust equal-budget
quality gain over SADND. SADND-CAP therefore retains routing + packing (which are
budget/layout choices, gated by equal-budget accept-only) and drops the
corrections. A decisive test of corrections would require a regime where INT8
materially degrades (e.g. W4 / lower-bit), which is out of scope here.

No speedup is claimed; `torch_reference` is correctness-only; a real
`custom_packed` kernel is not implemented.
