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
