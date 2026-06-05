"""SADND-CAP policy: output-aware routing, global FP budget, packing-aware
INT permutation, and equal-budget accept-only selection.

All decisions are made at calibration time and frozen. No runtime top-k / sort /
input-dependent routing. No correction modules. No speedup is claimed.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


# --------------------------------------------------------------------------- #
# 1. Output-aware SADND score
# --------------------------------------------------------------------------- #
def compute_output_aware_sadnd_score(
    delta_tilde: torch.Tensor, w_gate_norm: torch.Tensor, w_up_norm: torch.Tensor
) -> torch.Tensor:
    """score_c = delta_c * (||W_gate[:,c]||_2 + ||W_up[:,c]||_2).

    Weight column L2 norms (nn.Linear is [out, in]). Channels that are both hard
    to quantize AND heavily used by the projection are prioritized for FP.
    """
    return delta_tilde.float() * (w_gate_norm.float() + w_up_norm.float())


# --------------------------------------------------------------------------- #
# 2. FP/INT channel selection (FP = highest score)
# --------------------------------------------------------------------------- #
def select_fp_int_channels(
    score: torch.Tensor, k_fp: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(fp_indices, int_indices)`` in ORIGINAL channel order.

    FP = the ``k_fp`` highest-score channels; INT = the rest.
    """
    C = score.numel()
    k_fp = max(0, min(int(k_fp), C))
    if k_fp > 0:
        fp = torch.topk(score, k_fp, largest=True).indices
    else:
        fp = torch.empty(0, dtype=torch.long, device=score.device)
    mask = torch.zeros(C, dtype=torch.bool, device=score.device)
    mask[fp] = True
    fp_sorted = torch.sort(fp).values.long()
    int_sorted = torch.nonzero(~mask, as_tuple=False).squeeze(-1).long()
    return fp_sorted, int_sorted


# --------------------------------------------------------------------------- #
# 3. Global layer-wise FP budget allocation
# --------------------------------------------------------------------------- #
def allocate_global_fp_budget(
    scores: Dict[int, torch.Tensor], fp_ratio: float
) -> Dict[int, int]:
    """Distribute ONE global FP budget across layers by global score ranking.

    Total budget = floor(fp_ratio * sum_l C_l). Globally rank all (layer, channel)
    pairs by score; FP = the global top-``budget``. Each layer's k_fp is how many
    of its channels fall in that global top set. The TOTAL budget is preserved
    (``same_global_fp_budget``); layers with more high-distortion channels get
    more FP than a uniform per-layer split.
    """
    total_C = sum(int(s.numel()) for s in scores.values())
    budget = int(fp_ratio * total_C)
    if budget <= 0:
        return {i: 0 for i in scores}
    flat_vals, flat_layer = [], []
    for i, s in scores.items():
        flat_vals.append(s.float())
        flat_layer.append(torch.full((s.numel(),), i, dtype=torch.long))
    vals = torch.cat(flat_vals)
    lay = torch.cat(flat_layer)
    budget = min(budget, vals.numel())
    top = torch.topk(vals, budget, largest=True).indices
    chosen_layers = lay[top]
    k = {i: 0 for i in scores}
    for i in scores:
        k[i] = int((chosen_layers == i).sum().item())
    return k


# --------------------------------------------------------------------------- #
# 4. Packing-aware INT permutation
# --------------------------------------------------------------------------- #
def build_packing_aware_int_permutation(
    int_indices: torch.Tensor, int_scales: torch.Tensor, mode: str, group_size: int
) -> torch.Tensor:
    """Reorder INT channels so each W8-G128 group has similar activation scale.

    Returns the reordered ``int_indices`` (a permutation of the input). Only the
    INT channels are reordered; the FP block is untouched. ``int_scales`` is the
    per-INT-channel scale aligned with ``int_indices`` (same order).
    """
    n = int_indices.numel()
    if n == 0 or mode == "original":
        return int_indices.clone()
    s = int_scales.float()
    if mode in ("scale_sorted", "packing_aware"):
        order = torch.argsort(s)                       # ascending by scale
    elif mode == "scale_clustered":
        # coarse log-scale buckets, contiguous; stable within bucket
        eps = 1e-12
        bucket = torch.log(s.clamp_min(eps))
        nb = max(1, (n + group_size - 1) // group_size)
        lo, hi = bucket.min(), bucket.max()
        width = (hi - lo) / nb + eps
        bid = ((bucket - lo) / width).floor().clamp(0, nb - 1)
        order = torch.argsort(bid * (s.max() + 1.0) + s)   # bucket-major, scale-minor
    else:
        order = torch.arange(n)
    return int_indices[order].clone()


# --------------------------------------------------------------------------- #
# 5. Equal-budget accept-only selection
# --------------------------------------------------------------------------- #
def equal_budget_accept_only_select(
    candidate_ppls: Dict[str, float], sadnd_ppl: float, margin: float
) -> Tuple[str, bool]:
    """Pick the best candidate that beats SADND@same-budget by ``margin``.

    Returns ``(winner_name, clear_improvement)``. If no candidate clears the
    margin, falls back to ``"sadnd"`` with ``clear_improvement=False``.
    """
    if not candidate_ppls:
        return "sadnd", False
    best = min(candidate_ppls, key=candidate_ppls.get)
    if candidate_ppls[best] < sadnd_ppl - margin:
        return best, True
    return "sadnd", False


# --------------------------------------------------------------------------- #
# 6. Convenience: build per-layer FP/INT + packing-aware perm
# --------------------------------------------------------------------------- #
def build_layer_fp_int_perm(
    score: torch.Tensor, k_fp: int, int_scales_full: torch.Tensor,
    int_permutation_mode: str, group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(fp_indices, int_indices_packed, perm, mask)`` for one layer.

    ``int_scales_full`` is the per-channel scale for ALL channels (length C);
    INT scales are gathered in INT order before packing-aware reordering.
    """
    C = score.numel()
    fp, int_orig = select_fp_int_channels(score, k_fp)
    int_scales = int_scales_full[int_orig]
    int_packed = build_packing_aware_int_permutation(
        int_orig, int_scales, int_permutation_mode, group_size)
    perm = torch.cat([fp, int_packed], dim=0)
    mask = torch.zeros(C, dtype=torch.bool)
    mask[fp] = True
    return fp, int_packed, perm, mask
