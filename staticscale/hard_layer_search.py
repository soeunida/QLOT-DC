"""StaticScale hard-layer focused FP-budget reallocation.

After CAP+ + clip tuning, some layers retain larger residual MLP-output error than
others. This module moves a slice of the FP budget from low-error ("easy") layers to
the high-error ("hard") layers, **preserving the exact total FP-channel budget**, then
re-tunes the INT clip scales (``tau``) for the changed layers. The reallocation is a
static calibration decision (per-layer ``k_fp`` is frozen); inference does no runtime
top-k / sort / routing.

Whether this helps is an empirical question — it is gated by D_sel selection and
reported on D_eval only. No improvement is claimed unless a result file verifies it.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from staticscale import joint_mask_scale_search as J


# --------------------------------------------------------------------------- #
# Pure helpers (toy-testable, no model)
# --------------------------------------------------------------------------- #
def rank_hard_layers(layer_errors: Dict[int, float], top_k: int) -> List[int]:
    """Return the ``top_k`` layer ids with the largest error (descending)."""
    return sorted(layer_errors, key=lambda i: -float(layer_errors[i]))[: int(top_k)]


def reallocate_fp_budget(per_layer_k: Dict[int, int], layer_errors: Dict[int, float],
                         channels: Dict[int, int], top_k_hard: int,
                         move_frac: float = 0.2) -> Dict[int, int]:
    """Move FP budget from low-error donor layers to high-error hard layers.

    Each donor contributes ``round(move_frac * k)`` channels to a pool; the pool is
    added to the hard layers (capped at each layer's channel count ``C``). Any
    leftover that cannot be placed is returned to donors. The **total budget is
    preserved exactly** (asserted). Returns a new ``{layer: k_fp}`` dict.
    """
    total = int(sum(per_layer_k.values()))
    order = sorted(per_layer_k, key=lambda i: -float(layer_errors.get(i, 0.0)))
    hard = order[: int(top_k_hard)]
    donors = [i for i in order[int(top_k_hard):]]
    new = dict(per_layer_k)
    pool = 0
    for d in sorted(donors, key=lambda i: float(layer_errors.get(i, 0.0))):
        take = min(int(round(per_layer_k[d] * move_frac)), per_layer_k[d])
        new[d] -= take
        pool += take
    # distribute the pool across hard layers, capped at channel count
    guard = 0
    while pool > 0 and any(new[h] < channels[h] for h in hard) and guard < 10_000_000:
        for h in hard:
            if pool <= 0:
                break
            if new[h] < channels[h]:
                new[h] += 1
                pool -= 1
        guard += 1
    # return any leftover to donors (preserve total exactly)
    di = 0
    donor_list = donors or list(per_layer_k.keys())
    while pool > 0:
        d = donor_list[di % len(donor_list)]
        if new[d] < channels[d]:
            new[d] += 1
            pool -= 1
        di += 1
        if di > 10_000_000:
            break
    assert sum(new.values()) == total, "FP budget reallocation must preserve the total"
    return new


# --------------------------------------------------------------------------- #
# Per-layer error estimate (CAP+ + clip residual) -> budget_fn for the builder
# --------------------------------------------------------------------------- #
def _hard_layer_budget_fn(top_k_hard: int, move_frac: float):
    @torch.no_grad()
    def budget_fn(base_plan, y_by, model, cfg, device):
        per_layer_k, channels, errors = {}, {}, {}
        for i in sorted(base_plan.layers.keys()):
            lr = base_plan.layers[i]
            per_layer_k[i] = int(lr.k_fp)
            channels[i] = int(lr.num_channels)
            mlp = J._get_mlp(model, i)
            if lr.int_indices.numel() == 0:
                errors[i] = 0.0
                continue
            score = J._output_aware_score(lr.delta_tilde.to(device), mlp, device)
            res = J.evaluate_layer_orderings(
                y_by[i].to(device).float(), mlp, lr.fp_indices, lr.int_indices, score,
                tau_grid=list(cfg.gt_clip_candidates), group_size=cfg.gt_group_size,
                granularity=cfg.gt_clip_granularity, refine_margin=cfg.fp_refine_margin,
                max_swaps=0, candidate_pool=cfg.fp_refine_candidate_pool,
                qmax=cfg.qmax, eps=cfg.eps_scale, p_act=cfg.p_act,
                families=["capplus_clip"])
            errors[i] = float(res["capplus_clip"]["proxy"])
        return reallocate_fp_budget(per_layer_k, errors, channels, top_k_hard, move_frac)
    return budget_fn


@torch.no_grad()
def build_hard_layer_plan(model, tokenizer, cfg, top_k: int = 8, move_frac: float = 0.2,
                          device: str = "cpu", batch_size: int = 4,
                          allow_synthetic: bool = False):
    """Build a static plan that reallocates FP budget toward the hardest layers
    (by CAP+ + clip residual MLP-output error) and re-tunes tau. Budget total fixed."""
    return J.build_modified_plan(
        model, tokenizer, cfg, device=device, batch_size=batch_size,
        budget_fn=_hard_layer_budget_fn(int(top_k), float(move_frac)),
        refine_after=True, allow_synthetic=allow_synthetic)
