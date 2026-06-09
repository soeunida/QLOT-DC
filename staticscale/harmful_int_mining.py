"""StaticScale harmful-INT-channel mining (post clip tuning).

After CAP+ + clip tuning, a few INT channels may still contribute disproportionate
MLP-output error (their weight-normalized INT8 quantization residual stays large even
under the tuned clip scales). This module mines such "harmful" INT channels and tries
to promote them into the FP set, swapping out the weakest boundary FP channels so the
**FP budget is preserved exactly**, then re-tunes the affected group ``tau``. Each swap
is accepted only if a measured MLP-output proxy improves under the re-tuned scales.

The mined mask + tuned scales are static metadata; inference does no runtime
top-k / sort / routing. Gated by D_sel; reported on D_eval only.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from staticscale import joint_mask_scale_search as J
from qlot_rms.groupwise_clip_gain import build_gt_context, build_int_groups, tune_clip_multiplier
from qlot_rms.quant import quantize_activation_int8


# --------------------------------------------------------------------------- #
# Pure helpers (toy-testable)
# --------------------------------------------------------------------------- #
def score_int_harm(residual_energy: torch.Tensor, wcol2: torch.Tensor) -> torch.Tensor:
    """Per-INT-channel harm = quantization residual energy x (gate+up) weight col L2^2.
    Higher means the channel injects more MLP-output error even after clip tuning."""
    return residual_energy * wcol2


def mine_harmful_swaps(harm_int: torch.Tensor, fp_score: torch.Tensor,
                       fp_idx: torch.Tensor, int_idx: torch.Tensor,
                       max_swaps: int, pool: int) -> List[Tuple[int, int]]:
    """Pair the most-harmful INT channels (to promote to FP) with the weakest FP
    channels (to demote to INT). Returns ``[(fp_out, int_in), ...]`` (budget-preserving:
    one out, one in per pair)."""
    if fp_idx.numel() == 0 or int_idx.numel() == 0:
        return []
    n = int(min(pool, max_swaps, fp_idx.numel(), int_idx.numel()))
    fp_idx = fp_idx.long()
    int_idx = int_idx.long()
    weak_fp = fp_idx[torch.argsort(fp_score[fp_idx], descending=False)][:n]
    harm_order = torch.argsort(harm_int, descending=True)[:n]      # positions in int_idx
    strong_int = int_idx[harm_order]
    return [(int(weak_fp[i]), int(strong_int[i])) for i in range(n)]


# --------------------------------------------------------------------------- #
# Producer (fp_int_producer for build_modified_plan)
# --------------------------------------------------------------------------- #
def _harmful_int_producer(max_swaps: int, pool: int):
    @torch.no_grad()
    def producer(y, mlp, fp0, int0, score, base_full, cfg):
        device = y.device
        C = y.shape[1]
        k_fp = int(fp0.numel())
        tau_grid = list(cfg.gt_clip_candidates)
        gs, gran = cfg.gt_group_size, cfg.gt_clip_granularity
        qmax, eps = cfg.qmax, cfg.eps_scale
        w_gate = mlp.gate_proj.weight.data.to(device).float()
        w_up = mlp.up_proj.weight.data.to(device).float()
        w_down = mlp.down_proj.weight.data.to(device).float()
        b = (None if mlp.gate_proj.bias is None else mlp.gate_proj.bias.data.to(device).float(),
             None if mlp.up_proj.bias is None else mlp.up_proj.bias.data.to(device).float(),
             None if mlp.down_proj.bias is None else mlp.down_proj.bias.data.to(device).float())

        # harm score for current INT set under tuned scales
        from types import SimpleNamespace
        cfg_like = SimpleNamespace(qmax=qmax, eps_scale=eps)
        view = SimpleNamespace(fp_indices=fp0, int_indices=int0,
                               act_scales=base_full.index_select(0, int0))
        ctx = build_gt_context(y, view, mlp, cfg_like, device=device, max_tokens=y.shape[0])
        groups, _, _ = build_int_groups(int0.numel(), gs, gran)
        tau_pc, _ = tune_clip_multiplier(ctx, groups, tau_grid, gran)
        tuned_int = (ctx.base_scales * tau_pc).clamp_min(1e-12)
        q = quantize_activation_int8(ctx.y_I, tuned_int, qmax=qmax)
        resid = (q - ctx.y_I).pow(2).mean(0)                    # [n_int]
        harm = score_int_harm(resid, ctx.wcol2)                 # [n_int], aligned to int0

        cands = mine_harmful_swaps(harm, score, fp0, int0, max_swaps, pool)

        # greedily accept swaps that improve the unified proxy under re-tuned scales
        mask = J._mask_from_fp(C, fp0, device)
        tuned_full, _ = J._tuned_full_scales(y, mlp, fp0, int0, base_full, tau_grid,
                                             gs, gran, qmax, eps)
        base = J._proxy_under(y, w_gate, w_up, w_down, b, tuned_full, mask, qmax, eps)
        for fp_out, int_in in cands:
            if (not bool(mask[fp_out])) or bool(mask[int_in]):
                continue
            trial = mask.clone()
            trial[fp_out] = False
            trial[int_in] = True
            fp_t = torch.nonzero(trial, as_tuple=False).squeeze(-1).long()
            int_t = torch.nonzero(~trial, as_tuple=False).squeeze(-1).long()
            tuned_t, _ = J._tuned_full_scales(y, mlp, fp_t, int_t, base_full, tau_grid,
                                              gs, gran, qmax, eps)
            err = J._proxy_under(y, w_gate, w_up, w_down, b, tuned_t, trial, qmax, eps)
            if err < base - cfg.fp_refine_margin:
                mask, base = trial, err
        fp_new = torch.nonzero(mask, as_tuple=False).squeeze(-1).long()
        int_new = torch.nonzero(~mask, as_tuple=False).squeeze(-1).long()
        assert int(fp_new.numel()) == k_fp, "harmful-INT mining must preserve k_fp"
        return fp_new.to(device), int_new.to(device)
    return producer


@torch.no_grad()
def build_harmful_int_plan(model, tokenizer, cfg, max_swaps: int = 16, pool: int = 64,
                           device: str = "cpu", batch_size: int = 4,
                           allow_synthetic: bool = False):
    """Build a static plan that promotes post-clip harmful INT channels into FP
    (budget-preserving) and re-tunes tau. Gated by D_sel; D_eval is reporting only."""
    return J.build_modified_plan(
        model, tokenizer, cfg, device=device, batch_size=batch_size,
        fp_int_producer=_harmful_int_producer(int(max_swaps), int(pool)),
        refine_after=False, allow_synthetic=allow_synthetic)
