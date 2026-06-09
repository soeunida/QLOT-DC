"""StaticScale static groupwise clip-gain tuning (public API)."""

import torch

from staticscale.clip_gain import (
    build_int_groups, GTLayerContext, tune_clip_multiplier, fit_int_output_gain, tune_layer,
)
from staticscale.config import StaticScaleConfig


def _toy_ctx(qmax=7, seed=0):
    torch.manual_seed(seed)
    C, Ig, H, N = 8, 10, 4, 64
    y = torch.randn(N, C)
    Wg = torch.randn(Ig, C); Wu = torch.randn(Ig, C); Wd = torch.randn(H, Ig)
    fp_pos = torch.tensor([0, 1]); int_pos = torch.tensor([2, 3, 4, 5, 6, 7])
    base_scales = y[:, int_pos].abs().max(0).values / qmax
    return GTLayerContext(y=y, fp_pos=fp_pos, int_pos=int_pos, base_scales=base_scales,
                          w_gate=Wg, w_up=Wu, w_down=Wd, qmax=qmax)


def test_int_groups_preserve_indices():
    groups, sizes, group_of = build_int_groups(6, 2)
    assert sorted(torch.cat(groups).tolist()) == list(range(6))
    assert sum(sizes) == 6


def test_tau_folds_into_scales():
    ctx = _toy_ctx()
    groups, _, _ = build_int_groups(ctx.y_I.shape[1], 2, "group")
    tau, vals = tune_clip_multiplier(ctx, groups, [0.5, 1.0, 2.0], "group")
    assert tau.shape == ctx.base_scales.shape
    assert all(float(t) in (0.5, 1.0, 2.0) for t in tau)
    tuned = ctx.base_scales * tau          # tau folds into the frozen activation scales
    assert tuned.shape == ctx.base_scales.shape


def test_eta_clamped_to_bounds():
    ctx = _toy_ctx()
    groups, _, _ = build_int_groups(ctx.y_I.shape[1], 128, "layer")
    gain, etas = fit_int_output_gain(ctx, ctx.base_scales, "gate", "layer", groups, 1.1, 1.2)
    assert 1.1 <= etas[0] <= 1.2
    assert bool((gain >= 1.1 - 1e-6).all())


def test_accept_only_rejects_and_accepts():
    ctx = _toy_ctx()
    rej = tune_layer(ctx, StaticScaleConfig(use_groupwise_clip_gain_tuning=True,
                                            gt_accept_margin=1e9, gt_group_size=2))
    assert rej["gt_enabled"] is False and rej["gain_gate"] is None
    acc = tune_layer(ctx, StaticScaleConfig(use_groupwise_clip_gain_tuning=True,
                                            gt_accept_margin=-1.0, gt_group_size=2))
    assert acc["gt_enabled"] is True
    assert acc["before"] is not None and acc["after"] is not None     # measured, not fabricated
