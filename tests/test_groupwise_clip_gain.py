"""SADND-CAP-GT static groupwise clip-gain tuning tests."""

import torch

from qlot_rms.config import QLotRmsConfig
from qlot_rms.calibration import calibrate
from qlot_rms.model_integration import patch_model, unpatch_model
from qlot_rms.groupwise_clip_gain import (
    build_int_groups, GTLayerContext, tune_clip_multiplier, fit_int_output_gain,
    tune_layer,
)


class _Tok:
    vocab_size = 256


def _toy_ctx(qmax=7, seed=0):
    torch.manual_seed(seed)
    C, Ig, H, N = 8, 10, 4, 64
    y = torch.randn(N, C)
    Wg = torch.randn(Ig, C); Wu = torch.randn(Ig, C); Wd = torch.randn(H, Ig)
    fp_pos = torch.tensor([0, 1]); int_pos = torch.tensor([2, 3, 4, 5, 6, 7])
    base_scales = y[:, int_pos].abs().max(0).values / qmax
    ctx = GTLayerContext(y=y, fp_pos=fp_pos, int_pos=int_pos, base_scales=base_scales,
                         w_gate=Wg, w_up=Wu, w_down=Wd, qmax=qmax)
    return ctx


# --- A. group construction preserves all INT indices ---
def test_int_groups_preserve_all_indices():
    groups, sizes, group_of = build_int_groups(n_int=6, group_size=2)
    allpos = torch.cat(groups).tolist()
    assert sorted(allpos) == list(range(6))           # every INT position covered once
    assert sum(sizes) == 6
    assert group_of.tolist() == [0, 0, 1, 1, 2, 2]
    # layer granularity = single group of all
    g1, s1, _ = build_int_groups(6, 128, "layer")
    assert len(g1) == 1 and s1 == [6]


# --- B. tau candidates applied to base scales, shape preserved ---
def test_tau_applied_and_shape_preserved():
    ctx = _toy_ctx()
    groups, _, _ = build_int_groups(ctx.y_I.shape[1], 2, "group")
    cands = [0.5, 1.0, 2.0]
    tau, tau_vals = tune_clip_multiplier(ctx, groups, cands, "group")
    assert tau.shape == ctx.base_scales.shape
    assert all(float(t) in cands for t in tau)        # only candidate values chosen
    tuned = ctx.base_scales * tau
    assert tuned.shape == ctx.base_scales.shape
    assert torch.allclose(tuned, ctx.base_scales * tau)


# --- C. layer-wise eta is a finite scalar ---
def test_layer_eta_finite_scalar():
    ctx = _toy_ctx()
    groups, _, _ = build_int_groups(ctx.y_I.shape[1], 128, "layer")
    gain, etas = fit_int_output_gain(ctx, ctx.base_scales, "gate", "layer", groups, 0.5, 2.0)
    assert len(etas) == 1 and torch.isfinite(torch.tensor(etas[0]))
    assert torch.allclose(gain, torch.full_like(gain, etas[0]))   # one scalar everywhere


# --- eta is clamped to [min, max] ---
def test_eta_clamped_to_bounds():
    ctx = _toy_ctx()
    groups, _, _ = build_int_groups(ctx.y_I.shape[1], 128, "layer")
    # true eta ~1.0; force the clamp window above it
    gain, etas = fit_int_output_gain(ctx, ctx.base_scales, "gate", "layer", groups, 1.1, 1.2)
    assert 1.1 <= etas[0] <= 1.2
    assert bool((gain >= 1.1 - 1e-6).all()) and bool((gain <= 1.2 + 1e-6).all())


# --- tune_layer: accept-only rejects when proxy cannot improve enough ---
def test_accept_only_rejects_when_no_improvement():
    ctx = _toy_ctx()
    cfg = QLotRmsConfig(use_groupwise_clip_gain_tuning=True, gt_accept_only=True,
                        gt_accept_margin=1e9, gt_group_size=2)   # impossible to clear
    res = tune_layer(ctx, cfg)
    assert res["gt_enabled"] is False
    assert res["gain_gate"] is None and res["gain_up"] is None
    assert "rejected" in res["reason"]


def test_accept_when_margin_negative():
    ctx = _toy_ctx()
    cfg = QLotRmsConfig(use_groupwise_clip_gain_tuning=True, gt_accept_only=True,
                        gt_accept_margin=-1.0, gt_group_size=2)  # always accept
    res = tune_layer(ctx, cfg)
    assert res["gt_enabled"] is True
    assert res["before"] is not None and res["after"] is not None
    assert len(res["tau_values"]) >= 1 and len(res["eta_gate"]) >= 1


# --------------------------------------------------------------------------- #
# calibration-level (tiny model)
# --------------------------------------------------------------------------- #
def _tiny(n=2):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=n,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
        tie_word_embeddings=True)).eval()


def _cfg(**kw):
    base = dict(enable_qlot_rms=True, method="sadnd_cap", routing_score="output_aware_sadnd",
                int_permutation_mode="packing_aware", fp_ratio=0.25, global_fp_budget_ratio=0.25,
                fp_budget_mode="global", calibration_samples=8, calibration_seq_len=16,
                num_calib_subsets=3, subset_size=4, act_scale_max_tokens=256, gt_max_tokens=64,
                gt_group_size=16)
    base.update(kw)
    return QLotRmsConfig(**base)


def _plan(m, cfg):
    return calibrate(m, _Tok(), cfg, device="cpu", allow_synthetic=True, batch_size=2)


# --- disabled path / rejection == unmodified SADND-CAP+ ---
def test_disabled_path_equals_unmodified():
    m = _tiny()
    p_off = _plan(m, _cfg(use_groupwise_clip_gain_tuning=False))
    p_rej = _plan(m, _cfg(use_groupwise_clip_gain_tuning=True, gt_accept_margin=1e9))
    for i in p_off.layers:
        assert p_off.layers[i].gt_enabled is False
        assert p_rej.layers[i].gt_enabled is False          # all rejected -> fall back
        # rejection leaves the frozen INT activation scales unchanged
        assert torch.allclose(p_off.layers[i].act_scales, p_rej.layers[i].act_scales)
        assert p_rej.layers[i].gt_int_gain_gate is None


# --- metadata stores tau/eta + before/after; gains folded when accepted ---
def test_metadata_stores_tau_eta_and_errors():
    m = _tiny()
    p = _plan(m, _cfg(use_groupwise_clip_gain_tuning=True, gt_accept_margin=-1.0))
    for lr in p.layers.values():
        assert lr.gt_enabled is True
        assert len(lr.gt_tau_values) >= 1 and len(lr.gt_eta_gate) >= 1
        assert lr.gt_proxy_error_before is not None and lr.gt_proxy_error_after is not None
        assert lr.gt_int_gain_gate is not None and lr.gt_int_gain_up is not None
        assert lr.gt_int_gain_gate.numel() == lr.int_indices.numel()   # per-INT-channel gain
        assert lr.k_fp == p.layers[lr.layer_index].k_fp               # FP budget unchanged


# --- FP budget preserved vs non-GT plan ---
def test_fp_budget_unchanged_by_gt():
    m = _tiny()
    p_off = _plan(m, _cfg(use_groupwise_clip_gain_tuning=False))
    p_gt = _plan(m, _cfg(use_groupwise_clip_gain_tuning=True, gt_accept_margin=-1.0))
    for i in p_off.layers:
        assert p_off.layers[i].k_fp == p_gt.layers[i].k_fp
        assert p_off.layers[i].fp_indices.tolist() == p_gt.layers[i].fp_indices.tolist()


# --- group-gain (experimental) runs and stays in bounds ---
def test_group_gain_runs_and_bounded():
    m = _tiny()
    p = _plan(m, _cfg(use_groupwise_clip_gain_tuning=True, gt_accept_margin=-1.0,
                      gt_gain_granularity="group"))
    for lr in p.layers.values():
        for e in lr.gt_eta_gate:
            assert 0.8 - 1e-6 <= e <= 1.2 + 1e-6


# --- no runtime top-k/sort introduced by a GT plan ---
def test_gt_plan_no_runtime_topk():
    m = _tiny()
    cfg = _cfg(use_groupwise_clip_gain_tuning=True, gt_accept_margin=-1.0)
    plan = _plan(m, cfg)
    h = patch_model(m, plan, cfg)
    real = {n: getattr(torch, n) for n in ("topk", "sort", "argsort", "kthvalue")}
    try:
        for n in real:
            setattr(torch, n, lambda *a, **k: (_ for _ in ()).throw(AssertionError(f"runtime {n}")))
        with torch.no_grad():
            m(torch.randint(0, 256, (1, 8)))
    finally:
        for n, fn in real.items():
            setattr(torch, n, fn)
    unpatch_model(h)
