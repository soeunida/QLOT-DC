"""Q-LOT-DC tests: static diagonal compensation, error-bounded FP budget,
projection bias correction."""

import pytest
import torch
import torch.nn as nn

from qlot_rms.config import QLotRmsConfig, LayerRouting
from qlot_rms.diagcomp import compute_alpha, apply_inverse_weight_scale
from qlot_rms.grouprms import group_sizes_for
from qlot_rms.projection import PackedProjection
from qlot_rms.sadnd import assign_channels


def _routing_dc(C, k_int=None, fp_ratio=0.06, group_size=128, alpha=None):
    d = torch.rand(C)
    fp, intc, perm, mask = assign_channels(d, fp_ratio)
    ki = intc.numel()
    if alpha is None:
        alpha = torch.rand(ki) * 2 + 0.5
    gs = group_sizes_for(ki, group_size)
    return LayerRouting(
        layer_index=0, num_channels=C, k_fp=fp.numel(),
        fp_indices=fp, int_indices=intc, perm=perm, mask=mask, delta_tilde=d,
        grms_group_size=group_size, grms_num_groups=len(gs), grms_group_sizes=gs,
        mu_g=torch.ones(len(gs)), act_scales=torch.ones(ki),
        diag_comp_applied=True, diag_alpha=alpha,
    )


# --- compute_alpha ---
def test_alpha_clamped_and_finite():
    a = torch.tensor([1e-7, 1.0, 1e7, 0.3, 5.0])
    alpha = compute_alpha("median_scale", a, alpha_min=0.25, alpha_max=4.0)
    assert alpha.shape == a.shape
    assert torch.isfinite(alpha).all()
    assert bool((alpha >= 0.25).all()) and bool((alpha <= 4.0).all())


def test_alpha_smoothquant_clamped():
    a = torch.rand(50) + 0.1
    w = torch.rand(50) + 0.1
    alpha = compute_alpha("smoothquant_like", a, w, beta=0.5,
                          alpha_min=0.25, alpha_max=4.0)
    assert alpha.numel() == 50
    assert bool((alpha >= 0.25).all()) and bool((alpha <= 4.0).all())
    assert torch.isfinite(alpha).all()


# --- inverse weight scaling orientation ---
def test_inverse_weight_scale_only_int_columns():
    C = 80
    r = _routing_dc(C)
    W = torch.randn(32, C)
    Wc = apply_inverse_weight_scale(W, r.int_indices, r.diag_alpha)
    assert torch.allclose(Wc[:, r.int_indices], W[:, r.int_indices] * (1.0 / r.diag_alpha))
    assert torch.allclose(Wc[:, r.fp_indices], W[:, r.fp_indices])
    assert Wc.shape == W.shape


# --- function preservation BEFORE quantization ---
def test_dc_preserves_projection_function_pre_quant():
    C, O = 96, 40
    r = _routing_dc(C)
    W_I = torch.randn(O, r.int_indices.numel())
    y_base = torch.randn(7, r.int_indices.numel())
    # diagonal similarity: (alpha*y) @ (W/alpha)^T == y @ W^T
    lhs = (y_base * r.diag_alpha) @ (W_I * (1.0 / r.diag_alpha)).t()
    rhs = y_base @ W_I.t()
    assert torch.allclose(lhs, rhs, atol=1e-4)


# --- packed projection folds 1/alpha into W_I and keeps shape ---
def test_packed_dc_weight_and_shape():
    C = 96
    r = _routing_dc(C)
    cfg = QLotRmsConfig(enable_qlot_rms=True, method="qlot_dc",
                        use_grms=False, use_static_diag_comp=True,
                        diag_comp_mode="median_scale")
    lin = nn.Linear(C, 33, bias=True)
    pp = PackedProjection.from_linear(lin, r, torch.ones(C), None, cfg)
    # W_I == W[:, int] / alpha
    expect = (lin.weight.data[:, r.int_indices] * (1.0 / r.diag_alpha)).to(torch.float16)
    assert torch.allclose(pp.W_I.float(), expect.float(), atol=1e-2)
    # FP slice unchanged
    assert torch.allclose(pp.W_F.float(),
                          lin.weight.data[:, r.fp_indices].to(torch.float16).float(), atol=1e-2)
    # alpha stored, length k_int
    assert pp.diag_alpha is not None and pp.diag_alpha.numel() == r.int_indices.numel()
    # output shape preserved
    u = torch.randn(2, 5, C)
    assert pp(u).shape == lin(u).shape == (2, 5, 33)


def test_dc_does_not_call_group_rms():
    import qlot_rms.projection as projmod
    C = 64
    r = _routing_dc(C)
    cfg = QLotRmsConfig(enable_qlot_rms=True, use_grms=False,
                        use_static_diag_comp=True, diag_comp_mode="median_scale")
    lin = nn.Linear(C, 16, bias=False)
    pp = PackedProjection.from_linear(lin, r, torch.ones(C), None, cfg)
    real = projmod.group_rms

    def _boom(*a, **k):
        raise AssertionError("group_rms called under Q-LOT-DC")
    try:
        projmod.group_rms = _boom
        assert torch.isfinite(pp(torch.randn(3, C))).all()
    finally:
        projmod.group_rms = real


# --- calibration-level: alpha shape, error budget, metadata ---
class _Tok:
    vocab_size = 256


def _tiny(n_layers=3):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, tie_word_embeddings=True)).eval()


def _cfg(**kw):
    base = dict(enable_qlot_rms=True, grms_group_size=16, calibration_samples=8,
                calibration_seq_len=16, num_calib_subsets=3, subset_size=4,
                act_scale_max_tokens=256)
    base.update(kw)
    return QLotRmsConfig(**base)


def test_calib_dc_alpha_len_equals_k_int():
    from qlot_rms.calibration import calibrate
    m = _tiny()
    cfg = _cfg(method="qlot_dc", use_grms=False, use_static_diag_comp=True,
               diag_comp_mode="median_scale")
    plan = calibrate(m, _Tok(), cfg, device="cpu", routing_method="sadnd",
                     allow_synthetic=True, batch_size=2)
    for lr in plan.layers.values():
        assert lr.diag_comp_applied is True
        assert lr.diag_alpha is not None
        assert lr.diag_alpha.numel() == lr.int_indices.numel()
        assert bool((lr.diag_alpha >= cfg.diag_comp_alpha_min - 1e-6).all())
        assert bool((lr.diag_alpha <= cfg.diag_comp_alpha_max + 1e-6).all())
        assert lr.grms_enabled is False  # DC replaces GroupRMS


def test_error_bounded_budget_selects_valid_candidate():
    from qlot_rms.calibration import calibrate
    m = _tiny()
    cands = [0.0, 0.02, 0.06, 0.10]
    cfg = _cfg(use_static_diag_comp=True, diag_comp_mode="median_scale",
               fp_budget_mode="error_bounded", error_bound_metric="activation_mse",
               error_bound=0.01, fp_ratio_candidates=cands)
    plan = calibrate(m, _Tok(), cfg, device="cpu", routing_method="sadnd",
                     allow_synthetic=True, batch_size=2)
    for lr in plan.layers.values():
        assert lr.selected_fp_ratio in cands
        assert lr.fp_budget_errors is not None and len(lr.fp_budget_errors) == len(cands)
        # k_fp consistent with the chosen ratio
        assert lr.k_fp == int(lr.selected_fp_ratio * lr.num_channels)


def test_bias_correction_vectors_saved_and_runs():
    from qlot_rms.calibration import calibrate
    from qlot_rms.model_integration import patch_model, unpatch_model
    m = _tiny()
    cfg = _cfg(use_static_diag_comp=True, diag_comp_mode="median_scale",
               use_projection_bias_correction=True)
    plan = calibrate(m, _Tok(), cfg, device="cpu", routing_method="sadnd",
                     allow_synthetic=True, batch_size=2)
    inter = m.config.intermediate_size
    for lr in plan.layers.values():
        assert lr.bias_corr_gate is not None and lr.bias_corr_gate.numel() == inter
        assert lr.bias_corr_up is not None and lr.bias_corr_up.numel() == inter
    ids = torch.randint(0, 256, (1, 10))
    h = patch_model(m, plan, cfg)
    with torch.no_grad():
        out = m(ids).logits
    unpatch_model(h)
    assert torch.isfinite(out).all()


def test_output_aware_sadnd_now_implemented():
    # output_aware_sadnd was a stub in plain Q-LOT-DC; Q-LOT-DC+ implements it.
    from qlot_rms.calibration import calibrate
    m = _tiny()
    cfg = _cfg(routing_score="output_aware_sadnd")
    plan = calibrate(m, _Tok(), cfg, device="cpu", routing_method="output_aware_sadnd",
                     allow_synthetic=True, batch_size=2)
    for lr in plan.layers.values():
        assert lr.perm.tolist() == lr.fp_indices.tolist() + lr.int_indices.tolist()
        assert lr.k_fp == int(cfg.fp_ratio * lr.num_channels)
