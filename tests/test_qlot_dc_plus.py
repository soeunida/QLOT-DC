"""Q-LOT-DC+ tests: output-aware routing, bias correction, low-rank correction."""

import pytest
import torch
import torch.nn as nn

from qlot_rms.config import QLotRmsConfig
from qlot_rms.lowrank import fit_lowrank, apply_lowrank
from qlot_rms.sadnd import assign_channels


class _Tok:
    vocab_size = 256


def _tiny(n=2):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=n, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, tie_word_embeddings=True)).eval()


def _cfg(**kw):
    base = dict(enable_qlot_rms=True, method="qlot_dc_plus", grms_group_size=16,
                use_grms=False, use_static_diag_comp=True, diag_comp_mode="median_scale",
                calibration_samples=8, calibration_seq_len=16, num_calib_subsets=3,
                subset_size=4, act_scale_max_tokens=256, lowrank_max_tokens=64)
    base.update(kw)
    return QLotRmsConfig(**base)


def _calib(m, cfg, rm="sadnd"):
    from qlot_rms.calibration import calibrate
    return calibrate(m, _Tok(), cfg, device="cpu", routing_method=rm,
                     allow_synthetic=True, batch_size=2)


# --- output-aware routing ---
def test_output_aware_score_formula_and_shape():
    # score_c = delta_c * (||W_gate[:,c]||2 + ||W_up[:,c]||2), nn.Linear [out,in]
    C, O = 32, 20
    torch.manual_seed(0)
    delta = torch.rand(C)
    Wg = torch.randn(O, C)
    Wu = torch.randn(O, C)
    score = delta * (Wg.norm(dim=0) + Wu.norm(dim=0))
    assert score.shape == (C,)
    # scaling one channel's weight changes its score (and can change its rank)
    Wg2 = Wg.clone(); Wg2[:, 5] *= 50.0
    score2 = delta * (Wg2.norm(dim=0) + Wu.norm(dim=0))
    assert not torch.allclose(score, score2)
    assert score2[5] > score[5]


def test_output_aware_routing_runs_and_static(monkeypatch):
    m = _tiny()
    cfg = _cfg()
    plan = _calib(m, cfg, rm="output_aware_sadnd")
    for lr in plan.layers.values():
        assert lr.perm.tolist() == lr.fp_indices.tolist() + lr.int_indices.tolist()
    # no runtime sort/top-k under the patched forward
    from qlot_rms.model_integration import patch_model, unpatch_model
    h = patch_model(m, plan, cfg)
    real = {n: getattr(torch, n) for n in ("topk", "sort", "argsort", "kthvalue")}
    try:
        for n in real:
            setattr(torch, n, lambda *a, **k: (_ for _ in ()).throw(
                AssertionError(f"runtime {n}")))
        with torch.no_grad():
            m(torch.randint(0, 256, (1, 8)))
    finally:
        for n, fn in real.items():
            setattr(torch, n, fn)
    unpatch_model(h)


def test_output_aware_can_change_fp_set():
    # with a planted huge-norm column, output-aware can route a channel to FP
    # that plain SADND (by distortion alone) would not.
    torch.manual_seed(0)
    C = 50
    delta = torch.linspace(0.1, 0.2, C)  # nearly flat distortion
    fp_sadnd = set(assign_channels(delta, 0.1)[0].tolist())
    wnorm = torch.ones(C); wnorm[3] = 1000.0
    fp_oa = set(assign_channels(delta * wnorm, 0.1)[0].tolist())
    assert 3 in fp_oa and 3 not in fp_sadnd


# --- bias correction ---
def test_bias_correction_shape_and_applied_only_when_enabled():
    m = _tiny()
    O = m.config.intermediate_size
    p_on = _calib(m, _cfg(use_projection_bias_correction=True, bias_corr_scope="gate_up"))
    for lr in p_on.layers.values():
        assert lr.bias_corr_gate is not None and lr.bias_corr_gate.numel() == O
        assert lr.bias_corr_up is not None and lr.bias_corr_up.numel() == O
    p_off = _calib(_tiny(), _cfg(use_projection_bias_correction=False))
    for lr in p_off.layers.values():
        assert lr.bias_corr_gate is None and lr.bias_corr_up is None


def test_bias_correction_reduces_mean_error_synthetic():
    torch.manual_seed(0)
    z_ref = torch.randn(100, 16)
    z_quant = z_ref + 0.3 + 0.05 * torch.randn(100, 16)  # biased estimate
    b = (z_ref - z_quant).mean(0)
    before = (z_ref - z_quant).mean(0).abs().mean()
    after = (z_ref - (z_quant + b)).mean(0).abs().mean()
    assert after < before and after < 1e-5


def test_dc_plus_output_shape_unchanged():
    m = _tiny()
    cfg = _cfg(use_projection_bias_correction=True, use_lowrank_correction=True, lowrank_rank=2)
    plan = _calib(m, cfg)
    from qlot_rms.model_integration import patch_model, unpatch_model
    ids = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        base = m(ids).logits.shape
    h = patch_model(m, plan, cfg)
    with torch.no_grad():
        out = m(ids).logits
    unpatch_model(h)
    assert out.shape == base and torch.isfinite(out).all()


# --- low-rank correction ---
def test_lowrank_factor_shapes_and_disabled_by_default():
    cfg = QLotRmsConfig()
    assert cfg.use_lowrank_correction is False
    m = _tiny()
    plan = _calib(m, _cfg(use_lowrank_correction=True, lowrank_rank=4,
                          use_projection_bias_correction=True))
    for lr in plan.layers.values():
        ci = lr.int_indices.numel(); O = m.config.intermediate_size
        assert lr.lowrank_gate_A.shape[0] == ci and lr.lowrank_gate_A.shape[1] <= 4
        assert lr.lowrank_gate_B.shape == (lr.lowrank_gate_A.shape[1], O)


def test_lowrank_reduces_projection_mse_synthetic():
    torch.manual_seed(0)
    T, C, O = 256, 40, 24
    X = torch.randn(T, C)
    true_AB = torch.randn(C, 6) @ torch.randn(6, O)
    E = X @ true_AB + 0.01 * torch.randn(T, O)   # mostly low-rank residual
    A, B = fit_lowrank(X, E, rank=6)
    pred = apply_lowrank(X, A, B)
    before = E.pow(2).mean()
    after = (E - pred).pow(2).mean()
    assert after < before * 0.5   # meaningful reduction


def test_lowrank_disabled_gives_no_correction():
    from qlot_rms.projection import PackedProjection
    from qlot_rms.config import LayerRouting
    from qlot_rms.grouprms import group_sizes_for
    C = 64
    d = torch.rand(C)
    fp, intc, perm, mask = assign_channels(d, 0.06)
    gs = group_sizes_for(intc.numel(), 128)
    r = LayerRouting(layer_index=0, num_channels=C, k_fp=fp.numel(),
        fp_indices=fp, int_indices=intc, perm=perm, mask=mask, delta_tilde=d,
        grms_group_size=128, grms_num_groups=len(gs), grms_group_sizes=gs,
        mu_g=torch.ones(len(gs)), act_scales=torch.ones(intc.numel()))
    cfg = QLotRmsConfig(enable_qlot_rms=True, use_grms=False)
    pp = PackedProjection.from_linear(nn.Linear(C, 16, bias=False), r,
                                      torch.ones(C), None, cfg)
    assert pp.lowrank_correction(torch.randn(3, intc.numel())) is None
