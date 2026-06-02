"""Serving export, readiness checker, and custom_packed (experimental) tests."""

import os

import pytest
import torch
import torch.nn as nn

from qlot_rms.config import QLotRmsConfig
from qlot_rms.calibration import calibrate
from qlot_rms.projection import (
    PackedProjection, CustomPackedBackend, custom_packed_forward, get_backend,
    branch_inputs_from_slices,
)
from qlot_rms.kernels.triton_packed import triton_available


class _Tok:
    vocab_size = 256


def _tiny(n=2):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=n, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, tie_word_embeddings=True)).eval()


def _dc_cfg(**kw):
    base = dict(enable_qlot_rms=True, method="qlot_dc", grms_group_size=16,
                use_grms=False, use_static_diag_comp=True, diag_comp_mode="median_scale",
                calibration_samples=8, calibration_seq_len=16, num_calib_subsets=3,
                subset_size=4, act_scale_max_tokens=256)
    base.update(kw)
    return QLotRmsConfig(**base)


def _plan(model, cfg):
    return calibrate(model, _Tok(), cfg, device="cpu", routing_method="sadnd",
                     allow_synthetic=True, batch_size=2)


# --- serving export ---
def test_serving_export_shapes(tmp_path):
    from qlot_rms.serving_export import export_serving_artifacts
    m = _tiny(); cfg = _dc_cfg()
    plan = _plan(m, cfg)
    manifest = export_serving_artifacts(m, plan, cfg, str(tmp_path))
    assert set(manifest["layers"]) == set(int(i) for i in plan.layers)
    for i, lr in plan.layers.items():
        ldir = tmp_path / f"layer_{i}"
        for fn in ("indices.pt", "scales.pt", "gate.pt", "up.pt", "meta.json"):
            assert (ldir / fn).exists()
        idx = torch.load(ldir / "indices.pt")
        assert idx["fp_indices"].numel() == lr.k_fp
        assert idx["int_indices"].numel() == lr.int_indices.numel()
        sc = torch.load(ldir / "scales.pt")
        assert sc["act_scales"].numel() == lr.int_indices.numel()
        assert sc["diag_alpha"].numel() == lr.int_indices.numel()  # DC enabled
        gate = torch.load(ldir / "gate.pt")
        O = m.config.intermediate_size
        assert gate["W_F"].shape == (O, lr.k_fp)
        assert gate["W_I"].shape == (O, lr.int_indices.numel())
    assert os.path.exists(tmp_path / "manifest.json")


# --- readiness checker ---
def test_readiness_checker_passes_for_dc_plan(tmp_path):
    from eval.check_custom_packed_readiness import check_layer
    m = _tiny(); cfg = _dc_cfg()
    plan = _plan(m, cfg)
    from qlot_rms.model_integration import find_decoder_layers
    layers = find_decoder_layers(m)
    for i, lr in plan.layers.items():
        ok, checks = check_layer(lr, layers[i])
        assert ok, checks
        assert checks["diag_alpha_present"] is True
        assert checks["diag_alpha_len_eq_k_int"] is True


# --- custom_packed: clear NotImplementedError when no kernel ---
def test_custom_packed_fp_int_matmul_raise():
    be = get_backend("custom_packed")
    assert isinstance(be, CustomPackedBackend)
    assert be.experimental is True
    with pytest.raises(NotImplementedError):
        be.fp_matmul(torch.randn(2, 4), torch.randn(8, 4))
    with pytest.raises(NotImplementedError):
        be.int_matmul(torch.randn(2, 4), torch.randn(8, 4), torch.ones(4), 128, 127)


def test_custom_packed_forward_raises_without_kernel():
    if triton_available():
        pytest.skip("Triton+CUDA available; covered by the correctness test")
    with pytest.raises(NotImplementedError):
        custom_packed_forward(
            x_fp=torch.randn(3, 4), x_int=torch.randn(3, 8),
            w_fp=torch.randn(16, 4), w_int_packed=torch.randn(16, 8),
            act_scales=torch.ones(8))


# --- optional Triton correctness (skips on CPU/no-Triton) ---
@pytest.mark.skipif(not triton_available(), reason="Triton+CUDA unavailable")
def test_triton_packed_matches_reference():
    torch.manual_seed(0)
    dev = "cuda"
    C, O = 96, 48
    from qlot_rms.sadnd import assign_channels
    from qlot_rms.config import LayerRouting
    from qlot_rms.grouprms import group_sizes_for, broadcast_per_group_to_channels
    d = torch.rand(C)
    fp, intc, perm, mask = assign_channels(d, 0.06)
    gs = group_sizes_for(intc.numel(), 128)
    alpha = (torch.rand(intc.numel()) + 0.5)
    r = LayerRouting(layer_index=0, num_channels=C, k_fp=fp.numel(),
        fp_indices=fp, int_indices=intc, perm=perm, mask=mask, delta_tilde=d,
        grms_group_size=128, grms_num_groups=len(gs), grms_group_sizes=gs,
        mu_g=torch.ones(len(gs)), act_scales=torch.rand(intc.numel()) + 0.05,
        diag_comp_applied=True, diag_alpha=alpha)
    cfg = QLotRmsConfig(enable_qlot_rms=True, method="qlot_dc", use_grms=False,
                        use_static_diag_comp=True, diag_comp_mode="median_scale")
    lin = nn.Linear(C, O, bias=True).to(dev).half()
    pp = PackedProjection.from_linear(lin, r, torch.ones(C).to(dev), None, cfg).to(dev)
    u = torch.randn(10, C, device=dev)
    ref = pp(u).float()
    # build branch inputs the same way, then run the custom packed kernel
    y_F, y_I = branch_inputs_from_slices(
        u, pp.fp_indices, pp.int_indices, pp.gamma_F, pp.gamma_I,
        pp.beta_F, pp.beta_I, r.grms_group_size, cfg.eps,
        use_grms=False, int_alpha=pp.diag_alpha)
    got = custom_packed_forward(
        x_fp=y_F, x_int=y_I, w_fp=pp.W_F, w_int_packed=pp.W_I_dq,
        act_scales=pp.act_scales, alpha=pp.diag_alpha, bias=pp.bias,
        metadata={"qmax": cfg.qmax}).float()
    rel = (got - ref).norm() / (ref.norm() + 1e-8)
    assert rel < 1e-2, f"rel err {rel}"


# --- torch_reference invariance guard ---
def test_torch_reference_default_and_deterministic():
    assert get_backend("torch_reference").name == "torch_reference"
    m = _tiny(); cfg = _dc_cfg()
    plan = _plan(m, cfg)
    from qlot_rms.model_integration import patch_model, unpatch_model
    ids = torch.randint(0, 256, (1, 10))
    h = patch_model(m, plan, cfg)
    with torch.no_grad():
        a = m(ids).logits.clone(); b = m(ids).logits.clone()
    unpatch_model(h)
    assert torch.allclose(a, b)  # reference path deterministic / unchanged
