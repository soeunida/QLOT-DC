"""Per-layer GroupRMS gating tests (proxy-based)."""

import torch

from qlot_rms.config import QLotRmsConfig
from qlot_rms.calibration import calibrate
from qlot_rms.model_integration import patch_model, unpatch_model


class _Tok:
    vocab_size = 256


def _tiny():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, tie_word_embeddings=True)).eval()


def _cfg(**kw):
    base = dict(enable_qlot_rms=True, grms_group_size=16, calibration_samples=8,
                calibration_seq_len=16, num_calib_subsets=3, subset_size=4,
                act_scale_max_tokens=256)
    base.update(kw)
    return QLotRmsConfig(**base)


def test_gating_populates_metadata():
    m = _tiny()
    cfg = _cfg(use_grms=True, use_mean_comp=True, grms_gating=True)
    plan = calibrate(m, _Tok(), cfg, device="cpu", routing_method="sadnd",
                     allow_synthetic=True, batch_size=2)
    for lr in plan.layers.values():
        assert isinstance(lr.grms_enabled, bool)
        assert lr.grms_gate_reason
        # proxy errors are recorded when gating runs
        assert lr.grms_proxy_err_ptq is not None
        assert lr.grms_proxy_err_grms is not None
        # mean-comp is only applied where GroupRMS is enabled
        assert lr.mean_comp_applied == lr.grms_enabled


def test_gating_disabled_is_global_grms():
    m = _tiny()
    cfg = _cfg(use_grms=True, use_mean_comp=True, grms_gating=False)
    plan = calibrate(m, _Tok(), cfg, device="cpu", routing_method="sadnd",
                     allow_synthetic=True, batch_size=2)
    for lr in plan.layers.values():
        assert lr.grms_enabled is True
        assert lr.grms_proxy_err_ptq is None  # proxy not run


def test_routing_only_sets_grms_disabled():
    m = _tiny()
    cfg = _cfg(use_grms=False, use_mean_comp=False)
    plan = calibrate(m, _Tok(), cfg, device="cpu", routing_method="sadnd",
                     allow_synthetic=True, batch_size=2)
    for lr in plan.layers.values():
        assert lr.grms_enabled is False
        assert lr.mean_comp_applied is False


def test_gated_model_forward_runs_and_restores():
    m = _tiny()
    cfg = _cfg(use_grms=True, use_mean_comp=True, grms_gating=True)
    ids = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        base = m(ids).logits.clone()
    plan = calibrate(m, _Tok(), cfg, device="cpu", routing_method="sadnd",
                     allow_synthetic=True, batch_size=2)
    handle = patch_model(m, plan, cfg)
    with torch.no_grad():
        out = m(ids).logits.clone()
    unpatch_model(handle)
    assert out.shape == base.shape and torch.isfinite(out).all()
    with torch.no_grad():
        restored = m(ids).logits
    assert torch.allclose(restored, base, atol=1e-5)
