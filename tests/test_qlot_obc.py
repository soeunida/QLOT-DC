"""Q-LOT-OBC integration tests (block-output correction end-to-end)."""

import torch

from qlot_rms.config import QLotRmsConfig
from qlot_rms.calibration import calibrate
from qlot_rms.model_integration import patch_model, unpatch_model


class _Tok:
    vocab_size = 256


def _tiny(n=2):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=n, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, tie_word_embeddings=True)).eval()


def _cfg(mode, margin, **kw):
    base = dict(enable_qlot_rms=True, method="qlot_obc", grms_group_size=16,
                use_grms=False, calibration_samples=8, calibration_seq_len=16,
                num_calib_subsets=3, subset_size=4, act_scale_max_tokens=256,
                use_block_output_correction=True, block_correction_mode=mode,
                block_correction_margin=margin, block_correction_max_tokens=128,
                block_lowrank_rank=2)
    base.update(kw)
    return QLotRmsConfig(**base)


def _plan(m, cfg):
    return calibrate(m, _Tok(), cfg, device="cpu", routing_method="sadnd",
                     allow_synthetic=True, batch_size=2)


def test_obc_metadata_populated():
    m = _tiny()
    cfg = _cfg("affine", margin=-1.0)   # force accept to populate tensors
    plan = _plan(m, cfg)
    for lr in plan.layers.values():
        assert lr.block_corr_mode == "affine"
        assert lr.block_corr_enabled is True
        assert lr.block_affine_a is not None and lr.block_affine_b is not None
        assert lr.block_affine_a.numel() == m.config.hidden_size
        assert lr.block_mse_before is not None and lr.block_mse_after is not None


def test_obc_output_shape_and_finite():
    m = _tiny()
    cfg = _cfg("lowrank", margin=-1.0)
    plan = _plan(m, cfg)
    ids = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        base_shape = m(ids).logits.shape
    h = patch_model(m, plan, cfg)
    with torch.no_grad():
        out = m(ids).logits
    unpatch_model(h)
    assert out.shape == base_shape and torch.isfinite(out).all()


def test_obc_accept_gating_rejects_when_no_gain():
    # impossibly strict margin (needs 99.9% MSE reduction) -> reject everywhere
    m = _tiny()
    cfg = _cfg("affine", margin=0.999)
    plan = _plan(m, cfg)
    for lr in plan.layers.values():
        assert lr.block_corr_enabled is False
        assert lr.block_affine_a is None  # rejected -> tensors not stored


def test_obc_disabled_path_equals_baseline():
    # all layers rejected -> patched forward equals the no-block-correction plan
    ids = torch.randint(0, 256, (1, 12))
    m1 = _tiny(); cfg1 = _cfg("affine", margin=0.999)            # all rejected
    p1 = _plan(m1, cfg1); h1 = patch_model(m1, p1, cfg1)
    with torch.no_grad():
        o1 = m1(ids).logits.clone()
    unpatch_model(h1)

    m2 = _tiny()
    cfg2 = QLotRmsConfig(enable_qlot_rms=True, method="qlot_obc", grms_group_size=16,
                         use_grms=False, calibration_samples=8, calibration_seq_len=16,
                         num_calib_subsets=3, subset_size=4, act_scale_max_tokens=256,
                         use_block_output_correction=False)
    p2 = _plan(m2, cfg2); h2 = patch_model(m2, p2, cfg2)
    with torch.no_grad():
        o2 = m2(ids).logits.clone()
    unpatch_model(h2)
    assert torch.allclose(o1, o2, atol=1e-5)


def test_validation_ppl_gate_runs_and_decides():
    # the global validation_ppl safeguard must run end-to-end and record a decision
    m = _tiny()
    cfg = _cfg("bias", margin=0.0, block_correction_accept_rule="validation_ppl")
    plan = _plan(m, cfg)
    reasons = [lr.block_corr_reason for lr in plan.layers.values()]
    assert any("validation_ppl" in r or "accept" in r or "reject" in r for r in reasons)
    # forward must still run and be finite regardless of the gate decision
    ids = torch.randint(0, 256, (1, 10))
    h = patch_model(m, plan, cfg)
    with torch.no_grad():
        out = m(ids).logits
    unpatch_model(h)
    assert torch.isfinite(out).all()


def test_obc_no_runtime_sort_topk():
    m = _tiny()
    cfg = _cfg("bias", margin=-1.0)
    plan = _plan(m, cfg)
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
