"""SADND-CAP model-integration tests on a tiny Llama."""

import pytest
import torch
import torch.nn as nn

from qlot_rms.config import QLotRmsConfig
from qlot_rms.calibration import calibrate
from qlot_rms.model_integration import (
    patch_model, unpatch_model, find_decoder_layers, QLotRmsFFN,
)


def _calib(model, tok, cfg):
    return calibrate(model, tok, cfg, device="cpu", allow_synthetic=True, batch_size=2)


def test_invalid_scope_rejected(small_config):
    with pytest.raises(ValueError):
        QLotRmsConfig(**{**small_config.to_dict(), "qlot_scope": "mlp_attn"}).validate()


def test_patch_then_unpatch_restores_exactly(tiny_model, fake_tokenizer, small_config):
    layers = find_decoder_layers(tiny_model)
    orig_norms = [l.post_attention_layernorm for l in layers]
    orig_mlps = [l.mlp for l in layers]
    plan = _calib(tiny_model, fake_tokenizer, small_config)
    handle = patch_model(tiny_model, plan, small_config)
    for l in layers:
        assert isinstance(l.post_attention_layernorm, nn.Identity)
        assert isinstance(l.mlp, QLotRmsFFN)
    unpatch_model(handle)
    for l, n, m in zip(layers, orig_norms, orig_mlps):
        assert l.post_attention_layernorm is n and l.mlp is m


def test_down_proj_not_routed(tiny_model, fake_tokenizer, small_config):
    layers = find_decoder_layers(tiny_model)
    orig_down = [l.mlp.down_proj for l in layers]
    plan = _calib(tiny_model, fake_tokenizer, small_config)
    handle = patch_model(tiny_model, plan, small_config)
    for l, d in zip(layers, orig_down):
        assert l.mlp.down_proj is d
    hidden = tiny_model.config.hidden_size
    for lr in plan.layers.values():
        assert lr.num_channels == hidden
        assert lr.num_channels != tiny_model.config.intermediate_size
    unpatch_model(handle)


def test_no_runtime_topk_or_sort(tiny_model, fake_tokenizer, small_config):
    plan = _calib(tiny_model, fake_tokenizer, small_config)
    handle = patch_model(tiny_model, plan, small_config)
    real = {n: getattr(torch, n) for n in ("topk", "sort", "argsort", "kthvalue")}

    def _forbid(name):
        def _f(*a, **k):
            raise AssertionError(f"runtime {name} during inference")
        return _f
    ids = torch.randint(0, tiny_model.config.vocab_size, (1, 12))
    try:
        for n in real:
            setattr(torch, n, _forbid(n))
        with torch.no_grad():
            out = tiny_model(ids)
        assert torch.isfinite(out.logits).all()
    finally:
        for n, fn in real.items():
            setattr(torch, n, fn)
    unpatch_model(handle)


def test_routing_static_and_shape(tiny_model, fake_tokenizer, small_config):
    plan = _calib(tiny_model, fake_tokenizer, small_config)
    snap = {i: lr.perm.clone() for i, lr in plan.layers.items()}
    ids = torch.randint(0, tiny_model.config.vocab_size, (1, 12))
    with torch.no_grad():
        base = tiny_model(ids).logits.clone()
    handle = patch_model(tiny_model, plan, small_config)
    with torch.no_grad():
        out = tiny_model(ids).logits.clone()
        tiny_model(torch.randint(0, 256, (2, 7)))
    for i, lr in plan.layers.items():
        assert torch.equal(lr.perm, snap[i])
    assert out.shape == base.shape and torch.isfinite(out).all()
    unpatch_model(handle)
    with torch.no_grad():
        restored = tiny_model(ids).logits
    assert torch.allclose(restored, base, atol=1e-5)
