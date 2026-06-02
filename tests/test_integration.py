"""Model-integration tests on a tiny Llama: patch/unpatch, routing scope,
no-runtime-routing, function shift, mlp_attn stub."""

import pytest
import torch
import torch.nn as nn

from qlot_rms.config import QLotRmsConfig
from qlot_rms.calibration import calibrate
from qlot_rms.model_integration import (
    patch_model,
    unpatch_model,
    find_decoder_layers,
    QLotRmsFFN,
)


def _calib_plan(model, tok, cfg, method="sadnd"):
    return calibrate(model, tok, cfg, device="cpu", routing_method=method,
                     allow_synthetic=True, batch_size=2)


def test_mlp_attn_scope_raises(tiny_model, fake_tokenizer, small_config):
    cfg = small_config
    plan = _calib_plan(tiny_model, fake_tokenizer, cfg)
    cfg2 = QLotRmsConfig(**{**cfg.to_dict(), "qlot_scope": "mlp_attn"})
    plan2 = type(plan)(config=cfg2, layers=plan.layers)
    with pytest.raises(NotImplementedError):
        patch_model(tiny_model, plan2, cfg2)


def test_patch_then_unpatch_restores_exactly(tiny_model, fake_tokenizer, small_config):
    layers = find_decoder_layers(tiny_model)
    orig_norms = [l.post_attention_layernorm for l in layers]
    orig_mlps = [l.mlp for l in layers]

    plan = _calib_plan(tiny_model, fake_tokenizer, small_config)
    handle = patch_model(tiny_model, plan, small_config)

    # after patch: routed layers have Identity norm + QLotRmsFFN mlp
    for l in layers:
        assert isinstance(l.post_attention_layernorm, nn.Identity)
        assert isinstance(l.mlp, QLotRmsFFN)

    unpatch_model(handle)
    for l, n, m in zip(layers, orig_norms, orig_mlps):
        assert l.post_attention_layernorm is n
        assert l.mlp is m


def test_down_proj_not_routed(tiny_model, fake_tokenizer, small_config):
    layers = find_decoder_layers(tiny_model)
    orig_down = [l.mlp.down_proj for l in layers]
    plan = _calib_plan(tiny_model, fake_tokenizer, small_config)
    handle = patch_model(tiny_model, plan, small_config)
    for l, d in zip(layers, orig_down):
        # QLotRmsFFN keeps the ORIGINAL down_proj module (not packed/routed)
        assert l.mlp.down_proj is d
    # routing mask length equals LN2 channel count (hidden_size), NOT
    # the down_proj input dim (intermediate_size)
    hidden = tiny_model.config.hidden_size
    inter = tiny_model.config.intermediate_size
    for lr in plan.layers.values():
        assert lr.num_channels == hidden
        assert lr.num_channels != inter
    unpatch_model(handle)


def test_no_runtime_topk_or_sort_in_inference(tiny_model, fake_tokenizer, small_config):
    plan = _calib_plan(tiny_model, fake_tokenizer, small_config)
    handle = patch_model(tiny_model, plan, small_config)

    # forbid sort/topk/argsort during the patched forward; if the inference
    # path tried to (re)route dynamically it would call one of these.
    real = {n: getattr(torch, n) for n in ("topk", "sort", "argsort", "kthvalue")}

    def _forbid(name):
        def _f(*a, **k):
            raise AssertionError(f"runtime {name} called during inference")
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


def test_routing_is_static_across_inputs(tiny_model, fake_tokenizer, small_config):
    plan = _calib_plan(tiny_model, fake_tokenizer, small_config)
    snap = {i: lr.perm.clone() for i, lr in plan.layers.items()}
    handle = patch_model(tiny_model, plan, small_config)
    with torch.no_grad():
        tiny_model(torch.randint(0, 256, (1, 10)))
        tiny_model(torch.randint(0, 256, (2, 7)))
    # frozen permutation unchanged after running different inputs
    for i, lr in plan.layers.items():
        assert torch.equal(lr.perm, snap[i])
    unpatch_model(handle)


def test_fp16_grms_function_shift_finite(tiny_model, fake_tokenizer, small_config):
    ids = torch.randint(0, tiny_model.config.vocab_size, (1, 12))
    with torch.no_grad():
        base = tiny_model(ids).logits.clone()

    plan = _calib_plan(tiny_model, fake_tokenizer, small_config)
    handle = patch_model(tiny_model, plan, small_config)
    with torch.no_grad():
        patched = tiny_model(ids).logits.clone()
    unpatch_model(handle)

    assert patched.shape == base.shape
    assert torch.isfinite(patched).all()
    # GroupRMS + INT8 is NOT function-preserving -> outputs shift
    assert not torch.allclose(patched, base, atol=1e-3)
    # restored model reproduces the baseline exactly
    with torch.no_grad():
        restored = tiny_model(ids).logits
    assert torch.allclose(restored, base, atol=1e-5)
