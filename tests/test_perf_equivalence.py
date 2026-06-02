"""Regression tests: optimized torch_reference == original reference path.

These lock in that the performance optimizations (vectorized GroupRMS, precomputed
affine slices, cached fake-quantized W8-G128 weight) do NOT change numerics.
"""

import torch
import torch.nn as nn

from qlot_rms.config import QLotRmsConfig, LayerRouting
from qlot_rms.grouprms import group_rms, group_rms_loop, group_sizes_for, broadcast_per_group_to_channels
from qlot_rms.quant import simulated_int8_matmul
from qlot_rms.projection import (
    PackedProjection, compute_branch_inputs, branch_inputs_from_slices,
)
from qlot_rms.sadnd import assign_channels


def _routing(C, fp_ratio=0.06, group_size=16):
    d = torch.rand(C)
    fp, intc, perm, mask = assign_channels(d, fp_ratio)
    gs = group_sizes_for(intc.numel(), group_size)
    mu_groups = [1.0 + 0.1 * g for g in range(len(gs))]
    mu_chan = broadcast_per_group_to_channels(mu_groups, intc.numel(), group_size)
    return LayerRouting(
        layer_index=0, num_channels=C, k_fp=fp.numel(),
        fp_indices=fp, int_indices=intc, perm=perm, mask=mask, delta_tilde=d,
        grms_group_size=group_size, grms_num_groups=len(gs), grms_group_sizes=gs,
        mu_g=torch.tensor(mu_groups), mu_g_channels=mu_chan,
        act_scales=torch.rand(intc.numel()) + 0.05,
    )


def test_group_rms_vectorized_equals_loop():
    torch.manual_seed(0)
    for c, gsz in [(256, 128), (300, 128), (94, 128), (61, 16), (128, 128), (5, 8)]:
        u = torch.randn(13, c)
        a = group_rms(u, gsz, eps=1e-6)
        b = group_rms_loop(u, gsz, eps=1e-6)
        assert torch.allclose(a, b, atol=1e-6), (c, gsz)
        # return_r variant too
        a2, ra = group_rms(u, gsz, eps=1e-6, return_r=True)
        b2, rb = group_rms_loop(u, gsz, eps=1e-6, return_r=True)
        assert torch.allclose(ra, rb, atol=1e-6)


def test_branch_inputs_fast_equals_full():
    torch.manual_seed(1)
    C = 80
    r = _routing(C)
    u = torch.randn(4, 7, C)
    gamma = torch.randn(C)
    beta = torch.randn(C)
    y_F1, y_I1 = compute_branch_inputs(
        u, r.fp_indices, r.int_indices, gamma, beta, r.grms_group_size, 1e-6)
    gF = gamma.index_select(0, r.fp_indices)
    gI = gamma.index_select(0, r.int_indices)
    bF = beta.index_select(0, r.fp_indices)
    bI = beta.index_select(0, r.int_indices)
    y_F2, y_I2 = branch_inputs_from_slices(
        u, r.fp_indices, r.int_indices, gF, gI, bF, bI, r.grms_group_size, 1e-6)
    assert torch.allclose(y_F1, y_F2, atol=1e-6)
    assert torch.allclose(y_I1, y_I2, atol=1e-6)


def test_cached_dequant_equals_uncached():
    torch.manual_seed(2)
    C = 96
    r = _routing(C)
    lin = nn.Linear(C, 40, bias=True)
    gamma = torch.randn(C)
    cfg_cache = QLotRmsConfig(enable_qlot_rms=True, cache_dequant_weight=True)
    cfg_nocache = QLotRmsConfig(enable_qlot_rms=True, cache_dequant_weight=False)
    pp_c = PackedProjection.from_linear(lin, r, gamma, None, cfg_cache)
    pp_n = PackedProjection.from_linear(lin, r, gamma, None, cfg_nocache)
    assert pp_c.W_I_dq is not None and pp_n.W_I_dq is None
    u = torch.randn(3, 11, C)
    out_c = pp_c(u)
    out_n = pp_n(u)
    # caching only moves WHEN the static weight is fake-quantized -> identical
    assert torch.equal(out_c, out_n)


def test_cached_path_matches_simulated_matmul():
    # the cached int branch == simulated_int8_matmul on the same inputs
    torch.manual_seed(3)
    C = 64
    r = _routing(C)
    lin = nn.Linear(C, 24, bias=False)
    gamma = torch.ones(C)
    cfg = QLotRmsConfig(enable_qlot_rms=True, cache_dequant_weight=True)
    pp = PackedProjection.from_linear(lin, r, gamma, None, cfg)
    y_I = torch.randn(5, r.int_indices.numel())
    ref = simulated_int8_matmul(y_I, pp.W_I, pp.act_scales,
                                w_group_size=cfg.w8_group_size, qmax=cfg.qmax)
    from qlot_rms.quant import quantize_activation_int8
    yq = quantize_activation_int8(y_I, pp.act_scales, qmax=cfg.qmax).float()
    cached = torch.matmul(yq, pp.W_I_dq.t()).to(y_I.dtype)
    assert torch.allclose(ref, cached, atol=1e-4)


def _tiny_plan_and_model(cache):
    from transformers import LlamaConfig, LlamaForCausalLM
    from qlot_rms.calibration import calibrate
    torch.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, tie_word_embeddings=True)).eval()

    class _Tok:
        vocab_size = 256
    cfg = QLotRmsConfig(enable_qlot_rms=True, grms_group_size=16,
                        calibration_samples=8, calibration_seq_len=16,
                        num_calib_subsets=3, subset_size=4,
                        act_scale_max_tokens=256, cache_dequant_weight=cache)
    plan = calibrate(m, _Tok(), cfg, device="cpu", routing_method="sadnd",
                     allow_synthetic=True, batch_size=2)
    return m, plan, cfg


def test_patched_forward_cache_equals_nocache_tiny():
    from qlot_rms.model_integration import patch_model, unpatch_model
    ids = torch.randint(0, 256, (1, 12))

    m1, p1, c1 = _tiny_plan_and_model(cache=True)
    h1 = patch_model(m1, p1, c1)
    with torch.no_grad():
        out1 = m1(ids).logits.clone()
    unpatch_model(h1)

    m2, p2, c2 = _tiny_plan_and_model(cache=False)
    h2 = patch_model(m2, p2, c2)
    with torch.no_grad():
        out2 = m2(ids).logits.clone()
    unpatch_model(h2)

    assert torch.allclose(out1, out2, atol=1e-4)


def test_matmul_shared_equals_forward_from_branches():
    # shared (quantize-once) path must equal the per-projection cached path
    torch.manual_seed(7)
    from qlot_rms.quant import quantize_activation_int8
    C = 96
    r = _routing(C)
    cfg = QLotRmsConfig(enable_qlot_rms=True, cache_dequant_weight=True)
    gamma = torch.randn(C)
    lin_g = nn.Linear(C, 40, bias=True)
    lin_u = nn.Linear(C, 40, bias=False)
    pg = PackedProjection.from_linear(lin_g, r, gamma, None, cfg)
    pu = PackedProjection.from_linear(lin_u, r, gamma, None, cfg)
    u = torch.randn(2, 9, C)
    y_F, y_I = branch_inputs_from_slices(
        u, pg.fp_indices, pg.int_indices, pg.gamma_F, pg.gamma_I,
        pg.beta_F, pg.beta_I, r.grms_group_size, 1e-6, use_grms=pg.use_grms)
    # per-projection
    g1 = pg.forward_from_branches(y_F, y_I)
    u1 = pu.forward_from_branches(y_F, y_I)
    # shared
    y_F16 = y_F.to(torch.float16)
    yq = quantize_activation_int8(y_I, pg.act_scales, qmax=pg.qmax).float()
    g2 = pg.matmul_shared(y_F16, yq)
    u2 = pu.matmul_shared(y_F16, yq)
    assert torch.equal(g1, g2)
    assert torch.equal(u1, u2)


def test_use_grms_false_skips_group_rms_call():
    # routing-only fast path must NOT call group_rms at all
    import qlot_rms.projection as projmod
    C = 64
    r = _routing(C)
    cfg = QLotRmsConfig(enable_qlot_rms=True, use_grms=False, cache_dequant_weight=True)
    lin = nn.Linear(C, 16, bias=False)
    pp = PackedProjection.from_linear(lin, r, torch.ones(C), None, cfg)
    real = projmod.group_rms

    def _boom(*a, **k):
        raise AssertionError("group_rms called on routing-only path")
    try:
        projmod.group_rms = _boom
        out = pp(torch.randn(3, C))   # use_grms False -> must skip group_rms
        assert torch.isfinite(out).all()
    finally:
        projmod.group_rms = real


def test_no_runtime_sort_topk_with_cache():
    from qlot_rms.model_integration import patch_model, unpatch_model
    m, p, c = _tiny_plan_and_model(cache=True)
    handle = patch_model(m, p, c)
    real = {n: getattr(torch, n) for n in ("topk", "sort", "argsort", "kthvalue")}

    def _forbid(name):
        def _f(*a, **k):
            raise AssertionError(f"runtime {name} during inference")
        return _f
    try:
        for n in real:
            setattr(torch, n, _forbid(n))
        with torch.no_grad():
            m(torch.randint(0, 256, (1, 10)))
    finally:
        for n, fn in real.items():
            setattr(torch, n, fn)
    unpatch_model(handle)
