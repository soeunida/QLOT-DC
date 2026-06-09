"""StaticScale end-to-end on a tiny model: exact restore, static tau/eta, no runtime search."""

import torch

from staticscale import StaticScaleConfig, calibrate, patch_model, unpatch_model


class _Tok:
    vocab_size = 256


def _tiny(n=2):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=n,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
        tie_word_embeddings=True)).eval()


def _full_cfg(**kw):
    # full StaticScale stack (global budget to keep the tiny test fast), GT forced on
    base = dict(enable_qlot_rms=True, method="sadnd_cap", routing_score="output_aware_sadnd",
                int_permutation_mode="packing_aware", fp_ratio=0.25, global_fp_budget_ratio=0.25,
                fp_budget_mode="global", use_fp_mask_refinement=True, fp_refine_method="greedy_swap",
                use_groupwise_clip_gain_tuning=True, gt_accept_margin=-1.0, gt_group_size=16,
                calibration_samples=8, calibration_seq_len=16, num_calib_subsets=3, subset_size=4,
                act_scale_max_tokens=256, fp_refine_max_tokens=64, gt_max_tokens=64)
    base.update(kw)
    return StaticScaleConfig(**base)


def _plan(m, cfg):
    return calibrate(m, _Tok(), cfg, device="cpu", allow_synthetic=True, batch_size=2)


def test_patch_unpatch_restores_exactly():
    m = _tiny()
    x = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        before = m(x).logits.clone()
    cfg = _full_cfg()
    h = patch_model(m, _plan(m, cfg), cfg)
    with torch.no_grad():
        _ = m(x).logits
    unpatch_model(h)
    with torch.no_grad():
        after = m(x).logits
    assert torch.allclose(before, after, atol=0, rtol=0)     # exact restore


def test_tau_and_eta_are_static_metadata():
    m = _tiny()
    plan = _plan(m, _full_cfg())
    for lr in plan.layers.values():
        assert lr.gt_enabled is True
        # tau is folded into the frozen activation scales (static metadata present)
        assert len(lr.gt_tau_values) >= 1
        # eta is folded into the INT weight columns -> stored as a static per-INT gain
        assert lr.gt_int_gain_gate is not None and lr.gt_int_gain_up is not None
        assert lr.gt_int_gain_gate.numel() == lr.int_indices.numel()
        # equal budget preserved through refinement + clip-gain
        assert lr.k_fp == int(lr.fp_indices.numel())


def test_no_runtime_topk_or_sort():
    m = _tiny()
    cfg = _full_cfg()
    h = patch_model(m, _plan(m, cfg), cfg)
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
