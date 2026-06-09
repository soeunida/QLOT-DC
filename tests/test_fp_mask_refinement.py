"""SADND-CAP++ equal-budget FP mask refinement tests."""

import torch

from qlot_rms.config import QLotRmsConfig
from qlot_rms.calibration import calibrate
from qlot_rms.model_integration import patch_model, unpatch_model
from qlot_rms.fp_mask_refinement import (
    build_boundary_swap_candidates, build_refine_context_from_weights,
    evaluate_mask_proxy, greedy_refine_fp_mask, beam_refine_fp_mask,
    refine_policy_masks,
)


class _Tok:
    vocab_size = 256


# --------------------------------------------------------------------------- #
# synthetic context where promoting one INT channel clearly helps
# --------------------------------------------------------------------------- #
def _toy_ctx(qmax=7, seed=0):
    """4 channels: ch0 unused (zero weight), ch2 large + heavily used.

    Initial FP={0,1}, INT={2,3}: quantizing ch2 hurts a lot, ch0 is wasted FP.
    Swapping (0 -> INT, 2 -> FP) should reduce the MLP-output proxy error.
    """
    torch.manual_seed(seed)
    C, I, H, N = 4, 6, 3, 64
    y = torch.randn(N, C)
    y[:, 2] *= 3.0
    Wg = torch.zeros(I, C); Wu = torch.zeros(I, C)
    Wg[:, 2] = torch.randn(I); Wu[:, 2] = 1.0          # ch2 heavily used
    Wg[:, 1] = 0.1 * torch.randn(I); Wu[:, 1] = 0.1
    Wg[:, 3] = 0.1 * torch.randn(I); Wu[:, 3] = 0.1    # ch0 stays zero (unused)
    Wd = torch.randn(H, I)
    scales = y.abs().max(0).values / qmax
    ctx = build_refine_context_from_weights(y, scales, Wg, Wu, Wd, qmax=qmax)
    score = torch.tensor([0.10, 0.20, 0.15, 0.05])     # FP=top-2 -> {1,0}; weak FP=0, strong INT=2
    fp = torch.tensor([0, 1]); intc = torch.tensor([2, 3])
    return ctx, score, fp, intc


# --- A. boundary candidates ---
def test_boundary_candidates_return_fp_out_and_int_in():
    _, score, fp, intc = _toy_ctx()
    cands = build_boundary_swap_candidates(score, fp, intc, candidate_pool=4)
    assert len(cands) == 2
    for fp_out, int_in in cands:
        assert fp_out in fp.tolist()        # leaving FP
        assert int_in in intc.tolist()      # entering FP
    assert cands[0] == (0, 2)               # weakest FP paired with strongest INT


# --- C. accepted swap improves proxy & preserves count ---
def test_accepted_swap_improves_and_preserves_count():
    ctx, score, fp, intc = _toy_ctx()
    r = greedy_refine_fp_mask(ctx, score, fp, intc, margin=1e-6,
                              max_swaps=8, candidate_pool=4)
    assert r["num_swaps"] >= 1
    assert [0, 2] in r["accepted"]                       # ch0<->ch2 swap taken
    assert r["error_after"] < r["error_before"]          # proxy improved
    assert int(r["fp_indices"].numel()) == int(fp.numel())   # FP count preserved
    assert set(r["fp_indices"].tolist()) | set(r["int_indices"].tolist()) == {0, 1, 2, 3}


# --- rejected swap (huge margin) leaves mask unchanged ---
def test_rejected_swap_does_not_change_mask():
    ctx, score, fp, intc = _toy_ctx()
    r = greedy_refine_fp_mask(ctx, score, fp, intc, margin=10.0,    # impossible to clear
                              max_swaps=8, candidate_pool=4)
    assert r["num_swaps"] == 0
    assert r["accepted"] == []
    assert r["fp_indices"].tolist() == fp.tolist()        # unchanged
    assert r["int_indices"].tolist() == intc.tolist()


# --- proxy is lower for the better mask ---
def test_proxy_metric_prefers_protecting_important_channel():
    ctx, _, _, _ = _toy_ctx()
    C = ctx.y.shape[1]
    m_keep0 = torch.zeros(C, dtype=torch.bool); m_keep0[[0, 1]] = True   # protect ch0 (useless)
    m_keep2 = torch.zeros(C, dtype=torch.bool); m_keep2[[1, 2]] = True   # protect ch2 (important)
    assert evaluate_mask_proxy(ctx, m_keep2) < evaluate_mask_proxy(ctx, m_keep0)


# --- E. prioritize high-cascade layers under max_layers ---
def test_refine_prioritizes_high_priority_layer():
    ctx, score, fp, intc = _toy_ctx()
    inputs = {0: {"ctx": ctx, "score": score, "fp_indices": fp, "int_indices": intc},
              1: {"ctx": ctx, "score": score, "fp_indices": fp, "int_indices": intc}}
    cfg = QLotRmsConfig(use_fp_mask_refinement=True, fp_refine_method="greedy_swap",
                        fp_refine_margin=1e-6, fp_refine_max_layers=1)
    out = refine_policy_masks(inputs, cfg, priority={0: 0.1, 1: 10.0})
    assert out[1]["refined"] is True        # high priority refined
    assert out[0]["refined"] is False       # low priority passed through (max_layers=1)
    assert out[0]["fp_indices"].tolist() == fp.tolist()


# --- D. beam search is a clean NotImplementedError ---
def test_beam_not_implemented():
    try:
        beam_refine_fp_mask()
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass


# --------------------------------------------------------------------------- #
# calibration-level (tiny model)
# --------------------------------------------------------------------------- #
def _tiny(n=4):
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
                num_calib_subsets=3, subset_size=4, act_scale_max_tokens=256,
                fp_refine_max_tokens=64)
    base.update(kw)
    return QLotRmsConfig(**base)


def _plan(m, cfg):
    return calibrate(m, _Tok(), cfg, device="cpu", allow_synthetic=True, batch_size=2)


# --- INT permutation rebuilt from the refined mask ---
def test_int_permutation_rebuilt_after_refinement():
    m = _tiny()
    plan = _plan(m, _cfg(use_fp_mask_refinement=True, fp_refine_method="greedy_swap"))
    for lr in plan.layers.values():
        perm, k = lr.perm, lr.k_fp
        # FP block (perm[:k]) == the FP channels of the (refined) mask
        assert set(perm[:k].tolist()) == set(torch.nonzero(lr.mask).squeeze(-1).tolist())
        assert set(perm[k:].tolist()) == set(lr.int_indices.tolist())
        assert sorted(perm.tolist()) == list(range(lr.num_channels))     # valid permutation


# --- metadata stores accepted swaps + proxy errors, budget preserved ---
def test_metadata_stores_swaps_and_preserves_budget():
    m = _tiny()
    cfg_off = _cfg()                                          # unrefined CAP+
    cfg_on = _cfg(use_fp_mask_refinement=True, fp_refine_method="greedy_swap")
    p_off, p_on = _plan(m, cfg_off), _plan(m, cfg_on)
    for i in p_on.layers:
        lr = p_on.layers[i]
        assert lr.refined is True
        assert len(lr.refine_accepted_swaps) == lr.num_refine_swaps
        assert lr.refine_proxy_error_before is not None and lr.refine_proxy_error_after is not None
        # equal-budget: refinement preserves k_fp exactly
        assert lr.k_fp == p_off.layers[i].k_fp


# --- disabled path == unrefined SADND-CAP+ ---
def test_disabled_path_equals_unrefined():
    m = _tiny()
    p_off = _plan(m, _cfg(use_fp_mask_refinement=False))
    p_none = _plan(m, _cfg(use_fp_mask_refinement=True, fp_refine_method="none"))
    for i in p_off.layers:
        assert p_off.layers[i].refined is False
        assert p_none.layers[i].refined is False
        assert p_off.layers[i].fp_indices.tolist() == p_none.layers[i].fp_indices.tolist()
        assert p_off.layers[i].perm.tolist() == p_none.layers[i].perm.tolist()


# --- no runtime top-k/sort introduced by a refined plan ---
def test_refined_plan_no_runtime_topk():
    m = _tiny()
    cfg = _cfg(use_fp_mask_refinement=True, fp_refine_method="greedy_swap")
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
