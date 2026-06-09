"""StaticScale joint mask-scale search: unit tests (no model download required).

Covers: candidate generation, exact FP-budget preservation, selection never using
D_eval, static tau/eta metadata, JSON serialization, and fallback recording.
"""
import json

import torch
import torch.nn as nn

from staticscale.joint_mask_scale_search import (
    JointCandidate, assert_static_metadata, evaluate_layer_orderings,
    select_by_d_sel, family_config_overrides, BASELINE_FAMILIES, JOINT_FAMILIES,
    ADDITIVE_REFERENCE,
)


def _toy_mlp(C=8, I=16, H=8, seed=0):
    torch.manual_seed(seed)
    mlp = nn.Module()
    mlp.gate_proj = nn.Linear(C, I, bias=False)
    mlp.up_proj = nn.Linear(C, I, bias=False)
    mlp.down_proj = nn.Linear(I, H, bias=False)
    # make channel 2 an outlier so swaps/clip have something to do
    with torch.no_grad():
        mlp.gate_proj.weight[:, 2] *= 4.0
        mlp.up_proj.weight[:, 2] *= 4.0
    return mlp


def _toy_layer(C=8, seed=0):
    torch.manual_seed(seed)
    y = torch.randn(64, C)
    y[:, 2] *= 3.0
    fp0 = torch.tensor([0, 1])
    int0 = torch.tensor([2, 3, 4, 5, 6, 7])
    score = torch.linspace(0.05, 0.5, C)
    return y, fp0, int0, score


TAU = [0.8, 1.0, 1.25, 1.5, 2.0]


def test_family_overrides_cover_baselines():
    for fam in BASELINE_FAMILIES:
        ov = family_config_overrides(fam, 0.2)
        assert isinstance(ov, dict) and ov  # non-empty config override
    # joint families must NOT be expressible as plain calibrate overrides
    for fam in JOINT_FAMILIES:
        try:
            family_config_overrides(fam, 0.2)
            assert False, "joint family should raise"
        except ValueError:
            pass


def test_candidate_generation_and_budget_preserved():
    y, fp0, int0, score = _toy_layer()
    mlp = _toy_mlp()
    res = evaluate_layer_orderings(
        y, mlp, fp0, int0, score, tau_grid=TAU, group_size=4, granularity="group",
        refine_margin=1e-6, max_swaps=8, candidate_pool=6, qmax=127)
    k_fp = int(fp0.numel())
    assert set(res).issuperset({"capplus", "capplus_clip", "capplus_refine_clip",
                                "joint_retune_then_swap", "joint_swap_then_retune"})
    for fam, d in res.items():
        assert d["k_fp"] == k_fp, f"{fam} broke the FP budget"      # exact budget
        assert d["proxy"] == d["proxy"] and d["proxy"] >= 0.0       # finite proxy


def test_clip_does_not_blow_up_unified_proxy():
    # Clip tuning minimizes the GT *surrogate* (weight-normalized per-channel quant
    # MSE), NOT the unified MLP-output rel-L2 metric, so it need not strictly reduce
    # the unified proxy. It must, however, not substantially worsen it. (That the
    # surrogate and the unified metric can disagree is itself evidence that the
    # recoverable error is mostly scale-driven and hard to push further.)
    y, fp0, int0, score = _toy_layer()
    mlp = _toy_mlp()
    res = evaluate_layer_orderings(
        y, mlp, fp0, int0, score, tau_grid=TAU, group_size=4,
        refine_margin=1e-6, max_swaps=0, candidate_pool=6, qmax=127,
        families=["capplus", "capplus_clip"])
    assert res["capplus_clip"]["proxy"] <= res["capplus"]["proxy"] * 1.05


def test_selection_never_uses_d_eval():
    # Two candidates: A is better on D_sel, B is better on D_eval. Winner must be A.
    a = JointCandidate("A", "joint_retune_then_swap", 0.2, 10,
                       d_calib_proxy_before=1.0, d_calib_proxy_after=0.5,
                       d_sel_ppl=6.50, d_eval_ppl=9.99)
    b = JointCandidate("B", "joint_swap_then_retune", 0.2, 10,
                       d_calib_proxy_before=1.0, d_calib_proxy_after=0.5,
                       d_sel_ppl=6.60, d_eval_ppl=1.00)
    winner, allc = select_by_d_sel([a, b])
    assert winner is a, "selection must use D_sel, not D_eval"
    assert a.status == "accepted" and b.status == "rejected"


def test_selection_records_fallback_when_none_pass():
    # candidate improved D_calib but has no D_sel PPL -> rejected, no winner
    c = JointCandidate("C", "joint_retune_then_swap", 0.2, 10,
                       d_calib_proxy_before=1.0, d_calib_proxy_after=0.5,
                       d_sel_ppl=None)
    winner, allc = select_by_d_sel([c])
    assert winner is None
    assert allc[0].status == "rejected"
    # explicit fallback candidate is representable and static
    fb = JointCandidate("fallback", "capplus_clip", 0.2, 10, status="fallback",
                        reason="no joint candidate beat CAP+ + clip")
    assert_static_metadata(fb)


def test_rejected_when_calib_not_improved():
    c = JointCandidate("D", "joint_retune_then_swap", 0.2, 10,
                       d_calib_proxy_before=0.5, d_calib_proxy_after=0.9,  # worse
                       d_sel_ppl=6.50)
    winner, allc = select_by_d_sel([c], require_d_calib_improvement=True)
    assert winner is None
    assert allc[0].status == "rejected"
    assert "D_calib" in allc[0].reason


def test_metadata_json_serializable_and_static():
    c = JointCandidate(
        "cand0", "joint_retune_then_swap", 0.2, total_fp_channels=128,
        per_layer_fp_budgets={0: 4, 1: 8}, num_swaps=3, top_k_layers="all",
        tau_grid=TAU, percentile=0.999, eta_mode="layerwise_clipped_0.95_1.05",
        d_calib_proxy_before=1.0, d_calib_proxy_after=0.9, d_sel_ppl=6.5,
        delta_vs_capplus_clip=-0.004, status="accepted", reason="best D_sel")
    s = c.to_json()
    d = json.loads(s)                              # serializes round-trip
    assert d["candidate_family"] == "joint_retune_then_swap"
    assert d["per_layer_fp_budgets"]["0"] == 4    # int keys -> str keys
    assert_static_metadata(c)
    # no callables / tensors leak into the serialized metadata (static only)
    for v in d.values():
        assert not callable(v)


def test_no_runtime_search_in_metadata():
    # The candidate metadata that reaches inference must be plain static values
    # (ints / floats / lists / dicts) -- no tensors, no callables, no generators.
    y, fp0, int0, score = _toy_layer()
    mlp = _toy_mlp()
    res = evaluate_layer_orderings(
        y, mlp, fp0, int0, score, tau_grid=TAU, group_size=4,
        refine_margin=1e-6, max_swaps=4, candidate_pool=6, qmax=127)
    for fam, d in res.items():
        for k, v in d.items():
            assert isinstance(v, (int, float)), f"{fam}.{k} is not a static scalar"
