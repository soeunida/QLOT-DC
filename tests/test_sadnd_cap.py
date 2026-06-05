"""SADND-CAP policy tests: output-aware score, FP/INT split, global budget,
packing-aware permutation, equal-budget accept-only selection."""

import torch

from qlot_rms.quant import group_sizes_for
from qlot_rms.sadnd_cap import (
    compute_output_aware_sadnd_score, select_fp_int_channels,
    allocate_global_fp_budget, build_packing_aware_int_permutation,
    equal_budget_accept_only_select, build_layer_fp_int_perm,
)


# --- output-aware score ---
def test_output_aware_changes_ranking_under_weight_impact():
    torch.manual_seed(0)
    C = 32
    delta = torch.linspace(0.1, 0.2, C)            # nearly flat distortion
    wg = torch.ones(C); wu = torch.ones(C)
    base = compute_output_aware_sadnd_score(delta, wg, wu)
    wg2 = wg.clone(); wg2[5] = 100.0               # channel 5 heavily used
    s2 = compute_output_aware_sadnd_score(delta, wg2, wu)
    assert s2.shape == (C,)
    assert int(torch.argmax(base)) != 5            # not top before
    assert int(torch.argmax(s2)) == 5              # top after weight impact


# --- FP/INT split ---
def test_fp_int_split_count_and_partition():
    score = torch.rand(100)
    fp, intc = select_fp_int_channels(score, k_fp=6)
    assert fp.numel() == 6 and intc.numel() == 94
    assert sorted(fp.tolist() + intc.tolist()) == list(range(100))
    # FP = the 6 highest-score channels
    top6 = set(torch.topk(score, 6).indices.tolist())
    assert set(fp.tolist()) == top6


# --- global FP budget ---
def test_global_budget_preserves_total():
    torch.manual_seed(1)
    scores = {0: torch.rand(100), 1: torch.rand(100), 2: torch.rand(100)}
    k = allocate_global_fp_budget(scores, fp_ratio=0.06)   # total 300 -> 18
    assert sum(k.values()) == int(0.06 * 300)


def test_global_budget_layer_allocation_can_differ():
    # layer 0 has all-high scores, layer 1 all-low -> layer 0 gets more FP
    scores = {0: torch.full((50,), 10.0), 1: torch.full((50,), 0.1)}
    k = allocate_global_fp_budget(scores, fp_ratio=0.10)   # total 100 -> 10
    assert sum(k.values()) == 10
    assert k[0] > k[1]


# --- packing-aware permutation ---
def test_packing_aware_only_reorders_int_and_fp_block_first():
    C = 64
    score = torch.rand(C)
    int_scales_full = torch.rand(C) + 0.01
    fp, int_packed, perm, mask = build_layer_fp_int_perm(
        score, k_fp=6, int_scales_full=int_scales_full,
        int_permutation_mode="packing_aware", group_size=16)
    # FP block first, INT after
    assert perm.tolist() == fp.tolist() + int_packed.tolist()
    assert bool(mask[fp].all())
    # INT permutation only reorders the INT set (set preserved)
    int_orig = set(torch.nonzero(~mask, as_tuple=False).squeeze(-1).tolist())
    assert set(int_packed.tolist()) == int_orig


def test_scale_sorted_reduces_within_group_scale_variance():
    torch.manual_seed(0)
    n, gs = 256, 64
    int_idx = torch.arange(n)
    scales = torch.rand(n) * 100.0                 # high-variance scales
    packed = build_packing_aware_int_permutation(int_idx, scales, "scale_sorted", gs)

    def within_group_var(order):
        s = scales[order]
        gsizes = group_sizes_for(n, gs)
        v, start = [], 0
        for g in gsizes:
            v.append(s[start:start + g].var(unbiased=False))
            start += g
        return torch.stack(v).mean()
    before = within_group_var(int_idx)
    after = within_group_var(packed)               # packed is a permutation of int_idx
    assert after < before * 0.5


# --- equal-budget accept-only ---
def test_accept_only_rejects_worse_candidate():
    name, clear = equal_budget_accept_only_select(
        {"oa_packing": 8.05, "packing_aware": 8.04}, sadnd_ppl=8.00, margin=0.001)
    assert name == "sadnd" and clear is False       # none beats sadnd -> fallback


def test_accept_only_accepts_better_candidate():
    name, clear = equal_budget_accept_only_select(
        {"oa_packing": 7.990, "packing_aware": 7.995}, sadnd_ppl=8.00, margin=0.001)
    assert name == "oa_packing" and clear is True


def test_accept_only_within_margin_falls_back():
    name, clear = equal_budget_accept_only_select(
        {"oa_packing": 7.9995}, sadnd_ppl=8.00, margin=0.001)   # only 0.0005 better
    assert name == "sadnd" and clear is False


# --- SADND-CAP+ policy metadata via calibrate (tiny model) ---
def test_cascade_policy_metadata_and_per_layer_fp():
    import torch as _t
    from transformers import LlamaConfig, LlamaForCausalLM
    from qlot_rms.config import QLotRmsConfig
    from qlot_rms.calibration import calibrate

    class _Tok:
        vocab_size = 256
    _t.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
        tie_word_embeddings=True)).eval()
    cfg = QLotRmsConfig(enable_qlot_rms=True, method="sadnd_cap",
                        routing_score="output_aware_sadnd", int_permutation_mode="packing_aware",
                        fp_ratio=0.1, global_fp_budget_ratio=0.1, use_cascade_aware_budget=True,
                        calibration_samples=8, calibration_seq_len=16, num_calib_subsets=3,
                        subset_size=4, act_scale_max_tokens=256)
    plan = calibrate(m, _Tok(), cfg, device="cpu", allow_synthetic=True, batch_size=2)
    for lr in plan.layers.values():
        assert lr.budget_policy == "cascade"
        # cascade_budget_summary fields populated
        assert lr.cascade_local_error is not None
        assert lr.cascade_error is not None
        assert lr.budget_score is not None
        # per-layer fp ratio stored
        assert lr.selected_fp_ratio is not None
        assert abs(lr.selected_fp_ratio - lr.k_fp / lr.num_channels) < 1e-9
    # summary() exposes the diagnostics
    s = next(iter(plan.layers.values())).summary()
    assert "cascade_error" in s and "budget_policy" in s
