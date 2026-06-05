"""SADND-CAP+ cascade/marginal budget tests."""

import torch

from qlot_rms.cascade_budget import (
    compute_layer_quant_error, compute_cascade_error, compute_error_amplification,
    build_cascade_budget_scores, allocate_fp_budget_from_scores,
    compute_marginal_gain_table, allocate_by_marginal_gain,
)
from qlot_rms.config import QLotRmsConfig
from qlot_rms.calibration import calibrate
from qlot_rms.model_integration import patch_model, unpatch_model


class _Tok:
    vocab_size = 256


# --- math ---
def test_layer_quant_error_finite():
    hf = torch.randn(8, 16); hq = hf + 0.1 * torch.randn(8, 16)
    e = compute_layer_quant_error(hf, hq)
    assert isinstance(e, float) and e >= 0 and e == e  # finite, non-negative


def test_cascade_recurrence():
    e = [0.1, 0.2, 0.05]
    beta = 0.9
    c = compute_cascade_error(e, beta)
    assert abs(c[0] - 0.1) < 1e-9
    assert abs(c[1] - (0.9 * 0.1 + 0.2)) < 1e-9
    assert abs(c[2] - (0.9 * c[1] + 0.05)) < 1e-9


def test_amplification_shape():
    e = [0.1, 0.2, 0.05]
    amp = compute_error_amplification(e)
    assert len(amp) == 3 and amp[-1] == 1.0


def test_cascade_scores_normalized():
    s = build_cascade_budget_scores([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], gamma=1.0)
    assert abs(float(s.sum()) - 1.0) < 1e-9
    assert bool((s >= 0).all())


def test_global_budget_preserved_and_capped():
    scores = torch.tensor([1.0, 5.0, 2.0])
    C = [50, 50, 50]
    k = allocate_fp_budget_from_scores(30, scores, C)
    assert sum(k) == 30                       # total preserved
    assert all(0 <= k[i] <= C[i] for i in range(3))
    assert k[1] > k[0]                         # higher score -> more budget


def test_global_budget_respects_caps():
    scores = torch.tensor([10.0, 1.0])
    C = [5, 50]                                # layer 0 capped at 5
    k = allocate_fp_budget_from_scores(20, scores, C)
    assert sum(k) == 20 and k[0] <= 5


def test_marginal_gain_table_and_allocation():
    torch.manual_seed(0)
    scores = {0: torch.rand(40), 1: torch.rand(40)}
    table = compute_marginal_gain_table(scores, [0.0, 0.1, 0.2])
    assert set(table) == {0, 1}
    # greedy allocation preserves total and never exceeds caps
    k = allocate_by_marginal_gain(scores, 20)
    assert sum(k.values()) == 20
    assert all(0 <= k[i] <= 40 for i in (0, 1))


def test_marginal_layer_weight_shifts_allocation():
    scores = {0: torch.full((40,), 1.0), 1: torch.full((40,), 1.0)}
    k_plain = allocate_by_marginal_gain(scores, 20)
    k_weighted = allocate_by_marginal_gain(scores, 20, layer_weight={0: 10.0, 1: 1.0})
    assert k_weighted[0] > k_plain[0]          # weighting layer 0 gives it more FP


# --- calibration-level (tiny model) ---
def _tiny(n=4):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=n,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
        tie_word_embeddings=True)).eval()


def _cfg(**kw):
    base = dict(enable_qlot_rms=True, method="sadnd_cap", routing_score="output_aware_sadnd",
                int_permutation_mode="packing_aware", fp_ratio=0.1, global_fp_budget_ratio=0.1,
                calibration_samples=8, calibration_seq_len=16, num_calib_subsets=3,
                subset_size=4, act_scale_max_tokens=256)
    base.update(kw)
    return QLotRmsConfig(**base)


def _plan(m, cfg):
    return calibrate(m, _Tok(), cfg, device="cpu", allow_synthetic=True, batch_size=2)


def test_cascade_allocation_total_budget_and_diag():
    m = _tiny()
    cfg = _cfg(use_cascade_aware_budget=True)
    plan = _plan(m, cfg)
    total_C = sum(lr.num_channels for lr in plan.layers.values())
    total_k = sum(lr.k_fp for lr in plan.layers.values())
    assert total_k == int(0.1 * total_C)               # global budget preserved
    for lr in plan.layers.values():
        assert lr.budget_policy == "cascade"
        assert lr.cascade_local_error is not None and lr.cascade_error is not None
        assert lr.budget_score is not None


def test_layerwise_fp_can_differ():
    m = _tiny()
    plan = _plan(m, _cfg(use_cascade_aware_budget=True))
    ks = [lr.k_fp for lr in plan.layers.values()]
    assert len(set(ks)) > 1                              # not all-equal


def test_cascade_plan_no_runtime_topk():
    m = _tiny()
    cfg = _cfg(use_cascade_aware_budget=True, use_marginal_gain_allocation=True)
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
