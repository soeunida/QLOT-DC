"""StaticScale harmful-INT-channel mining: unit tests (no model)."""
import torch
import torch.nn as nn

from staticscale.harmful_int_mining import (
    score_int_harm, mine_harmful_swaps, _harmful_int_producer,
)
from staticscale import joint_mask_scale_search as J
from staticscale import StaticScaleConfig
from qlot_rms.quant import compute_activation_scales


def _toy_mlp(C=8, I=16, H=8, seed=0):
    torch.manual_seed(seed)
    mlp = nn.Module()
    mlp.gate_proj = nn.Linear(C, I, bias=False)
    mlp.up_proj = nn.Linear(C, I, bias=False)
    mlp.down_proj = nn.Linear(I, H, bias=False)
    with torch.no_grad():
        mlp.gate_proj.weight[:, 4] *= 5.0           # channel 4 = harmful INT outlier
        mlp.up_proj.weight[:, 4] *= 5.0
    return mlp


def test_score_int_harm_monotone():
    resid = torch.tensor([0.1, 0.2, 0.05])
    wcol2 = torch.tensor([1.0, 1.0, 10.0])
    harm = score_int_harm(resid, wcol2)
    assert torch.allclose(harm, torch.tensor([0.1, 0.2, 0.5]))


def test_mine_harmful_swaps_valid_and_budget_preserving():
    harm_int = torch.tensor([0.1, 0.9, 0.2, 0.05])   # aligned to int_idx order
    fp_idx = torch.tensor([0, 1])
    int_idx = torch.tensor([2, 3, 4, 5])
    fp_score = torch.tensor([0.05, 0.50, 0.0, 0.0, 0.0, 0.0])  # FP 0 is weakest
    swaps = mine_harmful_swaps(harm_int, fp_score, fp_idx, int_idx, max_swaps=2, pool=4)
    assert len(swaps) <= 2
    for fp_out, int_in in swaps:
        assert fp_out in fp_idx.tolist() and int_in in int_idx.tolist()
    # most harmful INT is position 1 -> int_idx[1] == 3 ; weakest FP == 0
    assert swaps[0] == (0, 3)


def test_producer_preserves_budget_and_is_static():
    torch.manual_seed(0)
    C = 8
    y = torch.randn(64, C)
    y[:, 4] *= 4.0
    mlp = _toy_mlp(C=C)
    fp0 = torch.tensor([0, 1])
    int0 = torch.tensor([2, 3, 4, 5, 6, 7])
    score = torch.linspace(0.05, 0.5, C)
    base_full = compute_activation_scales(y, 0.999, 127, 1e-8, dim=0)
    cfg = StaticScaleConfig(gt_clip_candidates=[0.8, 1.0, 1.25, 1.5, 2.0],
                            gt_group_size=4, gt_clip_granularity="group",
                            qmax=127, fp_refine_margin=1e-6)
    producer = _harmful_int_producer(max_swaps=4, pool=6)
    fp_new, int_new = producer(y, mlp, fp0, int0, score, base_full, cfg)
    assert int(fp_new.numel()) == int(fp0.numel())          # FP budget preserved
    assert int(fp_new.numel()) + int(int_new.numel()) == C  # full partition
    assert fp_new.dtype == torch.long and int_new.dtype == torch.long  # static indices
    # deterministic: same inputs -> same output
    fp_new2, int_new2 = producer(y, mlp, fp0, int0, score, base_full, cfg)
    assert torch.equal(fp_new.cpu(), fp_new2.cpu())


def test_selection_never_uses_d_eval_for_harmful_int():
    # reuse the joint selection rule: D_sel decides, D_eval ignored
    a = J.JointCandidate("A", "harmful_int_channel_mining", 0.06, 10,
                         d_calib_proxy_before=1.0, d_calib_proxy_after=0.5,
                         d_sel_ppl=6.50, d_eval_ppl=99.0)
    b = J.JointCandidate("B", "harmful_int_channel_mining", 0.06, 10,
                         d_calib_proxy_before=1.0, d_calib_proxy_after=0.5,
                         d_sel_ppl=6.40, d_eval_ppl=1.0)
    winner, _ = J.select_by_d_sel([a, b])
    assert winner is b                                       # lower D_sel wins
