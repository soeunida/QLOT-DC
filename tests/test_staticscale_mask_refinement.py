"""StaticScale equal-budget FP mask refinement (public API)."""

import torch

from staticscale.mask_refinement import (
    build_boundary_swap_candidates, build_refine_context_from_weights, greedy_refine_fp_mask,
)


def _toy(qmax=7, seed=0):
    torch.manual_seed(seed)
    C, I, H, N = 4, 6, 3, 64
    y = torch.randn(N, C); y[:, 2] *= 3.0
    Wg = torch.zeros(I, C); Wu = torch.zeros(I, C)
    Wg[:, 2] = torch.randn(I); Wu[:, 2] = 1.0
    Wg[:, 1] = 0.1 * torch.randn(I); Wu[:, 1] = 0.1
    Wg[:, 3] = 0.1 * torch.randn(I); Wu[:, 3] = 0.1
    Wd = torch.randn(H, I)
    scales = y.abs().max(0).values / qmax
    ctx = build_refine_context_from_weights(y, scales, Wg, Wu, Wd, qmax=qmax)
    score = torch.tensor([0.10, 0.20, 0.15, 0.05])
    return ctx, score, torch.tensor([0, 1]), torch.tensor([2, 3])


def test_boundary_candidates():
    _, score, fp, intc = _toy()
    cands = build_boundary_swap_candidates(score, fp, intc, 4)
    assert cands[0] == (0, 2)               # weakest FP <-> strongest INT
    for a, b in cands:
        assert a in fp.tolist() and b in intc.tolist()


def test_refinement_preserves_count_and_improves():
    ctx, score, fp, intc = _toy()
    r = greedy_refine_fp_mask(ctx, score, fp, intc, margin=1e-6, max_swaps=8, candidate_pool=4)
    assert int(r["fp_indices"].numel()) == int(fp.numel())     # k_fp preserved
    assert r["error_after"] <= r["error_before"]               # measured improvement


def test_refinement_rejects_under_huge_margin():
    ctx, score, fp, intc = _toy()
    r = greedy_refine_fp_mask(ctx, score, fp, intc, margin=10.0, max_swaps=8, candidate_pool=4)
    assert r["num_swaps"] == 0
    assert r["fp_indices"].tolist() == fp.tolist()             # unchanged on reject
