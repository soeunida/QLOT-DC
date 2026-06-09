"""StaticScale cascade / marginal FP budget (public API)."""

import torch

from staticscale.budget import (
    compute_cascade_error, build_cascade_budget_scores, allocate_fp_budget_from_scores,
    allocate_by_marginal_gain,
)


def test_cascade_recurrence():
    c = compute_cascade_error([0.1, 0.2, 0.05], beta=0.9)
    assert abs(c[0] - 0.1) < 1e-9
    assert abs(c[1] - (0.9 * 0.1 + 0.2)) < 1e-9


def test_cascade_scores_normalized():
    s = build_cascade_budget_scores([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], gamma=1.0)
    assert abs(float(s.sum()) - 1.0) < 1e-9 and bool((s >= 0).all())


def test_global_budget_preserved_and_capped():
    k = allocate_fp_budget_from_scores(30, torch.tensor([1.0, 5.0, 2.0]), [50, 50, 50])
    assert sum(k) == 30 and k[1] > k[0]


def test_marginal_gain_preserves_total():
    scores = {0: torch.rand(40), 1: torch.rand(40)}
    k = allocate_by_marginal_gain(scores, 20)
    assert sum(k.values()) == 20 and all(0 <= k[i] <= 40 for i in (0, 1))
