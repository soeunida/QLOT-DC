"""StaticScale diagnostics: pure-helper unit tests (no model)."""
import torch

from staticscale.diagnostics import (
    jaccard, budget_entropy, clip_explained_fraction,
    top_sensitive_protected_fraction, boundary_score_gap,
    classify_int_groups_by_scale, group_type_tau_grids, union_tau_grid,
)


def test_jaccard():
    a = torch.tensor([True, True, False, False])
    b = torch.tensor([True, False, False, False])
    assert abs(jaccard(a, a) - 1.0) < 1e-9
    assert abs(jaccard(a, b) - 0.5) < 1e-9          # |{0}| / |{0,1}|
    assert jaccard(torch.zeros(4, dtype=torch.bool),
                   torch.zeros(4, dtype=torch.bool)) == 1.0


def test_budget_entropy_uniform_vs_concentrated():
    uni = budget_entropy({0: 4, 1: 4, 2: 4, 3: 4})
    conc = budget_entropy({0: 12, 1: 0, 2: 0, 3: 0})
    assert abs(uni - 1.0) < 1e-6                      # uniform -> 1.0
    assert conc < 0.05                                # concentrated -> ~0


def test_clip_explained_fraction():
    # clip moves 0.7 of the 1.0 total improvement -> 0.7
    assert abs(clip_explained_fraction(1.0, 0.3, 0.0) - 0.7) < 1e-9
    # no total improvement -> defined as 1.0 (no-op safe)
    assert clip_explained_fraction(0.5, 0.5, 0.5) == 1.0


def test_top_sensitive_protected_fraction():
    score = torch.tensor([0.9, 0.8, 0.1, 0.05, 0.02, 0.01, 0.0, 0.0])
    fp_mask = torch.tensor([True, True, False, False, False, False, False, False])
    # top 25% (2 channels) are both FP -> 1.0
    assert abs(top_sensitive_protected_fraction(score, fp_mask, 0.25) - 1.0) < 1e-9


def test_boundary_score_gap_sign():
    score = torch.tensor([1.0, 0.9, 0.2, 0.1])
    gap = boundary_score_gap(score, torch.tensor([0, 1]), torch.tensor([2, 3]))
    assert gap > 0                                    # FP scores above INT scores


def test_group_type_tau_static_and_union():
    grids = group_type_tau_grids()
    assert set(grids) == {"low", "mid", "high", "outlier"}
    for g in grids.values():
        assert all(isinstance(x, float) for x in g)
    u = union_tau_grid()
    assert u == sorted(set(u)) and 1.0 in u and 2.5 in u   # sorted, deduped


def test_classify_int_groups_by_scale():
    scales = torch.tensor([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0, 0.03, 0.04])
    cls = classify_int_groups_by_scale(scales)
    assert len(cls) == scales.numel()
    assert set(cls).issubset({"low", "mid", "high", "outlier"})
    assert cls[-3] == "outlier"                       # the 5.0 is the outlier
