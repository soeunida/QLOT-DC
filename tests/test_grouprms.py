"""GroupRMS tests: group boundaries (incl. final smaller group) and values."""

import torch

from qlot_rms.grouprms import group_sizes_for, group_rms, estimate_mu_g


def test_group_sizes_exact_multiple():
    assert group_sizes_for(256, 128) == [128, 128]


def test_group_sizes_final_smaller():
    assert group_sizes_for(300, 128) == [128, 128, 44]
    assert group_sizes_for(94, 128) == [94]
    assert group_sizes_for(0, 128) == []


def test_group_rms_per_group_normalization():
    # build INT block with 3 groups: sizes [16, 16, 5]
    torch.manual_seed(0)
    u = torch.randn(8, 37) * torch.tensor([2.0] * 16 + [5.0] * 16 + [0.5] * 5)
    out = group_rms(u, group_size=16, eps=0.0)
    # each group's RMS over the group dim should be ~1 after normalization
    for sl in (slice(0, 16), slice(16, 32), slice(32, 37)):
        rms = out[:, sl].pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_group_rms_returns_r_broadcast():
    u = torch.randn(4, 20)
    out, r = group_rms(u, group_size=8, eps=1e-6, return_r=True)
    assert out.shape == u.shape and r.shape == u.shape
    # within a group r is constant across channels
    assert torch.allclose(r[:, 0:8], r[:, 0:1].expand(-1, 8))


def test_estimate_mu_g_positive():
    u = torch.randn(100, 50) * 3.0
    mu, n = estimate_mu_g(u, group_size=16, eps=1e-6)
    assert mu > 0 and n > 0
    assert n == 100 * len(group_sizes_for(50, 16))
