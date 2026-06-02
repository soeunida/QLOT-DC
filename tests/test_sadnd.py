"""SADND tests: score shape/finiteness, aggregation, scale-invariance."""

import torch

from qlot_rms.sadnd import (
    proxy_distortion_subset,
    aggregate_distortion,
    assign_channels,
)


def test_proxy_distortion_shape_and_finite():
    torch.manual_seed(0)
    u = torch.randn(2000, 32)
    d = proxy_distortion_subset(u, p_proxy=0.9995, qmax=127, eps=1e-6)
    assert d.shape == (32,)
    assert torch.isfinite(d).all()
    assert (d >= 0).all()  # relative squared distortion is non-negative


def test_aggregate_mean_plus_lambda_std():
    per_subset = torch.tensor([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]])
    agg = aggregate_distortion(per_subset, lambda_agg=1.0)
    mean = per_subset.mean(0)
    std = per_subset.std(0, unbiased=False)
    assert torch.allclose(agg, mean + std)
    # channel 0 has spread, channel 1 has none
    assert agg[1].item() == 2.0


def test_aggregate_single_subset_zero_std():
    per_subset = torch.tensor([[1.0, 2.0, 3.0]])
    agg = aggregate_distortion(per_subset, lambda_agg=1.0)
    assert torch.allclose(agg, per_subset[0])


def test_distortion_is_scale_invariant():
    # SADND uses RELATIVE distortion -> multiplying a channel by a constant
    # should not change its score (proxy scale tracks magnitude).
    torch.manual_seed(1)
    u = torch.randn(4000, 4)
    d1 = proxy_distortion_subset(u)
    u2 = u.clone()
    u2[:, 0] *= 100.0
    d2 = proxy_distortion_subset(u2)
    assert torch.allclose(d1, d2, atol=1e-3)


def test_assign_finiteness_on_heavy_tail():
    # a heavy-tailed channel should incur higher relative distortion -> FP
    torch.manual_seed(2)
    u = torch.randn(8000, 10)
    u[:, 7] = torch.randn(8000) ** 3 * 5  # heavy tail
    d = aggregate_distortion(torch.stack([proxy_distortion_subset(u) for _ in range(3)]))
    fp, intc, perm, mask = assign_channels(d, fp_ratio=0.2)  # K_F=2
    assert 7 in fp.tolist()
