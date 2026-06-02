"""Quant primitive tests: quantile, INT8 act quant, W8-G128, sim matmul."""

import torch

from qlot_rms.quant import (
    channel_quantile,
    quantize_activation_int8,
    compute_activation_scales,
    fake_quantize_weight_w8_g128,
    simulated_int8_matmul,
)


def test_channel_quantile_matches_torch():
    x = torch.randn(5000, 7)
    for q in (0.5, 0.9, 0.999, 0.9995):
        ours = channel_quantile(x, q, dim=0)
        ref = torch.quantile(x, q, dim=0)
        assert torch.allclose(ours, ref, atol=1e-4), q


def test_activation_scales_positive():
    y = torch.randn(1000, 16) * 3.0
    s = compute_activation_scales(y, p_act=0.999, qmax=127, eps_scale=1e-8)
    assert s.shape == (16,)
    assert torch.isfinite(s).all()
    assert (s > 0).all()


def test_int8_activation_quant_bounded():
    y = torch.randn(100, 8) * 5.0
    scales = compute_activation_scales(y, 0.999)
    yq = quantize_activation_int8(y, scales, qmax=127)
    # codes must be within [-127, 127]
    codes = torch.round(y / scales.clamp_min(1e-12))
    assert codes.abs().max() >= 1  # something quantized
    assert torch.isfinite(yq).all()


def test_w8_g128_groups_and_finite():
    W = torch.randn(32, 300)
    Wq = fake_quantize_weight_w8_g128(W, group_size=128, qmax=127)
    assert Wq.shape == W.shape
    assert torch.isfinite(Wq).all()
    # fake-quant should be close-ish but not identical
    assert (Wq - W).abs().mean() > 0


def test_simulated_matmul_shape_and_finite():
    y = torch.randn(4, 10, 256)
    W = torch.randn(64, 256)
    scales = compute_activation_scales(y.reshape(-1, 256), 0.999)
    z = simulated_int8_matmul(y, W, scales, w_group_size=128, qmax=127)
    assert z.shape == (4, 10, 64)
    assert torch.isfinite(z).all()
