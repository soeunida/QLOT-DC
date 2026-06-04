"""Unit tests for block-output correction fitting (Q-LOT-OBC)."""

import torch

from qlot_rms.block_correction import fit_bias, fit_affine, _apply
from qlot_rms.lowrank import fit_lowrank


def test_bias_shape_and_reduces_mean_error():
    torch.manual_seed(0)
    y_q = torch.randn(200, 16)
    y_fp = y_q + 0.4 + 0.02 * torch.randn(200, 16)   # constant offset
    b = fit_bias(y_fp - y_q)
    assert b.shape == (16,)
    before = (y_fp - y_q).mean(0).abs().mean()
    after = (y_fp - (y_q + b)).mean(0).abs().mean()
    assert after < before and after < 1e-5


def test_affine_shape_clamp_and_reduces_mse():
    torch.manual_seed(0)
    y_q = torch.randn(500, 8)
    a_true = torch.tensor([1.3, 0.7, 1.0, 1.5, 0.6, 1.1, 0.9, 1.2])
    b_true = torch.linspace(-0.5, 0.5, 8)
    y_fp = a_true * y_q + b_true + 0.01 * torch.randn(500, 8)
    a, b = fit_affine(y_q, y_fp, a_min=0.5, a_max=2.0)
    assert a.shape == (8,) and b.shape == (8,)
    assert bool((a >= 0.5).all()) and bool((a <= 2.0).all())
    before = ((y_fp - y_q) ** 2).mean()
    after = ((y_fp - (a * y_q + b)) ** 2).mean()
    assert after < before * 0.2


def test_affine_clamp_enforced():
    torch.manual_seed(1)
    y_q = torch.randn(300, 4)
    y_fp = 10.0 * y_q   # huge slope -> must clamp to a_max
    a, b = fit_affine(y_q, y_fp, a_min=0.5, a_max=2.0)
    assert bool((a <= 2.0 + 1e-6).all())


def test_block_lowrank_shapes_and_reduces_residual():
    torch.manual_seed(0)
    T, H = 400, 24
    X = torch.randn(T, H)
    AB = torch.randn(H, 4) @ torch.randn(4, H)
    E = X @ AB + 0.01 * torch.randn(T, H)
    A, B = fit_lowrank(X, E, rank=4)
    assert A.shape[0] == H and A.shape[1] <= 4 and B.shape[1] == H
    before = E.pow(2).mean()
    after = (E - X @ A @ B).pow(2).mean()
    assert after < before * 0.5


def test_apply_modes_shapes():
    T, H = 5, 10
    y_q = torch.randn(T, H); h = torch.randn(T, H)
    assert _apply("bias", y_q, h, {"b": torch.randn(H)}).shape == (T, H)
    assert _apply("affine", y_q, h, {"a": torch.randn(H), "b": torch.randn(H)}).shape == (T, H)
    assert _apply("lowrank", y_q, h, {"A": torch.randn(H, 3), "B": torch.randn(3, H)}).shape == (T, H)
    assert torch.equal(_apply("none", y_q, h, {}), y_q)
