"""Quantization primitives for Q-LOT-RMS (reference / simulated).

These are deliberately simple and numerically explicit so the reference
inference path is correct without any custom kernel.  The W8-G128 weight format
mirrors the existing SFPA/PQ8 convention (groups along the *input* dimension,
per-group symmetric scale ``wmax/qmax``) but is re-implemented here so this
package has no dependency on ``PQ8_v29.py``.

Orientation reminder: PyTorch ``nn.Linear`` weight is ``[out_features,
in_features]``.  Weight quantization here groups along the input dimension
(dim=1, the columns).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Quantiles
# --------------------------------------------------------------------------- #
def channel_quantile(x: torch.Tensor, q: float, dim: int = 0) -> torch.Tensor:
    """Per-channel quantile via sort (avoids ``torch.quantile`` size limits).

    Uses the same linear-interpolation convention as ``torch.quantile``.

    Parameters
    ----------
    x   : tensor; reduction is along ``dim``.
    q   : quantile in [0, 1].
    dim : dimension to reduce.

    Returns a tensor with ``dim`` removed.
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1]")
    x = x.float()
    n = x.shape[dim]
    if n == 0:
        raise ValueError("cannot take quantile of an empty dimension")
    if n == 1:
        return x.squeeze(dim)
    xs, _ = torch.sort(x, dim=dim)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    lo_v = xs.index_select(dim, torch.tensor([lo], device=x.device)).squeeze(dim)
    if hi == lo:
        return lo_v
    hi_v = xs.index_select(dim, torch.tensor([hi], device=x.device)).squeeze(dim)
    return lo_v * (1.0 - frac) + hi_v * frac


# --------------------------------------------------------------------------- #
# Symmetric per-channel INT8 (activations)
# --------------------------------------------------------------------------- #
def quantize_activation_int8(
    y: torch.Tensor, scales: torch.Tensor, qmax: int = 127
) -> torch.Tensor:
    """Fake-quantize activations with frozen *per-channel* scales.

    ``y`` is ``[..., C]``; ``scales`` is ``[C]``.  Returns a dequantized
    (fake-quantized) float tensor of the same shape.  Symmetric, no zero point.
    """
    s = scales.to(y.dtype).clamp_min(torch.finfo(torch.float32).tiny)
    q = torch.clamp(torch.round(y / s), -qmax, qmax)
    return q * s


def compute_activation_scales(
    y: torch.Tensor, p_act: float, qmax: int = 127, eps_scale: float = 1e-8, dim: int = 0
) -> torch.Tensor:
    """Frozen per-channel activation scale ``a_c = max(quantile(|y_c|,p)/qmax, eps)``.

    ``y`` flattened to ``[tokens, C]`` with reduction along ``dim`` (default 0).
    """
    absy = y.abs()
    qv = channel_quantile(absy, p_act, dim=dim)
    return torch.clamp(qv / float(qmax), min=eps_scale)


# --------------------------------------------------------------------------- #
# Simulated W8-G128 weight quantization (groups along input dim)
# --------------------------------------------------------------------------- #
def _group_reshape(W: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int, int]:
    """Reshape ``W`` ``[out, in]`` into ``[out, n_groups, group_size]`` with padding."""
    out, inp = W.shape
    n_g = math.ceil(inp / group_size)
    pad = n_g * group_size - inp
    if pad > 0:
        W = F.pad(W, (0, pad))
    return W.reshape(out, n_g, group_size), n_g, pad


def fake_quantize_weight_w8_g128(
    W: torch.Tensor, group_size: int = 128, qmax: int = 127
) -> torch.Tensor:
    """Return a fake-quantized W8-G128 weight, same shape/dtype as ``W``.

    Symmetric per-group scale ``s_g = max(|W_group|)/qmax`` with groups along the
    input dimension (columns).  FP32 internally; cast back to ``W.dtype``.
    """
    orig_dtype = W.dtype
    out, inp = W.shape
    W3, n_g, _ = _group_reshape(W.float(), group_size)
    wmax = W3.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    s_g = wmax / float(qmax)
    q = torch.clamp(torch.round(W3 / s_g), -qmax, qmax)
    W_dq3 = q * s_g
    W_dq = W_dq3.reshape(out, n_g * group_size)[:, :inp]
    return W_dq.to(orig_dtype)


# --------------------------------------------------------------------------- #
# Simulated INT8 matmul with FP32 accumulation
# --------------------------------------------------------------------------- #
def simulated_int8_matmul(
    y: torch.Tensor,
    W: torch.Tensor,
    act_scales: torch.Tensor,
    w_group_size: int = 128,
    qmax: int = 127,
) -> torch.Tensor:
    """Reference INT8 (act) x W8-G128 (weight) matmul, FP32 accumulation.

    This is a *fake-quantized* reference: both operands are quantized then
    dequantized, and the product is accumulated in FP32 via a float matmul.
    The integer-domain accumulation (per-group partial sums) is a kernel detail
    that does not change the numerical result of this fake-quant reference.

    Parameters
    ----------
    y          : ``[..., Cin]`` activation (already GroupRMS'd + affine on INT branch).
    W          : ``[out, Cin]`` weight (PyTorch nn.Linear orientation).
    act_scales : ``[Cin]`` frozen per-input-channel activation scales.
    """
    if W.shape[1] != y.shape[-1]:
        raise ValueError(
            f"weight in_features {W.shape[1]} != activation channels {y.shape[-1]}"
        )
    yq = quantize_activation_int8(y, act_scales, qmax=qmax).float()
    Wq = fake_quantize_weight_w8_g128(W, group_size=w_group_size, qmax=qmax).float()
    out = torch.matmul(yq, Wq.t())
    return out.to(y.dtype)
