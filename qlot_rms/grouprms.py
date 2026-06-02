"""GroupRMS for the INT-routed FFN input branch.

GroupRMS is applied ONLY to the INT-routed branch of ``gate_proj`` and
``up_proj``.  It is NOT applied to the FP branch, ``down_proj``, or the
attention output branch.

After the static permutation P = [FP, INT], INT channels are contiguous, so
GroupRMS partitions the INT block into contiguous groups of ``group_size``
(default 128).  The final group may be smaller when ``(C - K_F)`` is not a
multiple of ``group_size``.

For each token ``t`` and group ``g``::

    r_tg = sqrt(mean(u_I[t, group]^2) + eps)
    GroupRMS(u_I)[t, group] = u_I[t, group] / r_tg

GroupRMS is *not* function-preserving: it removes per-group scale information.
Mean-scale compensation (see :mod:`calibration`) restores the average scale by
folding ``mu_g`` into the INT weight columns at packing time.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F


def group_sizes_for(num_int_channels: int, group_size: int) -> List[int]:
    """Contiguous group sizes covering ``num_int_channels``; last may be smaller."""
    if num_int_channels < 0:
        raise ValueError("num_int_channels must be >= 0")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    sizes: List[int] = []
    remaining = num_int_channels
    while remaining > 0:
        s = min(group_size, remaining)
        sizes.append(s)
        remaining -= s
    return sizes


def group_rms_loop(
    u_int: torch.Tensor, group_size: int, eps: float = 1e-6, return_r: bool = False
):
    """Reference (loop) GroupRMS -- kept for the vectorized-equivalence test.

    Groups are contiguous blocks of ``group_size`` along the last dim; the final
    group may be smaller.  ``r_tg = sqrt(mean(block^2) + eps)``.
    """
    *_, c_int = u_int.shape
    sizes = group_sizes_for(c_int, group_size)
    out = torch.empty_like(u_int)
    r_full = torch.empty_like(u_int) if return_r else None

    start = 0
    for s in sizes:
        sl = slice(start, start + s)
        block = u_int[..., sl]
        r = block.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
        out[..., sl] = block / r
        if return_r:
            r_full[..., sl] = r.expand_as(block)
        start += s

    if return_r:
        return out, r_full
    return out


def group_rms(
    u_int: torch.Tensor, group_size: int, eps: float = 1e-6, return_r: bool = False
):
    """Vectorized GroupRMS over the INT block ``u_int`` of shape ``[..., C_int]``.

    Numerically equivalent to :func:`group_rms_loop` (verified by a regression
    test) but computed in one pass: pad to a whole number of groups, reduce
    sum-of-squares per group, divide by the *real* per-group element count
    (so the padded final group is unaffected), then broadcast-divide.
    """
    *lead, c_int = u_int.shape
    if c_int == 0:
        return (u_int, u_int) if return_r else u_int

    n_g = math.ceil(c_int / group_size)
    pad = n_g * group_size - c_int
    x = F.pad(u_int, (0, pad)) if pad > 0 else u_int
    x = x.reshape(*lead, n_g, group_size)

    sumsq = x.pow(2).sum(dim=-1)                       # [*lead, n_g]
    counts = torch.full((n_g,), float(group_size),
                        device=u_int.device, dtype=sumsq.dtype)
    if pad > 0:
        counts[-1] = float(group_size - pad)           # real size of last group
    r = (sumsq / counts).add(eps).sqrt().unsqueeze(-1)  # [*lead, n_g, 1]

    out = (x / r).reshape(*lead, n_g * group_size)[..., :c_int]
    if return_r:
        r_full = r.expand(*lead, n_g, group_size).reshape(*lead, n_g * group_size)[..., :c_int]
        return out, r_full
    return out


def estimate_mu_g(
    u_int: torch.Tensor, group_size: int, eps: float = 1e-6
) -> Tuple[float, int]:
    """Estimate the mean RMS scale ``mu_g`` over tokens & groups.

    ``u_int`` is ``[tokens, C_int]`` (pre-affine INT-channel activations in
    permuted order).  Returns ``(mu_g, n_samples)`` where ``n_samples`` is the
    number of (token, group) RMS values averaged (for streaming aggregation).
    """
    *_, c_int = u_int.shape
    sizes = group_sizes_for(c_int, group_size)
    total = 0.0
    count = 0
    start = 0
    for s in sizes:
        block = u_int[..., start:start + s]
        r = block.pow(2).mean(dim=-1).add(eps).sqrt()  # [tokens]
        total += float(r.sum().item())
        count += r.numel()
        start += s
    if count == 0:
        return 0.0, 0
    return total / count, count


def estimate_mu_g_per_group(
    u_int: torch.Tensor, group_size: int, eps: float = 1e-6
):
    """Per-group mean RMS scale (streaming-friendly).

    ``u_int`` is ``[tokens, C_int]``.  Returns ``(sums, counts)`` where ``sums[g]``
    is the sum over tokens of ``r_tg`` for group ``g`` and ``counts[g]`` is the
    number of tokens, so a caller can accumulate across batches and divide at the
    end.  Per-group compensation is required: once high-energy channels are
    routed to FP, the per-group RMS of the remaining INT channels differs sharply
    across groups, and a single scalar mu_g mis-compensates each group.
    """
    *_, c_int = u_int.shape
    sizes = group_sizes_for(c_int, group_size)
    sums: List[float] = []
    counts: List[int] = []
    start = 0
    for s in sizes:
        block = u_int[..., start:start + s]
        r = block.pow(2).mean(dim=-1).add(eps).sqrt()  # [tokens]
        sums.append(float(r.sum().item()))
        counts.append(int(r.numel()))
        start += s
    return sums, counts


def broadcast_per_group_to_channels(
    per_group, num_int_channels: int, group_size: int
) -> torch.Tensor:
    """Expand a per-group vector to a per-INT-channel vector (length C_int)."""
    sizes = group_sizes_for(num_int_channels, group_size)
    parts = [torch.full((s,), float(per_group[g])) for g, s in enumerate(sizes)]
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts, dim=0)
