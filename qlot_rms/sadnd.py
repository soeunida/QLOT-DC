"""SADND routing: Sensitivity-Aware Distortion-Normalized Decision.

Given pre-affine LN2/RMSNorm activations ``u^(l)`` collected over calibration
subsets, SADND scores each channel by the *relative proxy distortion* incurred
by an INT8 proxy, aggregates across subsets (mean + lambda * std), and assigns
the highest-distortion channels to the FP branch and the remainder to the INT
branch.

Definitions (per routed layer ``l``, per channel ``c``, per subset ``s``)::

    s_c     = quantile(|u_c|, p_proxy) / qmax          # high-quantile proxy scale
    uhat_c  = clamp(round(u_c / s_c), -qmax, qmax) * s_c
    delta_c = E[(u_c - uhat_c)^2] / (E[u_c^2] + eps)   # relative proxy distortion

Aggregation::

    delta_tilde_c = mean_s(delta_c) + lambda_agg * std_s(delta_c)

Assignment::

    K_F           = floor(fp_ratio * C)
    INT channels  = BottomK(delta_tilde, C - K_F)      # lowest distortion
    FP channels   = remaining K_F (highest distortion)
    permutation P = [FP channels (orig order), INT channels (orig order)]

All of this happens at calibration time only.  The inference path consumes the
frozen permutation/mask and never recomputes a routing score.
"""

from __future__ import annotations

from typing import Tuple

import torch

from .quant import channel_quantile


def proxy_distortion_subset(
    u: torch.Tensor, p_proxy: float = 0.9995, qmax: int = 127, eps: float = 1e-6
) -> torch.Tensor:
    """Relative proxy distortion ``delta_c`` for one subset.

    Parameters
    ----------
    u : ``[tokens, C]`` pre-affine activations for one subset & one layer.

    Returns ``[C]`` float tensor of per-channel relative distortion.
    """
    u = u.float()
    # high-quantile proxy scale per channel
    s_c = (channel_quantile(u.abs(), p_proxy, dim=0) / float(qmax)).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    # reconstruct proxy INT8 activation
    uhat = torch.clamp(torch.round(u / s_c), -qmax, qmax) * s_c
    num = (u - uhat).pow(2).mean(dim=0)      # E[(u - uhat)^2] per channel
    den = u.pow(2).mean(dim=0) + eps         # E[u^2] + eps
    return num / den


def aggregate_distortion(
    per_subset: torch.Tensor, lambda_agg: float = 1.0
) -> torch.Tensor:
    """Aggregate ``[num_subsets, C]`` distortions into ``delta_tilde`` ``[C]``.

    ``delta_tilde_c = mean_s(delta_c) + lambda_agg * std_s(delta_c)``.
    Uses population std (unbiased=False) so a single subset yields std=0.
    """
    if per_subset.dim() != 2:
        raise ValueError("per_subset must be [num_subsets, C]")
    mean = per_subset.mean(dim=0)
    if per_subset.shape[0] > 1:
        std = per_subset.std(dim=0, unbiased=False)
    else:
        std = torch.zeros_like(mean)
    return mean + lambda_agg * std


def assign_channels(
    delta_tilde: torch.Tensor, fp_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign channels to FP (high distortion) / INT (low distortion).

    Returns ``(fp_indices, int_indices, perm, mask)`` where:
      * ``fp_indices``  : long, FP channels in *original* order, len K_F
      * ``int_indices`` : long, INT channels in *original* order, len C-K_F
      * ``perm``        : long, ``[fp_indices, int_indices]``  (len C)
      * ``mask``        : bool [C], True = FP
    """
    if delta_tilde.dim() != 1:
        raise ValueError("delta_tilde must be 1-D [C]")
    C = delta_tilde.numel()
    k_fp = int(fp_ratio * C)  # floor via int()
    k_fp = max(0, min(k_fp, C))

    # FP = top-K_F by distortion; INT = remaining bottom (C - K_F).
    if k_fp > 0:
        fp_idx = torch.topk(delta_tilde, k_fp, largest=True, sorted=True).indices
    else:
        fp_idx = torch.empty(0, dtype=torch.long, device=delta_tilde.device)
    return _finalize(C, fp_idx)


# --------------------------------------------------------------------------- #
# Ablation routers (random / magnitude) -- same FP/INT count.
# --------------------------------------------------------------------------- #
def random_routing(
    C: int, fp_ratio: float, generator: torch.Generator
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random FP/INT assignment with the same K_F as SADND (for ablation)."""
    k_fp = max(0, min(int(fp_ratio * C), C))
    perm_rand = torch.randperm(C, generator=generator)
    fp_idx = perm_rand[:k_fp]
    return _finalize(C, fp_idx)


def magnitude_routing(
    channel_magnitude: torch.Tensor, fp_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Magnitude-based routing: FP = top-K_F by mean |activation| (ablation).

    ``channel_magnitude`` is ``[C]`` (e.g. E[|u_c|] over calibration).
    """
    return assign_channels(channel_magnitude, fp_ratio)


def _finalize(C: int, fp_idx: torch.Tensor):
    """Build (fp, int, perm, mask) preserving ORIGINAL channel order in groups."""
    mask = torch.zeros(C, dtype=torch.bool, device=fp_idx.device)
    mask[fp_idx] = True
    int_idx = torch.nonzero(~mask, as_tuple=False).squeeze(-1)
    fp_idx_sorted = torch.sort(fp_idx).values.long()
    int_idx_sorted = torch.sort(int_idx).values.long()
    perm = torch.cat([fp_idx_sorted, int_idx_sorted], dim=0)
    mask = torch.zeros(C, dtype=torch.bool, device=fp_idx.device)
    mask[fp_idx_sorted] = True
    return fp_idx_sorted, int_idx_sorted, perm, mask
