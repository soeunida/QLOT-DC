"""Static Diagonal Compensation (Q-LOT-DC).

A *static*, calibration-time, per-INT-channel scale ``alpha_c`` that replaces the
token-dependent GroupRMS.  It is applied as a diagonal similarity transform that
preserves the projection function before quantization:

    activation:  y_c     -> alpha_c * y_c
    weight:      W[:, c]  -> W[:, c] / alpha_c          (nn.Linear is [out, in])

so that ``(alpha_c * y_c) * (W[:, c] / alpha_c) == y_c * W[:, c]`` exactly (in
full precision).  Choosing ``alpha_c`` to flatten the per-channel activation
scale makes the subsequent per-channel INT8 quantization more uniform, without
any token-dependent normalization (unlike GroupRMS).

``alpha_c`` is computed once per routed layer (shared by gate_proj and up_proj,
which consume the same INT activation) and is aligned to the packed INT channel
order (length ``k_int``).
"""

from __future__ import annotations

from typing import Optional

import torch


def compute_alpha(
    mode: str,
    a_c: torch.Tensor,                 # per-INT-channel activation scale [k_int]
    w_c: Optional[torch.Tensor] = None,  # per-INT-channel weight scale [k_int]
    beta: float = 0.5,
    alpha_min: float = 0.25,
    alpha_max: float = 4.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the static per-INT-channel compensation scale ``alpha`` [k_int].

    Modes
    -----
    median_scale:
        ``alpha_c = clamp(median(a) / (a_c + eps), alpha_min, alpha_max)``
        (flattens INT activation scales toward their median).
    smoothquant_like:
        ``alpha_c = clamp((a_c**beta) / ((w_c+eps)**(1-beta)), min, max)``,
        then normalized by its median and re-clamped.
    """
    a_c = a_c.float()
    if mode == "median_scale":
        s_target = a_c.median()
        alpha = (s_target / (a_c + eps)).clamp(alpha_min, alpha_max)
    elif mode == "smoothquant_like":
        if w_c is None:
            raise ValueError("smoothquant_like requires w_c")
        w_c = w_c.float()
        alpha = ((a_c ** beta) / ((w_c + eps) ** (1.0 - beta)))
        alpha = alpha.clamp(alpha_min, alpha_max)
        med = alpha.median()
        if float(med) > 0:
            alpha = alpha / med
        alpha = alpha.clamp(alpha_min, alpha_max)
    elif mode == "none":
        alpha = torch.ones_like(a_c)
    else:
        raise ValueError(f"unknown diag_comp_mode {mode!r}")
    return alpha


def apply_inverse_weight_scale(
    weight: torch.Tensor, int_indices: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    """Return a copy of ``weight`` ``[out, in]`` with INT columns divided by alpha.

    Only ``weight[:, int_indices]`` is modified (inverse of the activation scale,
    broadcast over output rows).  FP columns are untouched.
    """
    if weight.dim() != 2:
        raise ValueError("weight must be 2-D [out, in]")
    if alpha.numel() != int_indices.numel():
        raise ValueError("alpha length must match int_indices")
    out = weight.clone()
    inv = (1.0 / alpha.to(out.device, out.dtype))
    out[:, int_indices] = out[:, int_indices] * inv
    return out
