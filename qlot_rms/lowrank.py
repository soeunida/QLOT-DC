"""Low-rank residual correction for Q-LOT-DC+ (no gradient training).

For one routed projection we have, on a calibration token subset:
  X  = INT-branch activation actually fed to quant   [T, C_int]
  E  = Z_fp_reference - Z_quant_reference            [T, O]   (the quant residual)

We fit a low-rank correction ``X @ A @ B ≈ E`` with rank ``r`` (default 4) using
a closed-form least-squares solution + truncated SVD (no SGD):

  M  = argmin_M || X M - E ||_F              (M = lstsq(X, E), [C_int, O])
  M ≈ A B   via rank-r SVD of M              (A = U_r·S_r [C_int, r], B = V_r^T [r, O])

At inference: ``z_corrected = z_quant + (y_I @ A) @ B``. Optional and off by
default (adds an extra small matmul per routed projection). No speedup is claimed.
"""

from __future__ import annotations

from typing import Tuple

import torch


def fit_lowrank(X: torch.Tensor, E: torch.Tensor, rank: int
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return factors ``(A [C_int, r], B [r, O])`` with ``X @ A @ B ≈ E``.

    Closed-form: least-squares ``M = lstsq(X, E)`` then rank-r truncated SVD.
    Falls back to zero factors if inputs are degenerate.
    """
    X = X.float()
    E = E.float()
    C_int = X.shape[1]
    O = E.shape[1]
    r = int(max(1, min(rank, C_int, O)))
    try:
        M = torch.linalg.lstsq(X, E).solution            # [C_int, O]
        if not torch.isfinite(M).all():
            raise RuntimeError("non-finite lstsq solution")
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        r = min(r, S.numel())
        A = (U[:, :r] * S[:r]).contiguous()              # [C_int, r]
        B = Vh[:r, :].contiguous()                       # [r, O]
        return A, B
    except Exception:  # noqa: BLE001 -- degenerate; return no-op correction
        A = torch.zeros(C_int, r)
        B = torch.zeros(r, O)
        return A, B


def apply_lowrank(y_I: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Compute the correction ``(y_I @ A) @ B`` -> ``[..., O]`` (in y_I dtype)."""
    return torch.matmul(torch.matmul(y_I, A.to(y_I.dtype)), B.to(y_I.dtype))
