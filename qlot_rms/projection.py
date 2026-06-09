"""Packed FP/INT projection (reference backend) for SADND-CAP.

For each routed projection ``W`` (gate_proj, up_proj) the input-channel dim is
split by the static FP/INT sets:

    W_F = W[:, fp_indices]    # FP16 slice
    W_I = W[:, int_indices]   # INT8 W8-G128 slice (int_indices in packing-aware order)

The FP and INT partial outputs are summed in the same output space; no inverse
permutation is needed (channels are pre-permuted ``[FP, INT]``). There are NO
correction modules (no GroupRMS / diagonal / bias / low-rank / block correction).

Backends:
  torch_reference  -- default, correct, simulated INT8 + W8-G128 (FP32 accum).
  custom_packed    -- pure stub (NotImplementedError); no kernel is implemented.

No speedup is claimed: torch_reference is fake-quantized (dequant + FP matmul).
"""

from __future__ import annotations

import abc
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .quant import (
    simulated_int8_matmul,
    quantize_activation_int8,
    fake_quantize_weight_w8_g128,
)


# --------------------------------------------------------------------------- #
# Branch inputs (affine only; no correction)
# --------------------------------------------------------------------------- #
def branch_inputs_from_slices(
    u: torch.Tensor,
    fp_indices: torch.Tensor,
    int_indices: torch.Tensor,
    gamma_F: torch.Tensor,
    gamma_I: torch.Tensor,
    beta_F: Optional[torch.Tensor],
    beta_I: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(y_F, y_I)`` from pre-affine ``u`` using precomputed affine slices."""
    u_F = u.index_select(-1, fp_indices)
    u_I = u.index_select(-1, int_indices)
    y_F = u_F * gamma_F
    y_I = u_I * gamma_I
    if beta_F is not None:
        y_F = y_F + beta_F
    if beta_I is not None:
        y_I = y_I + beta_I
    return y_F, y_I


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class ProjectionBackend(abc.ABC):
    name = "abstract"

    @abc.abstractmethod
    def fp_matmul(self, y_F, W_F): ...

    @abc.abstractmethod
    def int_matmul(self, y_I, W_I, act_scales, w_group_size, qmax): ...


class TorchReferenceBackend(ProjectionBackend):
    """Default correctness backend (simulated, FP32 accumulation)."""

    name = "torch_reference"

    def fp_matmul(self, y_F, W_F):
        if W_F.numel() == 0 or y_F.shape[-1] == 0:
            return y_F.new_zeros((*y_F.shape[:-1], W_F.shape[0]))
        return torch.matmul(y_F, W_F.t())

    def int_matmul(self, y_I, W_I, act_scales, w_group_size, qmax):
        if W_I.numel() == 0 or y_I.shape[-1] == 0:
            return y_I.new_zeros((*y_I.shape[:-1], W_I.shape[0]))
        return simulated_int8_matmul(y_I, W_I, act_scales,
                                     w_group_size=w_group_size, qmax=qmax)


class CustomPackedBackend(ProjectionBackend):
    """Pure stub for a future CUDA/Triton packed FP16+INT8 kernel.

    Not implemented. Calling raises so callers never fall through to an
    unverified path. A real kernel must pass correctness tests vs the reference
    before being enabled. No speedup is implied.
    """

    name = "custom_packed"
    experimental = True
    _MSG = ("custom_packed backend is a stub (no kernel implemented). "
            "Use backend='torch_reference'.")

    @staticmethod
    def available() -> bool:
        return False

    def fp_matmul(self, *a, **k):
        raise NotImplementedError(self._MSG)

    def int_matmul(self, *a, **k):
        raise NotImplementedError(self._MSG)


def get_backend(name: str) -> ProjectionBackend:
    if name == "torch_reference":
        return TorchReferenceBackend()
    if name == "custom_packed":
        return CustomPackedBackend()
    raise ValueError(f"unknown backend {name!r}")


# --------------------------------------------------------------------------- #
# Packed projection
# --------------------------------------------------------------------------- #
class PackedProjection(nn.Module):
    """Static packed FP/INT projection for one routed Linear (gate or up)."""

    def __init__(self, W_F, W_I, bias, fp_indices, int_indices, gamma_F, gamma_I,
                 beta_F, beta_I, act_scales, w_group_size, qmax, backend, out_features,
                 cache_dequant_weight=False):
        super().__init__()
        self.register_buffer("W_F", W_F, persistent=False)
        self.register_buffer("W_I", W_I, persistent=False)
        self.register_buffer("bias", bias if bias is not None else None, persistent=False)
        self.register_buffer("fp_indices", fp_indices, persistent=False)
        self.register_buffer("int_indices", int_indices, persistent=False)
        self.register_buffer("gamma_F", gamma_F, persistent=False)
        self.register_buffer("gamma_I", gamma_I, persistent=False)
        self.register_buffer("beta_F", beta_F if beta_F is not None else None, persistent=False)
        self.register_buffer("beta_I", beta_I if beta_I is not None else None, persistent=False)
        self.register_buffer("act_scales", act_scales, persistent=False)
        self.w_group_size = w_group_size
        self.qmax = qmax
        self.backend = backend
        self.out_features = out_features
        self._use_cached_dequant = bool(
            cache_dequant_weight and isinstance(backend, TorchReferenceBackend))
        if self._use_cached_dequant and W_I.numel() > 0:
            self.register_buffer("W_I_dq", fake_quantize_weight_w8_g128(
                W_I, group_size=w_group_size, qmax=qmax).float(), persistent=False)
        else:
            self.W_I_dq = None

    @classmethod
    def from_linear(cls, linear, routing, gamma, beta, cfg, backend=None, int_gain=None):
        """``int_gain`` (optional, [C-K_F] in packed INT order) is a static
        SADND-CAP-GT output gain folded into the INT weight columns:
        ``z = z_FP + (yq) @ (W_I * int_gain)^T``. None => unchanged."""
        if backend is None:
            backend = get_backend(cfg.backend)
        W = linear.weight.data
        out_features, _ = W.shape
        dev = W.device
        fp = routing.fp_indices.to(dev)
        intc = routing.int_indices.to(dev)
        W_F = W[:, fp].contiguous().to(torch.float16)
        W_I = W[:, intc].contiguous().to(torch.float16)
        if int_gain is not None and W_I.numel() > 0:
            gain_vec = int_gain.to(dev).float().reshape(1, -1)   # per-INT-channel gain
            W_I = (W_I.float() * gain_vec).contiguous().to(torch.float16)
        g = gamma.detach().float()
        gamma_F = g.index_select(0, fp)
        gamma_I = g.index_select(0, intc)
        if beta is not None:
            b = beta.detach().float()
            beta_F = b.index_select(0, fp)
            beta_I = b.index_select(0, intc)
        else:
            beta_F = beta_I = None
        bias = linear.bias.data.clone() if linear.bias is not None else None
        return cls(
            W_F=W_F, W_I=W_I, bias=bias, fp_indices=fp, int_indices=intc,
            gamma_F=gamma_F, gamma_I=gamma_I, beta_F=beta_F, beta_I=beta_I,
            act_scales=routing.act_scales.to(dev).float(),
            w_group_size=cfg.w8_group_size, qmax=cfg.qmax, backend=backend,
            out_features=out_features,
            cache_dequant_weight=getattr(cfg, "cache_dequant_weight", True))

    def forward_from_branches(self, y_F, y_I):
        z_F = self.backend.fp_matmul(y_F.to(self.W_F.dtype), self.W_F)
        if self.W_I_dq is not None:
            if self.W_I_dq.numel() == 0 or y_I.shape[-1] == 0:
                z_I = y_I.new_zeros((*y_I.shape[:-1], self.out_features))
            else:
                yq = quantize_activation_int8(y_I, self.act_scales, qmax=self.qmax).float()
                z_I = torch.matmul(yq, self.W_I_dq.t())
        else:
            z_I = self.backend.int_matmul(y_I, self.W_I, self.act_scales,
                                          self.w_group_size, self.qmax)
        z = z_F.float() + z_I.float()
        if self.bias is not None:
            z = z + self.bias.float()
        return z.to(torch.float16)

    def matmul_shared(self, y_F16, yq):
        """Cached-path matmul from pre-quantized branch inputs (shared gate/up)."""
        if self.W_I_dq is None:
            raise RuntimeError("matmul_shared requires cache_dequant_weight (torch_reference)")
        if self.W_F.numel() == 0 or y_F16.shape[-1] == 0:
            z_F = y_F16.new_zeros((*y_F16.shape[:-1], self.out_features))
        else:
            z_F = torch.matmul(y_F16, self.W_F.t())
        if self.W_I_dq.numel() == 0 or yq.shape[-1] == 0:
            z_I = yq.new_zeros((*yq.shape[:-1], self.out_features))
        else:
            z_I = torch.matmul(yq, self.W_I_dq.t())
        z = z_F.float() + z_I.float()
        if self.bias is not None:
            z = z + self.bias.float()
        return z.to(torch.float16)

    def forward(self, u):
        y_F, y_I = branch_inputs_from_slices(
            u, self.fp_indices, self.int_indices,
            self.gamma_F.to(u.dtype), self.gamma_I.to(u.dtype),
            self.beta_F.to(u.dtype) if self.beta_F is not None else None,
            self.beta_I.to(u.dtype) if self.beta_I is not None else None)
        return self.forward_from_branches(y_F, y_I)
