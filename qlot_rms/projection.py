"""Packed FP/INT projection (reference backend) + custom-kernel hook.

For each routed projection ``W`` in ``{gate_proj, up_proj}`` the input-channel
dimension is split by the FP and INT channel sets::

    W (nn.Linear weight, [out, Cin])
        W_F = W[:, fp_indices]    # FP16 slice
        W_I = W[:, int_indices]   # INT8 W8-G128 slice (mean-comp folded in)

The split is along the *input* dimension; the output dimension is unchanged.
The FP and INT branches produce partial outputs in the same output space and are
summed.  No inverse permutation is needed after the projection.

Branch inputs (computed once per layer, shared by gate_proj & up_proj):

    u (pre-affine, original channel order)
        u_F = u[:, fp_indices];  y_F = u_F * gamma_F + beta_F
        u_I = u[:, int_indices]; y_I = GroupRMS(u_I) * gamma_I + beta_I

Backends
--------
``torch_reference`` (default): correct, simulated INT8 + W8-G128, FP32 accum.
``custom_packed``  (stub):     raises NotImplementedError with a TODO list for a
                               future CUDA/Triton branched packed kernel.

No speedup is claimed or printed anywhere.
"""

from __future__ import annotations

import abc
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .grouprms import group_rms, broadcast_per_group_to_channels
from .diagcomp import apply_inverse_weight_scale
from .quant import (
    simulated_int8_matmul,
    quantize_activation_int8,
    fake_quantize_weight_w8_g128,
)


# --------------------------------------------------------------------------- #
# Mean-scale compensation (orientation-correct: modifies INPUT columns only)
# --------------------------------------------------------------------------- #
def apply_mean_scale_compensation(
    weight: torch.Tensor, int_indices: torch.Tensor, mu_g
) -> torch.Tensor:
    """Return a copy of ``weight`` ``[out, Cin]`` with INT *input* columns scaled.

    PyTorch ``nn.Linear`` stores weight as ``[out_features, in_features]``, so the
    INT input-channel slice is ``weight[:, int_indices]`` (columns), NOT rows.
    Only those columns are scaled; all other entries (including the FP input
    columns) are left unchanged.

    ``mu_g`` may be a scalar (single mean RMS) or a per-INT-channel tensor of
    length ``len(int_indices)`` (per-group compensation, broadcast across output
    rows).  Per-group is the correct mode; the scalar form is kept for
    backward-compat / tests.
    """
    if weight.dim() != 2:
        raise ValueError("weight must be 2-D [out, in]")
    out = weight.clone()
    if torch.is_tensor(mu_g) and mu_g.dim() == 1:
        scale = mu_g.to(out.device, out.dtype)            # [C_int]
        if scale.numel() != int_indices.numel():
            raise ValueError("per-channel mu_g length must match int_indices")
        out[:, int_indices] = out[:, int_indices] * scale  # broadcast over rows
    else:
        out[:, int_indices] = out[:, int_indices] * float(mu_g)
    return out


# --------------------------------------------------------------------------- #
# Branch inputs (shared between gate_proj and up_proj)
# --------------------------------------------------------------------------- #
def compute_branch_inputs(
    u: torch.Tensor,
    fp_indices: torch.Tensor,
    int_indices: torch.Tensor,
    gamma: torch.Tensor,
    beta: Optional[torch.Tensor],
    grms_group_size: int,
    eps: float,
    use_grms: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(y_F, y_I)`` from pre-affine ``u`` (original channel order).

    * FP branch keeps full precision; affine applied directly.
    * INT branch: GroupRMS over contiguous INT groups (if ``use_grms``), then
      affine.  When ``use_grms`` is False this is a plain static-routing INT
      branch (ablation): ``y_I = u_I * gamma_I + beta_I``.
    """
    fp = fp_indices.to(u.device)
    intc = int_indices.to(u.device)
    gamma_F = gamma.index_select(0, fp.to(gamma.device)).to(u.dtype)
    gamma_I = gamma.index_select(0, intc.to(gamma.device)).to(u.dtype)
    beta_F = beta.index_select(0, fp.to(beta.device)).to(u.dtype) if beta is not None else None
    beta_I = beta.index_select(0, intc.to(beta.device)).to(u.dtype) if beta is not None else None
    return branch_inputs_from_slices(
        u, fp, intc, gamma_F, gamma_I, beta_F, beta_I, grms_group_size, eps, use_grms
    )


def branch_inputs_from_slices(
    u: torch.Tensor,
    fp_indices: torch.Tensor,
    int_indices: torch.Tensor,
    gamma_F: torch.Tensor,
    gamma_I: torch.Tensor,
    beta_F: Optional[torch.Tensor],
    beta_I: Optional[torch.Tensor],
    grms_group_size: int,
    eps: float,
    use_grms: bool = True,
    int_alpha: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fast branch inputs from PRECOMPUTED affine slices and device indices.

    Identical math to :func:`compute_branch_inputs` but does no per-call
    index_select on gamma/beta and no device transfers (the caller supplies
    already-sliced, already-on-device tensors).  Only the dynamic activation
    ``u`` is split per call (unavoidable).

    ``int_alpha`` (Q-LOT-DC) is an optional per-INT-channel static scale applied
    to the INT activation AFTER affine (``y_I *= alpha``); the matching inverse
    is folded into the INT weight columns at packing, preserving the projection
    function before quantization.
    """
    u_F = u.index_select(-1, fp_indices)
    u_I = u.index_select(-1, int_indices)
    y_F = u_F * gamma_F
    base_I = group_rms(u_I, grms_group_size, eps=eps) if use_grms else u_I
    y_I = base_I * gamma_I
    if beta_F is not None:
        y_F = y_F + beta_F
    if beta_I is not None:
        y_I = y_I + beta_I
    if int_alpha is not None:
        y_I = y_I * int_alpha
    return y_F, y_I


# --------------------------------------------------------------------------- #
# Backend interface
# --------------------------------------------------------------------------- #
class ProjectionBackend(abc.ABC):
    """Abstraction so a future custom kernel can replace the reference path."""

    name: str = "abstract"

    @abc.abstractmethod
    def fp_matmul(self, y_F: torch.Tensor, W_F: torch.Tensor) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def int_matmul(
        self,
        y_I: torch.Tensor,
        W_I: torch.Tensor,
        act_scales: torch.Tensor,
        w_group_size: int,
        qmax: int,
    ) -> torch.Tensor:
        ...


class TorchReferenceBackend(ProjectionBackend):
    """Correct, simulated reference.  This is the default backend."""

    name = "torch_reference"

    def fp_matmul(self, y_F: torch.Tensor, W_F: torch.Tensor) -> torch.Tensor:
        if W_F.numel() == 0 or y_F.shape[-1] == 0:
            # no FP channels: produce a zero partial output of the right shape
            out_features = W_F.shape[0]
            return y_F.new_zeros((*y_F.shape[:-1], out_features))
        return torch.matmul(y_F, W_F.t())

    def int_matmul(
        self,
        y_I: torch.Tensor,
        W_I: torch.Tensor,
        act_scales: torch.Tensor,
        w_group_size: int,
        qmax: int,
    ) -> torch.Tensor:
        if W_I.numel() == 0 or y_I.shape[-1] == 0:
            out_features = W_I.shape[0]
            return y_I.new_zeros((*y_I.shape[:-1], out_features))
        return simulated_int8_matmul(
            y_I, W_I, act_scales, w_group_size=w_group_size, qmax=qmax
        )


class CustomPackedBackend(ProjectionBackend):
    """Clean stub for a future CUDA/Triton branched packed kernel.

    TODO (custom kernel, not yet implemented):
      1. Pack ``W_F`` (FP16, [out, K_F]) and ``W_I`` (INT8 W8-G128, [out, C-K_F]
         + per-group fp16 scales) into a single contiguous packed buffer with a
         static FP/INT split offset (= K_F) baked in at pack time.
      2. Implement a fused branched matmul that, for each output tile, runs the
         FP16 partial product over the first K_F input channels and the INT8
         W8-G128 partial product (per-channel activation scale * per-group weight
         scale, INT32 accumulation) over the remaining channels, summing into one
         FP32 accumulator, then writes FP16.
      3. Apply per-input-channel activation quantization (frozen ``act_scales``)
         inside the kernel prologue; GroupRMS + affine remain a separate
         elementwise pre-pass (or a fused epilogue of the preceding norm).
      4. Expose the same ``fp_matmul`` / ``int_matmul`` (or a single fused
         ``packed_matmul``) signature so this class is a drop-in for
         TorchReferenceBackend.  Verify against the reference within tolerance
         before enabling.

    Until implemented, instantiating and calling this backend raises so callers
    never silently fall through to an unverified path.
    """

    name = "custom_packed"

    _MSG = (
        "custom_packed backend is a stub. Use backend='torch_reference'. "
        "See CustomPackedBackend docstring for the CUDA/Triton TODO list."
    )

    experimental = True

    @staticmethod
    def available() -> bool:
        """True only if a real packed kernel can actually run (Triton + CUDA)."""
        try:
            from .kernels.triton_packed import triton_available
            return triton_available()
        except Exception:  # noqa: BLE001
            return False

    def fp_matmul(self, *args, **kwargs):  # noqa: D401
        # The elementwise-split (fp_matmul/int_matmul) path is NOT wired for the
        # custom backend; use packed_forward (a fused, branched entry) instead.
        raise NotImplementedError(self._MSG)

    def int_matmul(self, *args, **kwargs):  # noqa: D401
        raise NotImplementedError(self._MSG)

    def packed_forward(self, *args, **kwargs):
        return custom_packed_forward(*args, **kwargs)


def custom_packed_forward(
    x_fp,            # [T, K_F]  FP-branch activation (y_F)
    x_int,           # [T, C_int] INT-branch activation (y_I, post-affine/alpha)
    w_fp,            # [O, K_F]  FP weight slice
    w_int_packed,    # [O, C_int] packed/fake-quant INT weight (alpha-inversed)
    act_scales,      # [C_int]   frozen per-input-channel activation scales
    weight_scales=None,   # per-group weight scales (unused by the fp32 prototype)
    alpha=None,           # [C_int] Q-LOT-DC scale (already folded; informational)
    bias=None,            # [O]
    bias_corr=None,       # [O]
    metadata=None,        # dict (group_size, qmax, ...)
):
    """EXPERIMENTAL fused branched forward for ONE routed projection.

    Drop-in entry for the future ``custom_packed`` kernel. The current prototype
    dispatches to the Triton FP32-GEMM scaffold (``kernels/triton_packed``),
    matching the ``torch_reference`` cached path within tolerance. Raises
    ``NotImplementedError`` if no real kernel is available (never silently falls
    back). **No speedup is claimed.**
    """
    from .kernels.triton_packed import packed_projection_forward, triton_available
    if not triton_available():
        raise NotImplementedError(
            "custom_packed is experimental and requires a working Triton+CUDA "
            "kernel; none is available. Use backend='torch_reference'.")
    md = metadata or {}
    qmax = int(md.get("qmax", 127))
    yq = quantize_activation_int8(x_int, act_scales, qmax=qmax).float()
    return packed_projection_forward(
        x_fp=x_fp, yq=yq, w_fp=w_fp, w_int_dq=w_int_packed,
        bias=bias, bias_corr=bias_corr,
    )


def get_backend(name: str) -> ProjectionBackend:
    if name == "torch_reference":
        return TorchReferenceBackend()
    if name == "custom_packed":
        return CustomPackedBackend()
    raise ValueError(f"unknown backend {name!r}")


# --------------------------------------------------------------------------- #
# Packed projection module
# --------------------------------------------------------------------------- #
class PackedProjection(nn.Module):
    """Static packed FP/INT projection for one routed Linear (gate or up).

    Built from an existing ``nn.Linear`` plus the layer's :class:`LayerRouting`.
    Stores FP and INT weight slices separately (INT slice carries the folded
    mean-scale compensation).  ``forward`` accepts the pre-affine activation
    ``u`` (original channel order) and returns the summed partial outputs.
    """

    def __init__(
        self,
        W_F: torch.Tensor,
        W_I: torch.Tensor,
        bias: Optional[torch.Tensor],
        fp_indices: torch.Tensor,
        int_indices: torch.Tensor,
        gamma: torch.Tensor,
        beta: Optional[torch.Tensor],
        act_scales: torch.Tensor,
        grms_group_size: int,
        eps: float,
        w_group_size: int,
        qmax: int,
        backend: ProjectionBackend,
        out_features: int,
        use_grms: bool = True,
        cache_dequant_weight: bool = False,
        diag_alpha: Optional[torch.Tensor] = None,
        bias_corr: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.register_buffer("W_F", W_F, persistent=False)
        self.register_buffer("W_I", W_I, persistent=False)
        self.register_buffer("bias", bias if bias is not None else None, persistent=False)
        self.register_buffer("fp_indices", fp_indices, persistent=False)
        self.register_buffer("int_indices", int_indices, persistent=False)
        self.register_buffer("gamma", gamma, persistent=False)
        self.register_buffer("beta", beta if beta is not None else None, persistent=False)
        self.register_buffer("act_scales", act_scales, persistent=False)
        # Q-LOT-DC: static per-INT-channel activation scale (applied to y_I before
        # quant; its inverse is already folded into W_I at packing). bias_corr is
        # an optional per-output projection bias correction added to the output.
        self.register_buffer(
            "diag_alpha", diag_alpha if diag_alpha is not None else None, persistent=False)
        self.register_buffer(
            "bias_corr", bias_corr if bias_corr is not None else None, persistent=False)
        self.grms_group_size = grms_group_size
        self.eps = eps
        self.w_group_size = w_group_size
        self.qmax = qmax
        self.backend = backend
        self.out_features = out_features
        self.use_grms = use_grms

        # --- precompute affine slices once (avoids per-forward index_select) ---
        # kept in fp32 so the fp32 pre-affine activation multiplies cleanly.
        g = gamma.detach().float()
        self.register_buffer("gamma_F", g.index_select(0, fp_indices), persistent=False)
        self.register_buffer("gamma_I", g.index_select(0, int_indices), persistent=False)
        if beta is not None:
            b = beta.detach().float()
            self.register_buffer("beta_F", b.index_select(0, fp_indices), persistent=False)
            self.register_buffer("beta_I", b.index_select(0, int_indices), persistent=False)
        else:
            self.beta_F = None
            self.beta_I = None

        # --- reference-only: cache the fake-quantized W8-G128 weight once ---
        # Numerically identical to re-quantizing every forward (the weight is
        # static).  Only valid for the torch_reference backend.
        self._use_cached_dequant = bool(
            cache_dequant_weight and isinstance(backend, TorchReferenceBackend)
        )
        if self._use_cached_dequant and W_I.numel() > 0:
            W_I_dq = fake_quantize_weight_w8_g128(
                W_I, group_size=w_group_size, qmax=qmax
            ).float()
            self.register_buffer("W_I_dq", W_I_dq, persistent=False)
        else:
            self.W_I_dq = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        routing,                 # LayerRouting
        gamma: torch.Tensor,
        beta: Optional[torch.Tensor],
        cfg,                     # QLotRmsConfig
        backend: Optional[ProjectionBackend] = None,
        apply_mean_comp: Optional[bool] = None,
        bias_corr: Optional[torch.Tensor] = None,
    ) -> "PackedProjection":
        if backend is None:
            backend = get_backend(cfg.backend)
        # per-layer GroupRMS decision (defaults to True if the field is absent)
        layer_grms = bool(cfg.use_grms and getattr(routing, "grms_enabled", True))
        if apply_mean_comp is None:
            # mean-scale compensation only makes sense alongside GroupRMS, and is
            # gated by the same per-layer decision.
            apply_mean_comp = bool(cfg.use_mean_comp and layer_grms)
        W = linear.weight.data
        out_features, Cin = W.shape
        device = W.device

        fp_idx = routing.fp_indices.to(device)
        int_idx = routing.int_indices.to(device)

        # mean-scale compensation: fold per-group mu_g into INT *input columns*.
        if apply_mean_comp:
            if routing.mu_g_channels is not None:
                mu = routing.mu_g_channels.to(device)
            elif torch.is_tensor(routing.mu_g) and \
                    routing.mu_g.numel() == routing.grms_num_groups:
                # broadcast per-group mu_g to INT channels
                mu = broadcast_per_group_to_channels(
                    routing.mu_g.tolist(), int_idx.numel(), routing.grms_group_size
                ).to(device)
            else:
                mu = routing.mu_g  # scalar fallback (tests / legacy)
            W_comp = apply_mean_scale_compensation(W, int_idx, mu)
        else:
            W_comp = W

        # Q-LOT-DC: fold inverse alpha into INT weight columns (mutually exclusive
        # with mean-comp, which is gated by use_grms). Preserves the projection
        # function: (alpha*y) @ (W/alpha)^T == y @ W^T.
        diag_alpha = None
        if getattr(routing, "diag_comp_applied", False) and routing.diag_alpha is not None:
            diag_alpha = routing.diag_alpha.to(device).float()
            W_comp = apply_inverse_weight_scale(W_comp, int_idx, diag_alpha)

        W_F = W[:, fp_idx].contiguous().to(torch.float16)
        W_I = W_comp[:, int_idx].contiguous().to(torch.float16)

        bias = linear.bias.data.clone() if linear.bias is not None else None
        act_scales = routing.act_scales.to(device).float()

        return cls(
            W_F=W_F,
            W_I=W_I,
            bias=bias,
            fp_indices=fp_idx,
            int_indices=int_idx,
            gamma=gamma.to(device),
            beta=beta.to(device) if beta is not None else None,
            act_scales=act_scales,
            grms_group_size=routing.grms_group_size,
            eps=cfg.eps,
            w_group_size=cfg.w8_group_size,
            qmax=cfg.qmax,
            backend=backend,
            out_features=out_features,
            use_grms=layer_grms,
            cache_dequant_weight=getattr(cfg, "cache_dequant_weight", True),
            diag_alpha=diag_alpha,
            bias_corr=bias_corr.to(device).float() if bias_corr is not None else None,
        )

    def forward_from_branches(
        self, y_F: torch.Tensor, y_I: torch.Tensor
    ) -> torch.Tensor:
        z_F = self.backend.fp_matmul(y_F.to(self.W_F.dtype), self.W_F)
        if self.W_I_dq is not None:
            # reference-only fast path: weight already fake-quantized + cached.
            # Identical math to simulated_int8_matmul (quantize act, FP32 matmul).
            if self.W_I_dq.numel() == 0 or y_I.shape[-1] == 0:
                z_I = y_I.new_zeros((*y_I.shape[:-1], self.out_features))
            else:
                yq = quantize_activation_int8(y_I, self.act_scales, qmax=self.qmax).float()
                z_I = torch.matmul(yq, self.W_I_dq.t())
        else:
            z_I = self.backend.int_matmul(
                y_I, self.W_I, self.act_scales, self.w_group_size, self.qmax
            )
        z = z_F.float() + z_I.float()
        if self.bias is not None:
            z = z + self.bias.float()
        if self.bias_corr is not None:
            z = z + self.bias_corr.float()
        return z.to(torch.float16)

    def matmul_shared(self, y_F16: torch.Tensor, yq: torch.Tensor) -> torch.Tensor:
        """Cached-path matmul from PRE-quantized branch inputs.

        ``y_F16`` is the FP branch input already cast to fp16; ``yq`` is the INT
        branch activation already fake-quantized (and float) with this layer's
        shared ``act_scales``.  gate_proj and up_proj share the same ``y_I`` and
        ``act_scales``, so the caller quantizes ONCE and reuses for both,
        avoiding a redundant per-projection quantization.  Numerically identical
        to :meth:`forward_from_branches` (cached path).  Requires the cached
        dequant weight; raises otherwise.
        """
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
        if self.bias_corr is not None:
            z = z + self.bias_corr.float()
        return z.to(torch.float16)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Self-contained forward from pre-affine ``u`` (original channel order)."""
        alpha = self.diag_alpha.to(u.dtype) if self.diag_alpha is not None else None
        y_F, y_I = branch_inputs_from_slices(
            u,
            self.fp_indices,
            self.int_indices,
            self.gamma_F.to(u.dtype),
            self.gamma_I.to(u.dtype),
            self.beta_F.to(u.dtype) if self.beta_F is not None else None,
            self.beta_I.to(u.dtype) if self.beta_I is not None else None,
            self.grms_group_size,
            self.eps,
            use_grms=self.use_grms,
            int_alpha=alpha,
        )
        return self.forward_from_branches(y_F, y_I)
