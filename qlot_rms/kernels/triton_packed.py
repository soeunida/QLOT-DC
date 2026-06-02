"""Minimal Triton prototype for ONE Q-LOT-DC routed projection (EXPERIMENTAL).

Scope (intentionally narrow):
  * prefill, batch=1 first (works for any [tokens, C] 2-D activation)
  * one Linear projection (gate_proj OR up_proj)
  * CORRECTNESS first, speed second -- this is an integration scaffold, NOT an
    optimized kernel. **No speedup is claimed.**

Numerics: matches the ``torch_reference`` cached path exactly in intent --
FP32-accumulation GEMMs over (a) the FP branch ``x_fp @ W_F^T`` and (b) the INT
branch ``yq @ W_I_dq^T`` where ``yq`` is the per-channel fake-quantized activation
and ``W_I_dq`` the fake-quantized (and alpha-inversed / mean-comp'd) weight. The
partial outputs are summed in FP32 and cast to FP16, plus optional bias and
projection-bias-correction.

A true INT8 (INT32-accumulation) W8-G128 kernel is the next stage; see
``docs/serving_layout_and_kernel_plan.md``. If Triton is unavailable this module
degrades gracefully (``triton_available()`` returns False and callers raise a
clear NotImplementedError).
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # noqa: BLE001
    _HAS_TRITON = False


def triton_available() -> bool:
    """True if Triton is importable AND a CUDA device is present."""
    return bool(_HAS_TRITON and torch.cuda.is_available())


if _HAS_TRITON:

    @triton.jit
    def _matmul_fp32_kernel(
        A, B, C,
        M, N, K,
        sam, sak, sbk, sbn, scm, scn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        """C[M,N] = A[M,K] @ B[K,N], FP32 accumulation. Inputs read as fp32."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        a_ptrs = A + (offs_m[:, None] * sam + offs_k[None, :] * sak)
        b_ptrs = B + (offs_k[:, None] * sbk + offs_n[None, :] * sbn)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            k_mask = (offs_k[None, :] + k0) < K
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask, other=0.0).to(tl.float32)
            b_mask = ((offs_k[:, None] + k0) < K) & (offs_n[None, :] < N)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)
            acc += tl.dot(a, b, allow_tf32=False)
            a_ptrs += BK * sak
            b_ptrs += BK * sbk
        c_ptrs = C + (offs_m[:, None] * scm + offs_n[None, :] * scn)
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """FP32 GEMM a[M,K] @ b[K,N] -> [M,N] via Triton (tf32 disabled)."""
        assert a.is_cuda and b.is_cuda and a.shape[1] == b.shape[0]
        M, K = a.shape
        N = b.shape[1]
        a = a.contiguous().float()
        b = b.contiguous().float()
        c = torch.empty((M, N), device=a.device, dtype=torch.float32)
        BM, BN, BK = 64, 64, 32
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        _matmul_fp32_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
            BM=BM, BN=BN, BK=BK,
        )
        return c


def packed_projection_forward(
    x_fp: torch.Tensor,        # [T, K_F] fp16/fp32 FP-branch activation
    yq: torch.Tensor,          # [T, C_int] fake-quantized INT activation (float)
    w_fp: torch.Tensor,        # [O, K_F] FP weight
    w_int_dq: torch.Tensor,    # [O, C_int] fake-quantized (alpha-inversed) INT weight
    bias: Optional[torch.Tensor] = None,        # [O]
    bias_corr: Optional[torch.Tensor] = None,   # [O]
) -> torch.Tensor:
    """One-projection branched forward using the Triton FP32 GEMM. Returns fp16.

    Requires CUDA + Triton; raises NotImplementedError otherwise (callers must
    not silently fall back).
    """
    if not triton_available():
        raise NotImplementedError(
            "Triton/CUDA unavailable; custom_packed kernel cannot run. "
            "Use backend='torch_reference'.")
    O = w_fp.shape[0] if w_fp.numel() else w_int_dq.shape[0]
    T = x_fp.shape[0] if x_fp.numel() else yq.shape[0]
    z = torch.zeros((T, O), device=yq.device, dtype=torch.float32)
    if w_fp.numel() and x_fp.shape[-1] > 0:
        z = z + triton_matmul(x_fp.float(), w_fp.float().t().contiguous())
    if w_int_dq.numel() and yq.shape[-1] > 0:
        z = z + triton_matmul(yq.float(), w_int_dq.float().t().contiguous())
    if bias is not None:
        z = z + bias.float()
    if bias_corr is not None:
        z = z + bias_corr.float()
    return z.to(torch.float16)
