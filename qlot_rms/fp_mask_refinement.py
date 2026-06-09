"""SADND-CAP++ : equal-budget FP mask refinement.

SADND-CAP+ decides *how much* FP budget each layer receives. SADND-CAP++ keeps
that exact budget but refines *which* channels are FP inside each layer: it swaps
weak FP channels for strong INT candidates near the FP/INT boundary, accepting a
swap only if a measured proxy error improves. The FP count per layer (``k_fp``)
is preserved exactly, so the global FP budget and the static serving layout are
unchanged. The refined mask is fully static at inference (no runtime
top-k / sort / dynamic routing). This is a policy refinement, NOT a correction
module, and no speedup is claimed.

The proxy measures the relative L2 error of the MLP output caused by INT8
*activation* quantization of the INT channels (vs the FP16 MLP output), over a
small captured token batch. The routing score that drove CAP+ selection is a
cheap heuristic (proxy distortion x weight-norm); near the boundary the measured
MLP-output error can disagree with it, and that is exactly where a swap can help.
``hidden_l2`` and ``mlp_output_l2`` both use this local MLP-output proxy
(``hidden_l2`` is a documented local approximation of the residual hidden);
``validation_ppl`` is not evaluated here -- it is only used for the coarse final
selection in ``eval/select_sadnd_cap.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Proxy context (one routed layer)
# --------------------------------------------------------------------------- #
@dataclass
class LayerRefineContext:
    """Everything needed to measure a candidate FP mask's MLP-output error.

    ``y`` is the affine activation ``u*gamma + beta`` ([N, C]); ``scales`` is the
    per-channel symmetric INT8 activation scale (length C, used for INT channels
    only). The FP16 reference output ``out_fp`` is precomputed once.
    """
    y: torch.Tensor                 # [N, C] affine activation (device)
    scales: torch.Tensor            # [C] per-channel act scale (device)
    w_gate: torch.Tensor            # [I, C]
    w_up: torch.Tensor              # [I, C]
    w_down: torch.Tensor            # [H, I]
    b_gate: Optional[torch.Tensor] = None
    b_up: Optional[torch.Tensor] = None
    b_down: Optional[torch.Tensor] = None
    qmax: int = 127
    eps: float = 1e-8
    out_fp: torch.Tensor = field(default=None, repr=False)
    out_fp_norm: float = 0.0

    def __post_init__(self):
        if self.out_fp is None:
            self.out_fp = self._mlp_forward(self.y)
            self.out_fp_norm = float(self.out_fp.norm())

    def _mlp_forward(self, y: torch.Tensor) -> torch.Tensor:
        gate = y @ self.w_gate.t()
        if self.b_gate is not None:
            gate = gate + self.b_gate
        up = y @ self.w_up.t()
        if self.b_up is not None:
            up = up + self.b_up
        h = F.silu(gate) * up
        out = h @ self.w_down.t()
        if self.b_down is not None:
            out = out + self.b_down
        return out


def build_refine_context_from_weights(
    y: torch.Tensor, scales: torch.Tensor,
    w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
    b_gate=None, b_up=None, b_down=None, qmax: int = 127, eps: float = 1e-8,
    max_tokens: int = 256,
) -> LayerRefineContext:
    if y.shape[0] > max_tokens:
        y = y[:max_tokens]
    return LayerRefineContext(
        y=y.float(), scales=scales.float(), w_gate=w_gate.float(), w_up=w_up.float(),
        w_down=w_down.float(),
        b_gate=None if b_gate is None else b_gate.float(),
        b_up=None if b_up is None else b_up.float(),
        b_down=None if b_down is None else b_down.float(),
        qmax=qmax, eps=eps)


def build_refine_context(y, scales, mlp, device="cpu", qmax=127, eps=1e-8,
                         max_tokens=256) -> LayerRefineContext:
    """Build a context from a HF MLP module (gate_proj/up_proj/down_proj)."""
    def _w(lin):
        return lin.weight.data.to(device).float()

    def _b(lin):
        return None if lin.bias is None else lin.bias.data.to(device).float()
    return build_refine_context_from_weights(
        y.to(device), scales.to(device), _w(mlp.gate_proj), _w(mlp.up_proj),
        _w(mlp.down_proj), _b(mlp.gate_proj), _b(mlp.up_proj), _b(mlp.down_proj),
        qmax=qmax, eps=eps, max_tokens=max_tokens)


def _fake_quant_int_channels(y: torch.Tensor, scales: torch.Tensor,
                             fp_mask: torch.Tensor, qmax: int) -> torch.Tensor:
    """Per-channel symmetric INT8 fake-quant of the INT (``~fp_mask``) channels."""
    s = scales.clamp_min(1e-12).unsqueeze(0)             # [1, C]
    q = torch.clamp(torch.round(y / s), -qmax, qmax) * s  # quantized everywhere
    return torch.where(fp_mask.unsqueeze(0), y, q)        # keep FP channels exact


# --------------------------------------------------------------------------- #
# B. proxy evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_mask_proxy(ctx: LayerRefineContext, fp_mask: torch.Tensor) -> float:
    """Relative L2 error of the MLP output under ``fp_mask`` vs FP16 (lower better)."""
    y_q = _fake_quant_int_channels(ctx.y, ctx.scales, fp_mask, ctx.qmax)
    out_q = ctx._mlp_forward(y_q)
    return float((out_q - ctx.out_fp).norm() / (ctx.out_fp_norm + ctx.eps))


# --------------------------------------------------------------------------- #
# A. boundary swap candidates
# --------------------------------------------------------------------------- #
def build_boundary_swap_candidates(
    score: torch.Tensor, fp_indices: torch.Tensor, int_indices: torch.Tensor,
    candidate_pool: int,
) -> List[Tuple[int, int]]:
    """Pair the weakest FP channels with the strongest INT channels (by routing
    ``score``) near the FP/INT boundary. Returns ``[(fp_out, int_in), ...]``.

    These are the swaps most likely to help under the more accurate MLP-output
    proxy: a low-scoring FP channel handed back to INT, a high-scoring INT channel
    promoted to FP. The proxy (not the score) decides whether each is accepted.
    """
    if fp_indices.numel() == 0 or int_indices.numel() == 0:
        return []
    dev = score.device
    fp_idx = fp_indices.long().to(dev)
    int_idx = int_indices.long().to(dev)
    fp_scores = score[fp_idx]
    int_scores = score[int_idx]
    n = int(min(candidate_pool, fp_idx.numel(), int_idx.numel()))
    weak_fp = fp_idx[torch.argsort(fp_scores, descending=False)][:n]      # weakest FP first
    strong_int = int_idx[torch.argsort(int_scores, descending=True)][:n]  # strongest INT first
    return [(int(weak_fp[i]), int(strong_int[i])) for i in range(n)]


# --------------------------------------------------------------------------- #
# C. greedy refinement
# --------------------------------------------------------------------------- #
def greedy_refine_fp_mask(
    ctx: LayerRefineContext, score: torch.Tensor,
    fp_indices: torch.Tensor, int_indices: torch.Tensor,
    margin: float = 0.0005, max_swaps: int = 8, candidate_pool: int = 32,
    rebase: bool = True,
) -> Dict[str, object]:
    """Refine the FP mask by boundary swaps, preserving the FP count exactly.

    A swap (fp_out -> INT, int_in -> FP) is accepted only if the proxy error
    improves by at least ``margin``. ``rebase=True`` (greedy_swap) re-evaluates the
    running mask after each accept; ``rebase=False`` (boundary_swap) compares every
    candidate against the initial mask. Stops after ``max_swaps`` accepted swaps.
    """
    C = ctx.y.shape[1]
    mask = torch.zeros(C, dtype=torch.bool, device=ctx.y.device)
    mask[fp_indices.long().to(ctx.y.device)] = True
    err_before = evaluate_mask_proxy(ctx, mask)
    base = err_before
    accepted: List[List[int]] = []
    rejected: List[List[int]] = []

    candidates = build_boundary_swap_candidates(
        score.to(ctx.y.device), fp_indices, int_indices, candidate_pool)
    for fp_out, int_in in candidates:
        if len(accepted) >= max_swaps:
            break
        # endpoint already moved by a prior accepted swap -> skip
        if (not bool(mask[fp_out])) or bool(mask[int_in]):
            rejected.append([fp_out, int_in])
            continue
        trial = mask.clone()
        trial[fp_out] = False
        trial[int_in] = True
        err = evaluate_mask_proxy(ctx, trial)
        if err < base - margin:
            mask = trial
            accepted.append([fp_out, int_in])
            if rebase:
                base = err
        else:
            rejected.append([fp_out, int_in])

    err_after = base if (rebase and accepted) else evaluate_mask_proxy(ctx, mask)
    new_fp = torch.nonzero(mask, as_tuple=False).squeeze(-1).long().cpu()
    new_int = torch.nonzero(~mask, as_tuple=False).squeeze(-1).long().cpu()
    return {
        "fp_indices": new_fp, "int_indices": new_int, "mask": mask.cpu(),
        "accepted": accepted, "rejected": rejected,
        "error_before": err_before, "error_after": err_after,
        "num_swaps": len(accepted), "refined": True,
    }


# --------------------------------------------------------------------------- #
# D. beam refinement (optional)
# --------------------------------------------------------------------------- #
def beam_refine_fp_mask(*args, **kwargs):
    raise NotImplementedError(
        "beam_swap FP mask refinement is not implemented; use "
        "fp_refine_method='greedy_swap' (or 'boundary_swap'). Beam search over masks "
        "is costly and not needed for the equal-budget refinement reported here.")


# --------------------------------------------------------------------------- #
# helper: passthrough (no refinement) result for one layer
# --------------------------------------------------------------------------- #
def _passthrough(fp_indices: torch.Tensor, int_indices: torch.Tensor) -> Dict[str, object]:
    C = int(fp_indices.numel() + int_indices.numel())
    mask = torch.zeros(C, dtype=torch.bool)
    mask[fp_indices.long()] = True
    return {"fp_indices": fp_indices.long().cpu(), "int_indices": int_indices.long().cpu(),
            "mask": mask, "accepted": [], "rejected": [],
            "error_before": None, "error_after": None, "num_swaps": 0, "refined": False}


def refine_layer_mask(ctx, score, fp_indices, int_indices, cfg) -> Dict[str, object]:
    """Dispatch one layer's refinement by ``cfg.fp_refine_method``."""
    method = cfg.fp_refine_method
    if method == "none":
        return _passthrough(fp_indices, int_indices)
    if method in ("greedy_swap", "boundary_swap"):
        return greedy_refine_fp_mask(
            ctx, score, fp_indices, int_indices,
            margin=cfg.fp_refine_margin, max_swaps=cfg.fp_refine_max_swaps_per_layer,
            candidate_pool=cfg.fp_refine_candidate_pool,
            rebase=(method == "greedy_swap"))
    if method == "beam_swap":
        return beam_refine_fp_mask()
    raise ValueError(f"unknown fp_refine_method {method!r}")


# --------------------------------------------------------------------------- #
# E. refine across layers (prioritized, budget-preserving)
# --------------------------------------------------------------------------- #
def refine_policy_masks(
    layer_inputs: Dict[int, Dict[str, object]], cfg,
    priority: Optional[Dict[int, float]] = None,
) -> Dict[int, Dict[str, object]]:
    """Apply FP mask refinement to selected layers.

    ``layer_inputs[i]`` = ``{"ctx", "score", "fp_indices", "int_indices"}``.
    Layers are refined in priority order (high first); when
    ``cfg.fp_refine_max_layers`` is set, only the top-priority layers are refined,
    the rest pass through unchanged. Each refined layer preserves its FP count.
    """
    if not cfg.use_fp_mask_refinement or cfg.fp_refine_method == "none":
        return {i: _passthrough(d["fp_indices"], d["int_indices"])
                for i, d in layer_inputs.items()}
    pr = priority or {i: 0.0 for i in layer_inputs}
    order = sorted(layer_inputs, key=lambda i: -float(pr.get(i, 0.0)))
    if cfg.fp_refine_max_layers is not None:
        selected = set(order[: int(cfg.fp_refine_max_layers)])
    else:
        selected = set(order)
    out: Dict[int, Dict[str, object]] = {}
    for i in order:
        d = layer_inputs[i]
        if i in selected:
            out[i] = refine_layer_mask(d["ctx"], d["score"], d["fp_indices"],
                                       d["int_indices"], cfg)
        else:
            out[i] = _passthrough(d["fp_indices"], d["int_indices"])
    return out
