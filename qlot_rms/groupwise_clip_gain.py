"""SADND-CAP-GT : static groupwise clip-gain tuning of the INT branch.

After SADND-CAP+ has fixed the FP/INT routing and the packing-aware INT
permutation, the remaining INT branch is tuned ONCE at calibration time:

  1. Clip multiplier  tau_g  : a per-INT-group multiplier on the frozen INT8
     activation scale.   s_g' = tau_g * s_g
  2. Output gain      eta    : a layer- or group-wise multiplier on the INT
     partial output so it matches the FP16 INT-branch output.

Folding (NO runtime work, NOT GroupRMS):
  * tau folds into the frozen per-channel activation scales (``act_scales``).
  * eta folds into the INT weight columns at patch time, so the projection is
    ``z = z_FP + eta * z_INT`` with eta absorbed by the weight.

Objective (per routed layer):  minimize || z_FP16 - z ||^2  where
``z = z_FP + eta * Q(y_I; tau*s) @ W_I^T``.  tau is chosen to minimize the
weight-normalized INT activation quantization error (the diagonal of the
projection MSE); eta is a closed-form least-squares fit of the quantized INT
output to the exact FP INT output.  Each layer is accepted only if a held proxy
error improves by a margin, else GT is disabled for that layer (clean fallback
to SADND-CAP+).  No corrections, no runtime search/RMS/sort.  No speedup claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .quant import quantize_activation_int8


# --------------------------------------------------------------------------- #
# A. INT groups (contiguous over the packing-aware INT order)
# --------------------------------------------------------------------------- #
def build_int_groups(n_int: int, group_size: int, granularity: str = "group"
                     ) -> Tuple[List[torch.Tensor], List[int], torch.Tensor]:
    """Contiguous INT-position groups in packed order.

    Returns ``(groups, group_sizes, group_of)`` where ``groups[g]`` is a LongTensor
    of INT positions (0..n_int-1, packed order), and ``group_of[p]`` is the group id
    of INT position ``p``. ``granularity="layer"`` returns a single group covering
    all INT positions.
    """
    if n_int <= 0:
        return [], [], torch.zeros(0, dtype=torch.long)
    if granularity == "layer":
        return [torch.arange(n_int)], [n_int], torch.zeros(n_int, dtype=torch.long)
    groups, group_of, start, gid = [], torch.zeros(n_int, dtype=torch.long), 0, 0
    while start < n_int:
        end = min(start + group_size, n_int)
        idx = torch.arange(start, end)
        groups.append(idx)
        group_of[start:end] = gid
        start, gid = end, gid + 1
    return groups, [int(g.numel()) for g in groups], group_of


# --------------------------------------------------------------------------- #
# Layer context (one routed layer): precompute FP references
# --------------------------------------------------------------------------- #
@dataclass
class GTLayerContext:
    y: torch.Tensor                 # [N, C] affine activation (device)
    fp_pos: torch.Tensor            # [K_F] FP channel ids (orig order)
    int_pos: torch.Tensor          # [n_int] INT channel ids (packed order)
    base_scales: torch.Tensor       # [n_int] frozen INT8 act scales (packed order)
    w_gate: torch.Tensor            # [Ig, C]
    w_up: torch.Tensor              # [Iu, C]
    w_down: torch.Tensor            # [H, I]
    b_gate: Optional[torch.Tensor] = None
    b_up: Optional[torch.Tensor] = None
    b_down: Optional[torch.Tensor] = None
    qmax: int = 127
    eps: float = 1e-8
    # precomputed
    y_I: torch.Tensor = field(default=None, repr=False)
    Wg_I: torch.Tensor = field(default=None, repr=False)
    Wu_I: torch.Tensor = field(default=None, repr=False)
    z_F_gate: torch.Tensor = field(default=None, repr=False)
    z_F_up: torch.Tensor = field(default=None, repr=False)
    target_gate: torch.Tensor = field(default=None, repr=False)   # exact FP INT output (gate)
    target_up: torch.Tensor = field(default=None, repr=False)
    gate_fp: torch.Tensor = field(default=None, repr=False)
    up_fp: torch.Tensor = field(default=None, repr=False)
    out_fp: torch.Tensor = field(default=None, repr=False)
    wcol2: torch.Tensor = field(default=None, repr=False)         # [n_int] gate+up col L2^2

    def __post_init__(self):
        y_F = self.y.index_select(-1, self.fp_pos)
        self.y_I = self.y.index_select(-1, self.int_pos)
        self.Wg_I = self.w_gate.index_select(-1, self.int_pos)
        self.Wu_I = self.w_up.index_select(-1, self.int_pos)
        Wg_F = self.w_gate.index_select(-1, self.fp_pos)
        Wu_F = self.w_up.index_select(-1, self.fp_pos)
        self.z_F_gate = y_F @ Wg_F.t()
        self.z_F_up = y_F @ Wu_F.t()
        self.target_gate = self.y_I @ self.Wg_I.t()    # = z_fp16 - z_fp_branch (gate)
        self.target_up = self.y_I @ self.Wu_I.t()
        self.gate_fp = self.z_F_gate + self.target_gate + (self.b_gate if self.b_gate is not None else 0.0)
        self.up_fp = self.z_F_up + self.target_up + (self.b_up if self.b_up is not None else 0.0)
        h_fp = F.silu(self.gate_fp) * self.up_fp
        self.out_fp = h_fp @ self.w_down.t()
        if self.b_down is not None:
            self.out_fp = self.out_fp + self.b_down
        self.wcol2 = (self.Wg_I.pow(2).sum(0) + self.Wu_I.pow(2).sum(0))   # [n_int]

    # --- forward with tuned scales + per-INT-channel gains ---
    def _branches(self, scales, gain_gate, gain_up):
        yq = quantize_activation_int8(self.y_I, scales, qmax=self.qmax)
        zi_g = (yq * gain_gate) @ self.Wg_I.t() if gain_gate is not None else yq @ self.Wg_I.t()
        zi_u = (yq * gain_up) @ self.Wu_I.t() if gain_up is not None else yq @ self.Wu_I.t()
        gate = self.z_F_gate + zi_g + (self.b_gate if self.b_gate is not None else 0.0)
        up = self.z_F_up + zi_u + (self.b_up if self.b_up is not None else 0.0)
        return gate, up

    def proxy(self, metric, scales, gain_gate=None, gain_up=None) -> float:
        gate, up = self._branches(scales, gain_gate, gain_up)
        if metric == "projection_mse":
            return float((gate - self.gate_fp).pow(2).mean() + (up - self.up_fp).pow(2).mean())
        out = (F.silu(gate) * up) @ self.w_down.t()
        if self.b_down is not None:
            out = out + self.b_down
        if metric == "mlp_output_mse":
            return float((out - self.out_fp).pow(2).mean())
        # hidden_l2
        return float((out - self.out_fp).norm() / (self.out_fp.norm() + self.eps))


def build_gt_context(y, routing, mlp, cfg, device="cpu", max_tokens=256) -> GTLayerContext:
    if y.shape[0] > max_tokens:
        y = y[:max_tokens]

    def _w(lin):
        return lin.weight.data.to(device).float()

    def _b(lin):
        return None if lin.bias is None else lin.bias.data.to(device).float()
    return GTLayerContext(
        y=y.to(device).float(),
        fp_pos=routing.fp_indices.to(device).long(),
        int_pos=routing.int_indices.to(device).long(),
        base_scales=routing.act_scales.to(device).float(),
        w_gate=_w(mlp.gate_proj), w_up=_w(mlp.up_proj), w_down=_w(mlp.down_proj),
        b_gate=_b(mlp.gate_proj), b_up=_b(mlp.up_proj), b_down=_b(mlp.down_proj),
        qmax=cfg.qmax, eps=cfg.eps_scale)


# --------------------------------------------------------------------------- #
# B. clip multiplier tuning
# --------------------------------------------------------------------------- #
def tune_clip_multiplier(ctx: GTLayerContext, groups: List[torch.Tensor],
                         candidates: List[float], granularity: str
                         ) -> Tuple[torch.Tensor, List[float]]:
    """Per-group (or per-layer) tau minimizing weight-normalized INT activation
    quantization error (diagonal of projection MSE). Returns ``(tau_per_channel,
    tau_values)`` with ``tau_per_channel`` aligned to packed INT order.
    """
    n_int = ctx.y_I.shape[1]
    s = ctx.base_scales
    cands = [float(c) for c in candidates]
    # per-channel quant MSE for each candidate tau, weighted by gate+up col L2^2
    werr = []  # [n_cand, n_int]
    for tau in cands:
        S = (s * tau).clamp_min(1e-12).unsqueeze(0)
        q = torch.clamp(torch.round(ctx.y_I / S), -ctx.qmax, ctx.qmax) * S
        mse = (q - ctx.y_I).pow(2).mean(0)            # [n_int]
        werr.append(ctx.wcol2 * mse)
    werr = torch.stack(werr, 0)                       # [n_cand, n_int]
    tau_per_channel = torch.ones(n_int, device=ctx.y_I.device)
    tau_values: List[float] = []
    for g in groups:
        g = g.to(ctx.y_I.device)
        e_g = werr.index_select(1, g).sum(1)          # [n_cand]
        best = int(torch.argmin(e_g))
        tau_per_channel[g] = cands[best]
        tau_values.append(cands[best])
    return tau_per_channel, tau_values


# --------------------------------------------------------------------------- #
# C. INT output gain fitting (closed form least squares)
# --------------------------------------------------------------------------- #
def _fit_eta_scalar(target, z_int, lo, hi, eps) -> float:
    denom = float((z_int * z_int).sum())
    if denom <= eps:
        return 1.0
    eta = float((target * z_int).sum() / denom)
    return float(min(max(eta, lo), hi))


def fit_int_output_gain(ctx: GTLayerContext, tuned_scales: torch.Tensor, which: str,
                        granularity: str, groups: List[torch.Tensor],
                        lo: float, hi: float) -> Tuple[torch.Tensor, List[float]]:
    """Fit INT output gain so ``eta * Q(y_I)@W_I^T`` matches the exact FP INT output
    ``target = y_I @ W_I^T``. Returns ``(gain_per_channel, eta_values)``.

    ``granularity="layer"`` -> one scalar eta (closed form). ``"group"`` ->
    per-group independent least-squares fit (experimental, diagonal approximation).
    """
    W_I = ctx.Wg_I if which == "gate" else ctx.Wu_I
    target = ctx.target_gate if which == "gate" else ctx.target_up
    yq = quantize_activation_int8(ctx.y_I, tuned_scales, qmax=ctx.qmax)
    n_int = yq.shape[1]
    gain = torch.ones(n_int, device=yq.device)
    if granularity == "layer":
        z_int = yq @ W_I.t()
        eta = _fit_eta_scalar(target, z_int, lo, hi, ctx.eps)
        gain[:] = eta
        return gain, [eta]
    # group: independent per-group fit (experimental)
    eta_values: List[float] = []
    for g in groups:
        g = g.to(yq.device)
        z_g = (yq.index_select(1, g)) @ W_I.index_select(-1, g).t()   # [N, out]
        eta_g = _fit_eta_scalar(target, z_g, lo, hi, ctx.eps)
        gain[g] = eta_g
        eta_values.append(eta_g)
    return gain, eta_values


# --------------------------------------------------------------------------- #
# D/E. tune one layer + accept-only decision + metadata
# --------------------------------------------------------------------------- #
def tune_layer(ctx: GTLayerContext, cfg) -> Dict[str, object]:
    """Tune tau + eta for one layer; accept only if the proxy improves by margin."""
    n_int = ctx.y_I.shape[1]
    groups_clip, _, _ = build_int_groups(n_int, cfg.gt_group_size, cfg.gt_clip_granularity)
    groups_gain, _, _ = build_int_groups(n_int, cfg.gt_group_size, cfg.gt_gain_granularity)
    n_groups = len(build_int_groups(n_int, cfg.gt_group_size, "group")[0])

    if cfg.gt_tune_clip and n_int > 0:
        tau_per_channel, tau_values = tune_clip_multiplier(
            ctx, groups_clip, cfg.gt_clip_candidates, cfg.gt_clip_granularity)
    else:
        tau_per_channel, tau_values = torch.ones(n_int, device=ctx.y_I.device), [1.0]
    tuned_scales = (ctx.base_scales * tau_per_channel).clamp_min(1e-12)

    if cfg.gt_tune_gain and n_int > 0:
        gain_gate, eta_gate = fit_int_output_gain(
            ctx, tuned_scales, "gate", cfg.gt_gain_granularity, groups_gain,
            cfg.gt_gain_clip_min, cfg.gt_gain_clip_max)
        gain_up, eta_up = fit_int_output_gain(
            ctx, tuned_scales, "up", cfg.gt_gain_granularity, groups_gain,
            cfg.gt_gain_clip_min, cfg.gt_gain_clip_max)
    else:
        gain_gate = gain_up = None
        eta_gate = eta_up = [1.0]

    before = ctx.proxy(cfg.gt_metric, ctx.base_scales, None, None)
    after = ctx.proxy(cfg.gt_metric, tuned_scales, gain_gate, gain_up)
    # Per-layer gate is a RELATIVE proxy improvement (the proxy is a small MSE, not a
    # PPL); gt_accept_margin is reused as the required relative fraction. The absolute
    # gt_accept_margin (PPL) gate is applied separately at select time.
    improved = after < before * (1.0 - cfg.gt_accept_margin)
    accept = (not cfg.gt_accept_only) or improved
    if accept and n_int > 0:
        return {"gt_enabled": True, "tuned_scales": tuned_scales.cpu(),
                "gain_gate": None if gain_gate is None else gain_gate.cpu(),
                "gain_up": None if gain_up is None else gain_up.cpu(),
                "tau_values": tau_values, "eta_gate": eta_gate, "eta_up": eta_up,
                "before": before, "after": after, "num_groups": n_groups,
                "reason": "accepted" if improved else "accepted (accept_only off)"}
    return {"gt_enabled": False, "tuned_scales": ctx.base_scales.cpu(),
            "gain_gate": None, "gain_up": None,
            "tau_values": tau_values, "eta_gate": eta_gate, "eta_up": eta_up,
            "before": before, "after": after, "num_groups": n_groups,
            "reason": "rejected: proxy did not improve by gt_accept_margin"}
