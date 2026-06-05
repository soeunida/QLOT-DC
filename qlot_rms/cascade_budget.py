"""SADND-CAP+ : cascade-aware and marginal-gain FP budget allocation.

Transformer residual streams propagate quantization error across layers, so FP
budget should consider not just local sensitivity but accumulated (cascade) error.

  e_l        = ||h_l^quant - h_l^fp16|| / (||h_l^fp16|| + eps)   (per-layer)
  cascade_l  = beta * cascade_{l-1} + e_l                         (EMA accumulation)
  budget_score_l = local_sensitivity_l + gamma*cascade_l (+ amp_weight*amp_l)

FP budget (a fixed global total) is then allocated across layers by these scores
(`allocate_fp_budget_from_scores`) or greedily by per-channel marginal gain
(`allocate_by_marginal_gain`). All static, at calibration time. No corrections,
no runtime routing, no speedup claim.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


# --------------------------------------------------------------------------- #
# Cascade error math
# --------------------------------------------------------------------------- #
def compute_layer_quant_error(h_fp: torch.Tensor, h_q: torch.Tensor,
                              eps: float = 1e-8) -> float:
    """Relative L2 error ||h_q - h_fp|| / (||h_fp|| + eps) over a token batch."""
    hf = h_fp.float().reshape(-1)
    hq = h_q.float().reshape(-1)
    return float((hq - hf).norm() / (hf.norm() + eps))


def compute_cascade_error(local_err: List[float], beta: float) -> List[float]:
    """cascade_0 = e_0; cascade_l = beta*cascade_{l-1} + e_l."""
    casc, prev = [], 0.0
    for i, e in enumerate(local_err):
        c = e if i == 0 else beta * prev + e
        casc.append(c)
        prev = c
    return casc


def compute_error_amplification(local_err: List[float], eps: float = 1e-8) -> List[float]:
    """amp_l = e_{l+1} / (e_l + eps); last layer = 1.0 (no downstream)."""
    n = len(local_err)
    amp = []
    for l in range(n):
        amp.append(local_err[l + 1] / (local_err[l] + eps) if l + 1 < n else 1.0)
    return amp


def build_cascade_budget_scores(local_sens: List[float], cascade: List[float],
                                gamma: float, amp: List[float] = None,
                                amp_weight: float = 0.0) -> torch.Tensor:
    """budget_score_l = local_sens_l + gamma*cascade_l (+ amp_weight*amp_l), normalized."""
    s = torch.tensor(local_sens, dtype=torch.float64) + gamma * torch.tensor(cascade, dtype=torch.float64)
    if amp is not None and amp_weight:
        s = s + amp_weight * torch.tensor(amp, dtype=torch.float64)
    s = s.clamp_min(0)
    tot = s.sum()
    return (s / tot) if tot > 0 else torch.full_like(s, 1.0 / len(s))


# --------------------------------------------------------------------------- #
# Budget allocation
# --------------------------------------------------------------------------- #
def allocate_fp_budget_from_scores(total_budget: int, scores: torch.Tensor,
                                   c_list: List[int]) -> List[int]:
    """Largest-remainder allocation of ``total_budget`` FP channels by ``scores``.

    Respects per-layer caps ``c_list``; preserves the TOTAL exactly (subject to
    aggregate capacity). Used by cascade-aware allocation.
    """
    L = len(c_list)
    total_budget = int(min(total_budget, sum(c_list)))
    s = scores.double()
    s = s / s.sum() if float(s.sum()) > 0 else torch.full((L,), 1.0 / L, dtype=torch.float64)
    raw = s * total_budget
    k = [min(int(torch.floor(raw[i])), c_list[i]) for i in range(L)]
    leftover = total_budget - sum(k)
    # distribute leftover by largest fractional part, among layers with capacity
    frac = [(float(raw[i] - torch.floor(raw[i])), i) for i in range(L)]
    frac.sort(reverse=True)
    idx = 0
    while leftover > 0 and any(k[i] < c_list[i] for i in range(L)):
        i = frac[idx % L][1]
        if k[i] < c_list[i]:
            k[i] += 1
            leftover -= 1
        idx += 1
    return k


def compute_marginal_gain_table(per_layer_scores: Dict[int, torch.Tensor],
                                candidates: List[float]) -> Dict[int, Dict[str, float]]:
    """Proxy marginal-gain table: gain_l(r) = SADND distortion protected by moving
    from k(r_{i-1}) to k(r_i) FP channels (sum of the next-highest channel scores).
    """
    table: Dict[int, Dict[str, float]] = {}
    cands = sorted(set(round(float(c), 4) for c in candidates))
    for i, s in per_layer_scores.items():
        C = s.numel()
        sorted_s = torch.sort(s, descending=True).values
        cum = torch.cat([torch.zeros(1), torch.cumsum(sorted_s, 0)])  # cum[k] = sum top-k
        row = {}
        prev_k = 0
        for r in cands:
            k = min(int(r * C), C)
            row[f"{r:.4f}"] = float(cum[k] - cum[prev_k])   # marginal distortion protected
            prev_k = k
        table[i] = row
    return table


def allocate_by_marginal_gain(per_layer_scores: Dict[int, torch.Tensor],
                              total_budget: int,
                              layer_weight: Dict[int, float] = None) -> Dict[int, int]:
    """Greedy global allocation: pick the ``total_budget`` highest per-channel scores
    across all layers (optionally scaled by a per-layer weight, e.g. cascade), and
    count how many fall in each layer. Equivalent to giving each FP increment to the
    layer with the current highest marginal gain.
    """
    vals, lays = [], []
    for i, s in per_layer_scores.items():
        w = (layer_weight or {}).get(i, 1.0)
        vals.append(s.float() * w)
        lays.append(torch.full((s.numel(),), i, dtype=torch.long))
    v = torch.cat(vals); lay = torch.cat(lays)
    total_budget = int(min(total_budget, v.numel()))
    k = {i: 0 for i in per_layer_scores}
    if total_budget > 0:
        top = torch.topk(v, total_budget, largest=True).indices
        chosen = lay[top]
        for i in per_layer_scores:
            k[i] = int((chosen == i).sum().item())
    return k


# --------------------------------------------------------------------------- #
# Hidden-state error capture (needs a baseline quantized plan)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def capture_layer_errors(model, baseline_plan, cfg, chunks, indices,
                         device="cpu", metric="hidden_l2", max_chunks=4) -> Dict[int, float]:
    """Per-routed-layer relative L2 error of a baseline quantization vs FP16.

    Uses ``output_hidden_states`` (residual stream). ``hidden_l2`` and
    ``residual_l2`` are the residual-stream hidden; ``mlp_output_l2`` falls back to
    the residual hidden (documented approximation).
    """
    from .model_integration import patch_model, unpatch_model
    model.eval()
    n = min(max_chunks, chunks.shape[0])
    batch = chunks[:n].to(device)
    H_fp = model(batch, use_cache=False, output_hidden_states=True).hidden_states
    handle = patch_model(model, baseline_plan, cfg)
    try:
        H_q = model(batch, use_cache=False, output_hidden_states=True).hidden_states
    finally:
        unpatch_model(handle)
    # hidden_states[l+1] is the output (residual stream) of decoder layer l
    err = {}
    for i in indices:
        hf, hq = H_fp[i + 1], H_q[i + 1]
        err[i] = compute_layer_quant_error(hf, hq, cfg.eps_scale)
    return err
