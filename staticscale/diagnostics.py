"""StaticScale diagnostics.

Quantifies *why* the full StaticScale pipeline adds little over CAP+ + clip, and
whether tighter FP budgets change the picture. Produces five CSVs:

* ``budget_saturation.csv`` : how saturated FP protection of high-risk channels is.
* ``clip_dominance.csv``     : per-layer proxy decomposition + tau/eta stats + the
  fraction of the total proxy improvement explained by clip tuning.
* ``mask_overlap.csv``       : Jaccard overlap of FP masks across stages.
* ``hard_layers.csv``        : residual error per layer after CAP+ + clip; whether the
  full pipeline actually changed those layers.
* ``proxy_mismatch.csv``     : proxy-vs-proxy deltas (D_sel/D_eval columns populated
  only by the joint-search runner, never used for selection here).

The pure helpers are unit-testable on toy tensors; the orchestrator reuses the same
calibration-time primitives as the joint search. Static at inference. No download.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List

import torch

from staticscale import joint_mask_scale_search as J


# --------------------------------------------------------------------------- #
# Pure helpers (toy-testable)
# --------------------------------------------------------------------------- #
def jaccard(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    """Jaccard overlap of two boolean FP masks."""
    a = mask_a.bool()
    b = mask_b.bool()
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union > 0 else 1.0


def budget_entropy(per_layer_k: Dict[int, int]) -> float:
    """Normalized Shannon entropy (0..1) of the FP-budget distribution over layers.
    1.0 = perfectly uniform; lower = budget concentrated in few layers."""
    vals = torch.tensor([float(v) for v in per_layer_k.values()])
    tot = float(vals.sum())
    if tot <= 0 or vals.numel() <= 1:
        return 0.0
    p = vals / tot
    p = p[p > 0]
    h = float(-(p * p.log()).sum())
    return h / float(torch.log(torch.tensor(float(vals.numel()))))


def clip_explained_fraction(p_capplus: float, p_capplus_clip: float,
                            p_full: float) -> float:
    """Fraction of the total proxy improvement (CAP+ -> full) that the clip step
    (CAP+ -> CAP+ + clip) already explains. ~1.0 means clip is the dominant driver."""
    total = p_capplus - p_full
    if abs(total) < 1e-12:
        return 1.0
    return (p_capplus - p_capplus_clip) / total


def top_sensitive_protected_fraction(score: torch.Tensor, fp_mask: torch.Tensor,
                                     frac: float = 0.05) -> float:
    """Of the top-``frac`` channels by routing score, what fraction are FP-protected.
    ~1.0 means FP protection of the highest-risk channels is saturated."""
    C = int(score.numel())
    k = max(1, int(round(frac * C)))
    top = torch.argsort(score, descending=True)[:k]
    return float(fp_mask.bool()[top].float().mean())


def boundary_score_gap(score: torch.Tensor, fp_idx: torch.Tensor,
                       int_idx: torch.Tensor) -> float:
    """Gap between the weakest FP score and the strongest INT score (normalized by the
    score scale). Small gap => boundary channels are nearly tied (refinement marginal)."""
    if fp_idx.numel() == 0 or int_idx.numel() == 0:
        return 0.0
    weak_fp = float(score[fp_idx.long()].min())
    strong_int = float(score[int_idx.long()].max())
    denom = float(score.abs().mean()) + 1e-12
    return (weak_fp - strong_int) / denom


# ---- group-type tau grids (static metadata) ----
def group_type_tau_grids() -> Dict[str, List[float]]:
    return {
        "low": [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20],
        "mid": [0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.35],
        "high": [1.00, 1.10, 1.20, 1.35, 1.50, 1.75, 2.00],
        "outlier": [1.10, 1.25, 1.50, 1.75, 2.00, 2.50],
    }


def union_tau_grid() -> List[float]:
    """Sorted union of all group-type tau grids. Used as a superset grid so each INT
    group independently picks its best tau (static; per-group argmin at calibration)."""
    s = set()
    for g in group_type_tau_grids().values():
        s.update(g)
    return sorted(s)


def classify_int_groups_by_scale(group_mean_scales: torch.Tensor) -> List[str]:
    """Classify each INT group as low/mid/high/outlier by quantiles of its mean
    activation scale. Pure function over the group scales."""
    s = group_mean_scales.float()
    if s.numel() == 0:
        return []
    if s.numel() < 4:
        return ["mid"] * int(s.numel())
    q = torch.quantile(s, torch.tensor([0.33, 0.66, 0.95]))
    out = []
    for v in s.tolist():
        if v <= float(q[0]):
            out.append("low")
        elif v <= float(q[1]):
            out.append("mid")
        elif v <= float(q[2]):
            out.append("high")
        else:
            out.append("outlier")
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_diagnostics(model, tokenizer, cfg_dict, fp_ratios: List[float], device: str,
                    out_root: str, batch_size: int = 4, n_layers: int = 12,
                    allow_synthetic: bool = False) -> Dict[str, str]:
    """Compute the five diagnostic CSVs over the given FP ratios. Layers are sampled by
    CAP+ budget priority (the sample size is logged; no silent truncation)."""
    from qlot_rms.calibration import calibrate
    from staticscale import StaticScaleConfig
    os.makedirs(out_root, exist_ok=True)

    def _cfg(fp):
        base = {k: v for k, v in cfg_dict.items() if k in StaticScaleConfig.__dataclass_fields__}
        base.update(J.family_config_overrides("capplus", fp))
        # master flags required by the patcher used inside cascade-budget calibration
        base["enable_qlot_rms"] = True
        base["method"] = "sadnd_cap"
        base["qlot_scope"] = "mlp_only"
        base["calibration_samples"] = cfg_dict.get("calib_chunks", base.get("calibration_samples", 128))
        base["calibration_seq_len"] = cfg_dict.get("calib_seq_len", base.get("calibration_seq_len", 512))
        return StaticScaleConfig.from_dict(base)

    rows_bs, rows_cd, rows_mo, rows_hl, rows_pm = [], [], [], [], []
    summary = [f"# StaticScale diagnostics  fp_ratios={fp_ratios}  layer_sample={n_layers}"]

    for fp in fp_ratios:
        scfg = _cfg(fp)
        plan = calibrate(model, tokenizer, scfg, device=device,
                         allow_synthetic=allow_synthetic, batch_size=batch_size)
        y_by = J.capture_affine_activations(model, tokenizer, scfg, device=device,
                                            batch_size=batch_size,
                                            max_tokens=scfg.gt_max_tokens,
                                            allow_synthetic=allow_synthetic)
        indices = sorted(plan.layers.keys())
        per_layer_k = {i: int(plan.layers[i].k_fp) for i in indices}
        prio = {i: (plan.layers[i].budget_score if plan.layers[i].budget_score is not None
                    else float(plan.layers[i].k_fp)) for i in indices}
        sample = sorted(indices, key=lambda i: -float(prio[i]))[:n_layers]

        # budget_saturation (per fp, aggregated over sample)
        ent = budget_entropy(per_layer_k)
        n_changed = sum(1 for i in indices
                        if abs(per_layer_k[i] - round(fp * plan.layers[i].num_channels)) > 0)
        prot, gap = [], []
        tau_means = []
        for i in sample:
            lr = plan.layers[i]
            if lr.int_indices.numel() == 0:
                continue
            mlp = J._get_mlp(model, i)
            score = J._output_aware_score(lr.delta_tilde.to(device), mlp, device)
            fp_mask = torch.zeros(lr.num_channels, dtype=torch.bool, device=device)
            fp_mask[lr.fp_indices.to(device)] = True
            prot.append(top_sensitive_protected_fraction(score, fp_mask, 0.05))
            gap.append(boundary_score_gap(score, lr.fp_indices.to(device),
                                          lr.int_indices.to(device)))
            res = J.evaluate_layer_orderings(
                y_by[i].to(device).float(), mlp, lr.fp_indices, lr.int_indices, score,
                tau_grid=list(scfg.gt_clip_candidates), group_size=scfg.gt_group_size,
                granularity=scfg.gt_clip_granularity, refine_margin=scfg.fp_refine_margin,
                max_swaps=scfg.fp_refine_max_swaps_per_layer,
                candidate_pool=scfg.fp_refine_candidate_pool, qmax=scfg.qmax,
                eps=scfg.eps_scale, p_act=scfg.p_act,
                families=["capplus", "capplus_clip", "capplus_refine", "capplus_refine_clip"])
            pc, pcc = res["capplus"]["proxy"], res["capplus_clip"]["proxy"]
            pr, pf = res["capplus_refine"]["proxy"], res["capplus_refine_clip"]["proxy"]
            tau_means.append(res["capplus_clip"]["tau_mean"])
            cef = clip_explained_fraction(pc, pcc, pf)
            rows_cd.append([fp, i, f"{pc:.8f}", f"{pcc:.8f}", f"{pr:.8f}", f"{pf:.8f}",
                            f"{res['capplus_clip']['tau_mean']:.4f}",
                            f"{res['capplus_refine_clip']['num_swaps']}", f"{cef:.4f}"])
            rows_hl.append([fp, i, f"{pcc:.8f}",
                            f"{res['capplus_refine_clip']['num_swaps']}",
                            res['capplus_refine_clip']['num_swaps'] > 0])
            # mask overlap: clean_sadnd (fixed top-k, same k_fp) vs capplus vs refined
            fp_clean, int_clean = J._select_fp_int_by_score(score, lr.k_fp)
            m_clean = torch.zeros(lr.num_channels, dtype=torch.bool, device=device)
            m_clean[fp_clean.to(device)] = True
            rows_mo.append([fp, i, f"{jaccard(m_clean, fp_mask):.4f}",
                            f"{res['capplus_refine_clip']['num_swaps']}"])
            rows_pm.append([fp, i, f"{(pc - pcc):.8f}", f"{(pcc - pf):.8f}", "", ""])
        rows_bs.append([fp,
                        f"{(sum(prot)/len(prot) if prot else 0):.4f}",
                        f"{ent:.4f}",
                        f"{(sum(gap)/len(gap) if gap else 0):.4f}",
                        n_changed,
                        f"{(sum(tau_means)/len(tau_means) if tau_means else 1.0):.4f}"])
        summary.append(f"fp={fp}: top5%-sensitive FP-protected="
                       f"{(sum(prot)/len(prot) if prot else 0):.3f}, budget_entropy={ent:.3f}, "
                       f"mean clip_explained_fraction="
                       f"{(sum(float(r[8]) for r in rows_cd if r[0]==fp)/max(1,sum(1 for r in rows_cd if r[0]==fp))):.3f}")

    def _w(name, header, rows):
        p = os.path.join(out_root, name)
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        return p

    paths = {
        "budget_saturation": _w("budget_saturation.csv",
            ["fp_ratio", "top5pct_sensitive_fp_protected", "budget_entropy",
             "mean_boundary_score_gap", "n_layers_changed_budget", "mean_tau"], rows_bs),
        "clip_dominance": _w("clip_dominance.csv",
            ["fp_ratio", "layer", "proxy_capplus", "proxy_capplus_clip",
             "proxy_capplus_refine", "proxy_full", "tau_mean", "full_swaps",
             "clip_explained_fraction"], rows_cd),
        "mask_overlap": _w("mask_overlap.csv",
            ["fp_ratio", "layer", "jaccard_cleansadnd_capplus", "full_swaps"], rows_mo),
        "hard_layers": _w("hard_layers.csv",
            ["fp_ratio", "layer", "residual_proxy_capplus_clip", "full_swaps",
             "full_changed_layer"], rows_hl),
        "proxy_mismatch": _w("proxy_mismatch.csv",
            ["fp_ratio", "layer", "d_calib_clip_delta", "d_calib_refine_delta",
             "d_sel_delta", "d_eval_delta"], rows_pm),
    }
    with open(os.path.join(out_root, "summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    paths["summary"] = os.path.join(out_root, "summary.txt")
    return paths
