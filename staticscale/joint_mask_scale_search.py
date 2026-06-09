"""StaticScale joint mask-scale search.

Motivation (reviewer concern)
-----------------------------
In the additive StaticScale pipeline the equal-budget FP mask refinement and the
static groupwise clip-scale tuning are applied *independently*: the mask is
refined under the *pre-tuning* INT activation scales, and only afterwards are the
group clip multipliers ``tau`` tuned. Because most of the measured improvement is
scale-driven, the additive ``refine`` step contributes little on top of
``CAP+ + clip``.

This module tests whether *jointly* optimizing the FP mask and the clip scales
helps: refine the FP mask under the **post-tuning** proxy (i.e. after ``tau`` has
been tuned), because a channel that looks weak under untuned INT scales may become
safe after ``tau`` tuning, and vice-versa. Two joint orderings are provided:

* ``joint_retune_then_swap`` : tune ``tau`` first, refine the mask under the tuned
  scales, then re-tune ``tau`` on the changed INT set.
* ``joint_swap_then_retune`` : for each boundary swap candidate, re-tune the
  affected groups *before* accepting/rejecting the swap (per-swap retune).

Everything here is **calibration-time only**. The artifacts that leave this module
(FP mask, ``tau`` folded into the frozen activation scales, ``eta`` folded into the
INT weight columns) are static metadata; inference does no runtime sort / top-k /
token-dependent routing / normalization. The FP budget (``k_fp`` per layer) is
preserved exactly by every candidate.

This module reuses the existing, tested primitives:
* ``qlot_rms.groupwise_clip_gain`` : ``build_gt_context``, ``build_int_groups``,
  ``tune_clip_multiplier``, ``fit_int_output_gain``, ``tune_layer``.
* ``qlot_rms.fp_mask_refinement`` : ``build_refine_context``,
  ``greedy_refine_fp_mask``, ``evaluate_mask_proxy``.

No speedup is claimed (``torch_reference`` is correctness-only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch

from qlot_rms.groupwise_clip_gain import (
    build_gt_context, build_int_groups, tune_clip_multiplier, fit_int_output_gain,
)
from qlot_rms.fp_mask_refinement import (
    build_refine_context_from_weights, evaluate_mask_proxy, greedy_refine_fp_mask,
)
from qlot_rms.quant import compute_activation_scales
from qlot_rms import sadnd_cap


# --------------------------------------------------------------------------- #
# Family definitions
# --------------------------------------------------------------------------- #
# Baseline families are expressible as plain config overrides driving the
# existing ``calibrate()`` (no new calibration code). Joint families need the
# reordered (post-tuning) refinement implemented here.
BASELINE_FAMILIES = [
    "clean_sadnd",
    "clean_sadnd_clip_only",
    "clean_sadnd_clip",
    "capplus",
    "capplus_clip",
    "capplus_refine",
    "capplus_refine_clip",
    "full_staticscale_existing",
    "full_staticscale",
    "group_type_tau_grid",
]
JOINT_FAMILIES = [
    "joint_swap_then_retune",
    "joint_retune_then_swap",
    "cascade_joint_top_layers",
    "hard_layer_extra_search",
    "group_boundary_rebuild",
]
# Families with a dedicated static-plan builder (not plain calibrate overrides, not the
# joint proxy search): see staticscale.hard_layer_search / harmful_int_mining and the
# block-output meta-selector in the runner.
DEDICATED_BUILDER_FAMILIES = [
    "hard_layer_focused_refinement",
    "harmful_int_channel_mining",
    "block_output_proxy_selection",
]
# Group-type tau grid is a plain config-override baseline (union tau grid; each INT
# group still picks its own tau by per-group argmin at calibration).
GROUP_TYPE_TAU_UNION = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.25, 1.35, 1.50,
                        1.75, 2.00, 2.50]
ALL_FAMILIES = BASELINE_FAMILIES + JOINT_FAMILIES + DEDICATED_BUILDER_FAMILIES + ["group_type_tau_grid"]

# The additive full pipeline that the joint families must beat at the proxy level.
ADDITIVE_REFERENCE = "capplus_refine_clip"


def family_config_overrides(family: str, fp: float) -> Dict[str, object]:
    """Config overrides (on top of a CAP+ base config) for a *baseline* family.

    These drive the existing ``calibrate()``; joint families are NOT produced this
    way (raise here so callers route them through ``build_joint_plan``).
    """
    if family in JOINT_FAMILIES:
        raise ValueError(f"{family!r} is a joint family; use build_joint_plan()")
    if family in DEDICATED_BUILDER_FAMILIES:
        raise ValueError(f"{family!r} has a dedicated builder; not a calibrate override")

    capplus = dict(
        routing_score="output_aware_sadnd", fp_budget_mode="cascade",
        int_permutation_mode="packing_aware", use_cascade_aware_budget=True,
        use_marginal_gain_allocation=True, cascade_beta=0.9, cascade_gamma=1.0,
        fp_ratio=fp, global_fp_budget_ratio=fp,
    )
    refine = dict(
        use_fp_mask_refinement=True, fp_refine_method="greedy_swap",
        fp_refine_metric="hidden_l2", fp_refine_candidate_pool=32,
        fp_refine_max_swaps_per_layer=8, fp_refine_margin=0.0005,
    )
    clip = dict(
        use_groupwise_clip_gain_tuning=True, gt_tune_clip=True, gt_tune_gain=True,
        gt_clip_granularity="group", gt_gain_granularity="layer", gt_group_size=128,
        gt_clip_candidates=[0.75, 1.0, 1.25, 1.5, 2.0], gt_gain_clip_min=0.8,
        gt_gain_clip_max=1.2, gt_metric="projection_mse", gt_accept_only=True,
        gt_accept_margin=0.0005,
    )
    if family == "clean_sadnd":
        return dict(routing_score="sadnd", fp_budget_mode="fixed",
                    int_permutation_mode="original", fp_ratio=fp,
                    global_fp_budget_ratio=fp)
    if family in ("clean_sadnd_clip_only", "clean_sadnd_clip"):
        return dict(routing_score="sadnd", fp_budget_mode="fixed",
                    int_permutation_mode="original", fp_ratio=fp,
                    global_fp_budget_ratio=fp, **clip)
    if family == "capplus":
        return dict(capplus)
    if family == "capplus_clip":
        return {**capplus, **clip}
    if family == "capplus_refine":
        return {**capplus, **refine}
    if family in ("capplus_refine_clip", "full_staticscale_existing", "full_staticscale"):
        return {**capplus, **refine, **clip}
    if family == "group_type_tau_grid":
        # full StaticScale but with the union of the scale-class tau grids; each INT
        # group still selects its own tau by per-group argmin (static).
        return {**capplus, **refine, **{**clip, "gt_clip_candidates": list(GROUP_TYPE_TAU_UNION)}}
    raise ValueError(f"unknown baseline family {family!r}")


# --------------------------------------------------------------------------- #
# Candidate metadata (serializable)
# --------------------------------------------------------------------------- #
@dataclass
class JointCandidate:
    candidate_id: str
    candidate_family: str
    fp_ratio: float
    total_fp_channels: int
    per_layer_fp_budgets: Dict[int, int] = field(default_factory=dict)
    num_swaps: int = 0
    top_k_layers: object = "all"
    tau_grid: List[float] = field(default_factory=list)
    percentile: Optional[float] = None
    eta_mode: str = "none"
    d_calib_proxy_before: Optional[float] = None
    d_calib_proxy_after: Optional[float] = None
    d_sel_ppl: Optional[float] = None
    d_eval_ppl: Optional[float] = None
    delta_vs_clean_sadnd: Optional[float] = None
    delta_vs_capplus: Optional[float] = None
    delta_vs_capplus_clip: Optional[float] = None
    status: str = "pending"           # accepted | rejected | fallback | pending
    reason: str = ""

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        # JSON-friendly keys for per-layer dict (ints -> str)
        d["per_layer_fp_budgets"] = {str(k): int(v)
                                     for k, v in self.per_layer_fp_budgets.items()}
        return d


def assert_static_metadata(cand: JointCandidate) -> None:
    """Sanity-check that a candidate's tuning metadata is static (serializable,
    finite, fixed-size). Raises AssertionError on any dynamic/ill-formed field."""
    json.loads(cand.to_json())  # must round-trip
    for t in cand.tau_grid:
        assert isinstance(t, (int, float)) and t == t, "tau_grid must be static floats"
    assert isinstance(cand.eta_mode, str), "eta_mode must be a static descriptor"
    assert cand.total_fp_channels >= 0
    assert cand.status in ("accepted", "rejected", "fallback", "pending")


# --------------------------------------------------------------------------- #
# Per-layer joint proxy core (the decisive cheap signal)
# --------------------------------------------------------------------------- #
def _tuned_full_scales(y: torch.Tensor, mlp, fp_idx: torch.Tensor, int_idx: torch.Tensor,
                       base_full: torch.Tensor, tau_grid: List[float], group_size: int,
                       granularity: str, qmax: int, eps: float
                       ) -> Tuple[torch.Tensor, List[float]]:
    """Per-channel activation scales [C] with ``tau`` tuned on the INT set ``int_idx``.

    ``tau`` is tuned by the same weight-normalized INT activation quant-error
    criterion used by ``groupwise_clip_gain.tune_clip_multiplier``. FP channels keep
    their base scale (irrelevant; they are exact). Returns ``(scales[C], tau_values)``.
    """
    if int_idx.numel() == 0:
        return base_full.clone(), [1.0]
    cfg_like = SimpleNamespace(qmax=qmax, eps_scale=eps)
    view = SimpleNamespace(fp_indices=fp_idx, int_indices=int_idx,
                           act_scales=base_full.index_select(0, int_idx))
    ctx = build_gt_context(y, view, mlp, cfg_like, device=y.device, max_tokens=y.shape[0])
    groups, _, _ = build_int_groups(int_idx.numel(), group_size, granularity)
    tau_pc, tau_vals = tune_clip_multiplier(ctx, groups, tau_grid, granularity)
    tuned = base_full.clone()
    tuned[int_idx] = base_full.index_select(0, int_idx) * tau_pc
    return tuned, tau_vals


def _mask_from_fp(C: int, fp_idx: torch.Tensor, device) -> torch.Tensor:
    m = torch.zeros(C, dtype=torch.bool, device=device)
    m[fp_idx.long().to(device)] = True
    return m


def _refine(y, w_gate, w_up, w_down, b, scales_full, score, fp_idx, int_idx,
            margin, max_swaps, candidate_pool, qmax, eps):
    """Greedy boundary-swap refinement under ``scales_full`` (a [C] scale vector).
    Returns (fp_new, int_new, mask, num_swaps, err_before, err_after)."""
    ctx = build_refine_context_from_weights(
        y, scales_full, w_gate, w_up, w_down, b[0], b[1], b[2],
        qmax=qmax, eps=eps, max_tokens=y.shape[0])
    r = greedy_refine_fp_mask(ctx, score, fp_idx, int_idx, margin=margin,
                              max_swaps=max_swaps, candidate_pool=candidate_pool,
                              rebase=True)
    return (r["fp_indices"], r["int_indices"], r["mask"], r["num_swaps"],
            r["error_before"], r["error_after"])


def _proxy_under(y, w_gate, w_up, w_down, b, scales_full, mask, qmax, eps) -> float:
    ctx = build_refine_context_from_weights(
        y, scales_full, w_gate, w_up, w_down, b[0], b[1], b[2],
        qmax=qmax, eps=eps, max_tokens=y.shape[0])
    return evaluate_mask_proxy(ctx, mask.to(ctx.y.device))


@torch.no_grad()
def evaluate_layer_orderings(
    y: torch.Tensor, mlp, fp0: torch.Tensor, int0: torch.Tensor, score: torch.Tensor,
    *, tau_grid: List[float], group_size: int = 128, granularity: str = "group",
    refine_margin: float = 0.0005, max_swaps: int = 8, candidate_pool: int = 32,
    qmax: int = 127, eps: float = 1e-8, p_act: float = 0.999,
    families: Optional[List[str]] = None,
) -> Dict[str, Dict[str, object]]:
    """Compute, for one routed layer, the *unified* final proxy (MLP-output relative
    L2 under each family's final scales+mask) for the baseline and joint orderings.

    The unified metric makes orderings directly comparable: lower is better. The FP
    count is preserved by construction (boundary swaps are budget-preserving).
    Returns ``{family: {"proxy": float, "num_swaps": int, "k_fp": int,
    "tau_mean": float}}``.
    """
    families = families or [
        "capplus", "capplus_clip", "capplus_refine", "capplus_refine_clip",
        "joint_retune_then_swap", "joint_swap_then_retune",
    ]
    device = y.device
    C = y.shape[1]
    fp0 = fp0.long().to(device)
    int0 = int0.long().to(device)
    score = score.to(device)
    k_fp = int(fp0.numel())
    w_gate = mlp.gate_proj.weight.data.to(device).float()
    w_up = mlp.up_proj.weight.data.to(device).float()
    w_down = mlp.down_proj.weight.data.to(device).float()
    b = (None if mlp.gate_proj.bias is None else mlp.gate_proj.bias.data.to(device).float(),
         None if mlp.up_proj.bias is None else mlp.up_proj.bias.data.to(device).float(),
         None if mlp.down_proj.bias is None else mlp.down_proj.bias.data.to(device).float())
    base_full = compute_activation_scales(y, p_act, qmax, 1e-8, dim=0).to(device)
    mask0 = _mask_from_fp(C, fp0, device)

    def tau_mean(vals):
        return float(sum(vals) / len(vals)) if vals else 1.0

    out: Dict[str, Dict[str, object]] = {}

    # ---- capplus (no refine, no clip): base scales, base mask ----
    if "capplus" in families:
        p = _proxy_under(y, w_gate, w_up, w_down, b, base_full, mask0, qmax, eps)
        out["capplus"] = {"proxy": p, "num_swaps": 0, "k_fp": k_fp, "tau_mean": 1.0}

    # ---- capplus_clip: tau on base INT, base mask ----
    tuned0 = tau0 = None
    if any(f in families for f in
           ("capplus_clip", "joint_retune_then_swap", "joint_swap_then_retune")):
        tuned0, tau0 = _tuned_full_scales(y, mlp, fp0, int0, base_full, tau_grid,
                                          group_size, granularity, qmax, eps)
    if "capplus_clip" in families:
        p = _proxy_under(y, w_gate, w_up, w_down, b, tuned0, mask0, qmax, eps)
        out["capplus_clip"] = {"proxy": p, "num_swaps": 0, "k_fp": k_fp,
                               "tau_mean": tau_mean(tau0)}

    # ---- capplus_refine: refine under base scales, base scales for proxy ----
    if "capplus_refine" in families:
        fpR, intR, maskR, ns, _, _ = _refine(
            y, w_gate, w_up, w_down, b, base_full, score, fp0, int0,
            refine_margin, max_swaps, candidate_pool, qmax, eps)
        assert int(fpR.numel()) == k_fp, "refine must preserve k_fp"
        p = _proxy_under(y, w_gate, w_up, w_down, b, base_full, maskR, qmax, eps)
        out["capplus_refine"] = {"proxy": p, "num_swaps": ns, "k_fp": int(fpR.numel()),
                                 "tau_mean": 1.0}

    # ---- capplus_refine_clip (ADDITIVE FULL): refine under base, THEN tune tau ----
    if "capplus_refine_clip" in families:
        fpR, intR, maskR, ns, _, _ = _refine(
            y, w_gate, w_up, w_down, b, base_full, score, fp0, int0,
            refine_margin, max_swaps, candidate_pool, qmax, eps)
        fpR = fpR.to(device); intR = intR.to(device)
        tunedR, tauR = _tuned_full_scales(y, mlp, fpR, intR, base_full, tau_grid,
                                          group_size, granularity, qmax, eps)
        p = _proxy_under(y, w_gate, w_up, w_down, b, tunedR, maskR, qmax, eps)
        assert int(fpR.numel()) == k_fp, "refine must preserve k_fp"
        out["capplus_refine_clip"] = {"proxy": p, "num_swaps": ns,
                                      "k_fp": int(fpR.numel()), "tau_mean": tau_mean(tauR)}

    # ---- joint_retune_then_swap: tune tau on base INT, refine under tuned, re-tune ----
    if "joint_retune_then_swap" in families:
        fpJ, intJ, maskJ, ns, _, _ = _refine(
            y, w_gate, w_up, w_down, b, tuned0, score, fp0, int0,
            refine_margin, max_swaps, candidate_pool, qmax, eps)
        fpJ = fpJ.to(device); intJ = intJ.to(device)
        tunedJ, tauJ = _tuned_full_scales(y, mlp, fpJ, intJ, base_full, tau_grid,
                                          group_size, granularity, qmax, eps)
        p = _proxy_under(y, w_gate, w_up, w_down, b, tunedJ, maskJ, qmax, eps)
        assert int(fpJ.numel()) == k_fp, "joint refine must preserve k_fp"
        out["joint_retune_then_swap"] = {"proxy": p, "num_swaps": ns,
                                         "k_fp": int(fpJ.numel()), "tau_mean": tau_mean(tauJ)}

    # ---- joint_swap_then_retune: per-swap retune before accept/reject ----
    if "joint_swap_then_retune" in families:
        res = _swap_then_retune_layer(
            y, mlp, w_gate, w_up, w_down, b, base_full, score, fp0, int0,
            tau_grid, group_size, granularity, refine_margin, max_swaps,
            candidate_pool, qmax, eps)
        out["joint_swap_then_retune"] = res

    return out


@torch.no_grad()
def _swap_then_retune_layer(y, mlp, w_gate, w_up, w_down, b, base_full, score, fp0,
                            int0, tau_grid, group_size, granularity, refine_margin,
                            max_swaps, candidate_pool, qmax, eps) -> Dict[str, object]:
    """For each boundary swap candidate, RE-TUNE tau on the trial INT set before
    measuring the proxy; accept only if the (retuned) proxy improves. The clip
    scales and the mask are thus optimized together, per swap."""
    from qlot_rms.fp_mask_refinement import build_boundary_swap_candidates
    device = y.device
    C = y.shape[1]
    k_fp = int(fp0.numel())
    mask = _mask_from_fp(C, fp0, device)
    cur_fp, cur_int = fp0.clone(), int0.clone()
    tuned, tau_cur = _tuned_full_scales(y, mlp, cur_fp, cur_int, base_full, tau_grid,
                                        group_size, granularity, qmax, eps)
    base = _proxy_under(y, w_gate, w_up, w_down, b, tuned, mask, qmax, eps)
    cands = build_boundary_swap_candidates(score, fp0, int0, candidate_pool)
    n_acc = 0
    for fp_out, int_in in cands:
        if n_acc >= max_swaps:
            break
        if (not bool(mask[fp_out])) or bool(mask[int_in]):
            continue
        trial = mask.clone()
        trial[fp_out] = False
        trial[int_in] = True
        fp_t = torch.nonzero(trial, as_tuple=False).squeeze(-1).long()
        int_t = torch.nonzero(~trial, as_tuple=False).squeeze(-1).long()
        tuned_t, tau_t = _tuned_full_scales(y, mlp, fp_t, int_t, base_full, tau_grid,
                                            group_size, granularity, qmax, eps)
        err = _proxy_under(y, w_gate, w_up, w_down, b, tuned_t, trial, qmax, eps)
        if err < base - refine_margin:
            mask = trial
            base = err
            tau_cur = tau_t
            n_acc += 1
    fp_new = torch.nonzero(mask, as_tuple=False).squeeze(-1).long()
    assert int(fp_new.numel()) == k_fp, "swap_then_retune must preserve k_fp"
    return {"proxy": base, "num_swaps": n_acc, "k_fp": int(fp_new.numel()),
            "tau_mean": float(sum(tau_cur) / len(tau_cur)) if tau_cur else 1.0}


# --------------------------------------------------------------------------- #
# Selection rule (NEVER uses D_eval)
# --------------------------------------------------------------------------- #
def select_by_d_sel(candidates: List[JointCandidate],
                    require_d_calib_improvement: bool = True
                    ) -> Tuple[Optional[JointCandidate], List[JointCandidate]]:
    """Select the candidate with the lowest ``d_sel_ppl``.

    A candidate is eligible only if it has a ``d_sel_ppl`` and (optionally) improved
    its ``d_calib`` proxy (after < before). ``d_eval_ppl`` is *never* read here.
    Candidates that improved D_calib but worsened D_sel relative to the additive
    reference are marked ``rejected``. Returns ``(winner_or_None, all_candidates)``
    with each candidate's ``status``/``reason`` updated.
    """
    eligible = []
    for c in candidates:
        improved_calib = (c.d_calib_proxy_after is not None
                          and c.d_calib_proxy_before is not None
                          and c.d_calib_proxy_after < c.d_calib_proxy_before)
        if c.d_sel_ppl is None:
            c.status = "rejected"
            c.reason = c.reason or "no D_sel PPL"
            continue
        if require_d_calib_improvement and not improved_calib:
            c.status = "rejected"
            c.reason = "did not improve D_calib proxy"
            continue
        eligible.append(c)
    if not eligible:
        return None, candidates
    winner = min(eligible, key=lambda c: c.d_sel_ppl)
    for c in eligible:
        if c is winner:
            c.status = "accepted"
            c.reason = c.reason or "best D_sel PPL among eligible"
        elif c.status == "pending":
            c.status = "rejected"
            c.reason = "not best D_sel PPL"
    return winner, candidates


# --------------------------------------------------------------------------- #
# Real plan builder for the joint families (D_sel / D_eval path)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_joint_plan(model, tokenizer, cfg, mode: str = "joint_retune_then_swap",
                     device: str = "cpu", batch_size: int = 4,
                     allow_synthetic: bool = False, top_k_layers=None):
    """Build a patchable ``RoutingPlan`` for a joint family.

    Strategy: obtain the CAP+ base routing from the existing ``calibrate()`` (refine
    OFF, clip OFF), then per layer apply the joint reorder (refine the FP mask under
    the *tuned* scales) and the final ``tau``/``eta`` tuning. The resulting plan has
    ``act_scales`` = tau-folded INT scales, ``gt_int_gain_*`` = eta gains, and a
    refined, budget-preserving FP mask. Fully static at inference.

    ``mode`` in {"joint_retune_then_swap", "joint_swap_then_retune",
    "cascade_joint_top_layers", "hard_layer_extra_search", "group_boundary_rebuild"}.
    ``top_k_layers`` restricts the joint reorder to the top-priority layers (others
    keep the CAP+ routing); None = all routed layers.
    """
    from qlot_rms.calibration import calibrate
    from qlot_rms.config import LayerRouting, RoutingPlan
    import copy

    base_cfg = copy.deepcopy(cfg)
    base_cfg.use_fp_mask_refinement = False
    base_cfg.use_groupwise_clip_gain_tuning = False
    base_plan = calibrate(model, tokenizer, base_cfg, device=device,
                          allow_synthetic=allow_synthetic, batch_size=batch_size)

    y_by = capture_affine_activations(
        model, tokenizer, cfg, device=device, batch_size=batch_size,
        max_tokens=cfg.gt_max_tokens, allow_synthetic=allow_synthetic)

    indices = sorted(base_plan.layers.keys())
    # priority order = cascade budget score if present else k_fp
    prio = {i: (base_plan.layers[i].budget_score
                if base_plan.layers[i].budget_score is not None
                else float(base_plan.layers[i].k_fp)) for i in indices}
    order = sorted(indices, key=lambda i: -float(prio[i]))
    if top_k_layers is not None and top_k_layers != "all":
        selected = set(order[: int(top_k_layers)])
    else:
        selected = set(indices)

    new_layers: Dict[int, LayerRouting] = {}
    for i in indices:
        lr = base_plan.layers[i]
        mlp = _get_mlp(model, i)
        y = y_by[i].to(device).float()
        C = lr.num_channels
        fp0 = lr.fp_indices.to(device).long()
        int0 = lr.int_indices.to(device).long()
        k_fp = int(lr.k_fp)
        # recompute output-aware routing score for boundary candidates
        score = _output_aware_score(lr.delta_tilde.to(device), mlp, device)
        base_full = compute_activation_scales(y, cfg.p_act, cfg.qmax, cfg.eps_scale,
                                              dim=0).to(device)
        chan_scale = base_full  # affine-activation scale proxy for packing order

        if i not in selected or int0.numel() == 0:
            new_layers[i] = lr  # keep CAP+ routing untouched
            continue

        fp_new, int_new = _joint_reorder_layer(
            y, mlp, fp0, int0, score, base_full, cfg, mode)
        assert int(fp_new.numel()) == k_fp, "joint plan must preserve k_fp"
        new_layers[i] = _assemble_gt_layer(i, lr, mlp, y, fp_new, int_new,
                                           base_full, chan_scale, cfg, device)

    return RoutingPlan(config=cfg, layers=new_layers)


def _assemble_gt_layer(i, lr, mlp, y, fp_new, int_new, base_full, chan_scale, cfg,
                       device, tau_candidates=None):
    """Rebuild the packing-aware INT order for a (possibly modified) FP/INT split,
    run the final static tau/eta tuning (``tune_layer``), and return a ``LayerRouting``.

    ``tau_candidates`` optionally overrides ``cfg.gt_clip_candidates`` for the tau grid
    (e.g. group-type union grid). Everything produced is static metadata.
    """
    import copy
    from qlot_rms.config import LayerRouting
    from qlot_rms import groupwise_clip_gain as gcg
    C = lr.num_channels
    k_fp = int(fp_new.numel())
    int_packed = sadnd_cap.build_packing_aware_int_permutation(
        int_new.cpu(), chan_scale.index_select(0, int_new.to(chan_scale.device)).cpu(),
        cfg.int_permutation_mode, cfg.w8_group_size)
    perm = torch.cat([fp_new.cpu(), int_packed], dim=0)
    mask = torch.zeros(C, dtype=torch.bool)
    mask[fp_new.cpu()] = True
    gt_cfg = cfg
    if tau_candidates is not None:
        gt_cfg = copy.deepcopy(cfg)
        gt_cfg.gt_clip_candidates = list(tau_candidates)
    view = SimpleNamespace(
        fp_indices=fp_new.cpu(), int_indices=int_packed,
        act_scales=base_full.index_select(0, int_packed.to(base_full.device)).cpu())
    gctx = gcg.build_gt_context(y.cpu(), view, mlp, gt_cfg, device=device,
                                max_tokens=cfg.gt_max_tokens)
    res = gcg.tune_layer(gctx, gt_cfg)
    act_scales = (res["tuned_scales"] if res["gt_enabled"]
                  else base_full.index_select(0, int_packed.to(base_full.device)).cpu())
    return LayerRouting(
        layer_index=i, num_channels=C, k_fp=k_fp,
        fp_indices=fp_new.cpu(), int_indices=int_packed, perm=perm, mask=mask,
        delta_tilde=lr.delta_tilde, act_scales=act_scales.cpu(),
        gt_enabled=bool(res["gt_enabled"]),
        gt_clip_granularity=cfg.gt_clip_granularity,
        gt_gain_granularity=cfg.gt_gain_granularity,
        gt_num_groups=int(res.get("num_groups", 0)),
        gt_tau_values=list(res.get("tau_values", []) or []),
        gt_eta_gate=list(res.get("eta_gate", []) or []),
        gt_eta_up=list(res.get("eta_up", []) or []),
        gt_int_gain_gate=(res.get("gain_gate") if res["gt_enabled"] else None),
        gt_int_gain_up=(res.get("gain_up") if res["gt_enabled"] else None),
        gt_proxy_error_before=res.get("before"),
        gt_proxy_error_after=res.get("after"), gt_reason=res.get("reason"),
        w8_group_size=cfg.w8_group_size, routing_score=lr.routing_score,
        int_permutation_mode=cfg.int_permutation_mode,
        selected_fp_ratio=k_fp / max(1, C), budget_policy=lr.budget_policy,
        cascade_local_error=lr.cascade_local_error,
        cascade_error=lr.cascade_error, budget_score=lr.budget_score,
        refined=True, norm_type=lr.norm_type)


def _select_fp_int_by_score(score: torch.Tensor, k_fp: int):
    """Top-``k_fp`` channels by ``score`` are FP (original order preserved). Static."""
    C = int(score.numel())
    k_fp = max(0, min(int(k_fp), C))
    order = torch.argsort(score, descending=True)
    fp = order[:k_fp].sort().values.long()
    intc = order[k_fp:].sort().values.long()
    return fp, intc


@torch.no_grad()
def build_modified_plan(model, tokenizer, cfg, device="cpu", batch_size=4,
                        *, budget_override=None, budget_fn=None, fp_int_producer=None,
                        tau_candidates=None, refine_after=True,
                        allow_synthetic=False):
    """Generalized static-plan builder used by the hard-layer and harmful-INT families.

    * ``budget_override`` : ``{layer: new_k_fp}`` (caller guarantees the total is
      preserved). FP/INT are re-selected from the routing score at the new budget.
    * ``fp_int_producer`` : optional ``fn(y, mlp, fp0, int0, score, base_full, cfg)
      -> (fp_new, int_new)`` that proposes a budget-preserving FP/INT split (e.g.
      harmful-INT mining). Defaults to ``joint_retune_then_swap`` refinement when
      ``refine_after`` else the (possibly re-budgeted) CAP+ split.
    * ``tau_candidates`` : optional tau grid override (group-type union grid).

    Returns a patchable ``RoutingPlan``. Fully static at inference.
    """
    from qlot_rms.calibration import calibrate
    from qlot_rms.config import RoutingPlan
    import copy

    base_cfg = copy.deepcopy(cfg)
    base_cfg.use_fp_mask_refinement = False
    base_cfg.use_groupwise_clip_gain_tuning = False
    base_plan = calibrate(model, tokenizer, base_cfg, device=device,
                          allow_synthetic=allow_synthetic, batch_size=batch_size)
    y_by = capture_affine_activations(
        model, tokenizer, cfg, device=device, batch_size=batch_size,
        max_tokens=cfg.gt_max_tokens, allow_synthetic=allow_synthetic)

    indices = sorted(base_plan.layers.keys())
    if budget_fn is not None and budget_override is None:
        budget_override = budget_fn(base_plan, y_by, model, cfg, device)
    new_layers = {}
    for i in indices:
        lr = base_plan.layers[i]
        mlp = _get_mlp(model, i)
        y = y_by[i].to(device).float()
        C = lr.num_channels
        score = _output_aware_score(lr.delta_tilde.to(device), mlp, device)
        base_full = compute_activation_scales(y, cfg.p_act, cfg.qmax, cfg.eps_scale,
                                              dim=0).to(device)
        # re-budget if requested (re-select FP/INT at the new k_fp)
        if budget_override is not None and i in budget_override:
            k_fp = int(budget_override[i])
            fp0, int0 = _select_fp_int_by_score(score, k_fp)
            fp0, int0 = fp0.to(device), int0.to(device)
        else:
            fp0 = lr.fp_indices.to(device).long()
            int0 = lr.int_indices.to(device).long()
            k_fp = int(fp0.numel())
        if int0.numel() == 0:
            new_layers[i] = lr
            continue
        if fp_int_producer is not None:
            fp_new, int_new = fp_int_producer(y, mlp, fp0, int0, score, base_full, cfg)
        elif refine_after:
            fp_new, int_new = _joint_reorder_layer(y, mlp, fp0, int0, score,
                                                   base_full, cfg, "joint_retune_then_swap")
        else:
            fp_new, int_new = fp0, int0
        assert int(fp_new.numel()) == k_fp, "modified plan must preserve k_fp"
        new_layers[i] = _assemble_gt_layer(i, lr, mlp, y, fp_new, int_new, base_full,
                                           base_full, cfg, device,
                                           tau_candidates=tau_candidates)
    return RoutingPlan(config=cfg, layers=new_layers)


def _joint_reorder_layer(y, mlp, fp0, int0, score, base_full, cfg, mode):
    """Return (fp_new, int_new) for one layer under the chosen joint ``mode``."""
    device = y.device
    w_gate = mlp.gate_proj.weight.data.to(device).float()
    w_up = mlp.up_proj.weight.data.to(device).float()
    w_down = mlp.down_proj.weight.data.to(device).float()
    b = (None if mlp.gate_proj.bias is None else mlp.gate_proj.bias.data.to(device).float(),
         None if mlp.up_proj.bias is None else mlp.up_proj.bias.data.to(device).float(),
         None if mlp.down_proj.bias is None else mlp.down_proj.bias.data.to(device).float())
    tau_grid = list(cfg.gt_clip_candidates)
    gs, gran = cfg.gt_group_size, cfg.gt_clip_granularity
    mar, msw, pool = cfg.fp_refine_margin, cfg.fp_refine_max_swaps_per_layer, cfg.fp_refine_candidate_pool
    qmax, eps = cfg.qmax, cfg.eps_scale

    if mode in ("joint_swap_then_retune",):
        r = _swap_then_retune_layer(y, mlp, w_gate, w_up, w_down, b, base_full, score,
                                    fp0, int0, tau_grid, gs, gran, mar, msw, pool, qmax, eps)
        # reconstruct fp from mask: recompute via greedy (swap fn returns counts only)
        # Re-run to obtain indices deterministically:
        return _swap_then_retune_indices(y, mlp, w_gate, w_up, w_down, b, base_full,
                                         score, fp0, int0, tau_grid, gs, gran, mar,
                                         msw, pool, qmax, eps)
    # default: retune_then_swap / group_boundary_rebuild / cascade_joint / hard_layer
    tuned0, _ = _tuned_full_scales(y, mlp, fp0, int0, base_full, tau_grid, gs, gran, qmax, eps)
    fpJ, intJ, _, _, _, _ = _refine(y, w_gate, w_up, w_down, b, tuned0, score, fp0,
                                    int0, mar, msw, pool, qmax, eps)
    return fpJ.to(device), intJ.to(device)


def _swap_then_retune_indices(y, mlp, w_gate, w_up, w_down, b, base_full, score, fp0,
                              int0, tau_grid, gs, gran, mar, msw, pool, qmax, eps):
    from qlot_rms.fp_mask_refinement import build_boundary_swap_candidates
    device = y.device
    C = y.shape[1]
    mask = _mask_from_fp(C, fp0, device)
    tuned, _ = _tuned_full_scales(y, mlp, fp0, int0, base_full, tau_grid, gs, gran, qmax, eps)
    base = _proxy_under(y, w_gate, w_up, w_down, b, tuned, mask, qmax, eps)
    cands = build_boundary_swap_candidates(score, fp0, int0, pool)
    n_acc = 0
    for fp_out, int_in in cands:
        if n_acc >= msw:
            break
        if (not bool(mask[fp_out])) or bool(mask[int_in]):
            continue
        trial = mask.clone()
        trial[fp_out] = False
        trial[int_in] = True
        fp_t = torch.nonzero(trial, as_tuple=False).squeeze(-1).long()
        int_t = torch.nonzero(~trial, as_tuple=False).squeeze(-1).long()
        tuned_t, _ = _tuned_full_scales(y, mlp, fp_t, int_t, base_full, tau_grid, gs, gran, qmax, eps)
        err = _proxy_under(y, w_gate, w_up, w_down, b, tuned_t, trial, qmax, eps)
        if err < base - mar:
            mask, base, n_acc = trial, err, n_acc + 1
    fp_new = torch.nonzero(mask, as_tuple=False).squeeze(-1).long()
    int_new = torch.nonzero(~mask, as_tuple=False).squeeze(-1).long()
    return fp_new.to(device), int_new.to(device)


# --------------------------------------------------------------------------- #
# Activation capture (reuses the tested PreAffineCapture machinery)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def capture_affine_activations(model, tokenizer, cfg, device="cpu", batch_size=4,
                               max_tokens=256, allow_synthetic=False
                               ) -> Dict[int, torch.Tensor]:
    """Capture per-layer affine activations ``y = u*gamma + beta`` (capped to
    ``max_tokens`` tokens per layer) for the routed layers. Returns ``{i: y[N, C]}``
    on CPU."""
    from qlot_rms.capture import PreAffineCapture
    from qlot_rms.data import build_calibration_chunks, iter_batches
    from qlot_rms.model_integration import resolve_routed_layer_indices, get_ln2_modules

    indices = resolve_routed_layer_indices(model, cfg)
    ln2 = get_ln2_modules(model, indices)
    affine = {i: (ln2[i].weight, getattr(ln2[i], "bias", None)) for i in indices}
    chunks = build_calibration_chunks(
        tokenizer, seq_len=cfg.calibration_seq_len, num_chunks=cfg.calibration_samples,
        seed=cfg.seed, allow_synthetic=allow_synthetic)
    cap = PreAffineCapture(ln2, store_device="cpu").attach()
    store: Dict[int, List[torch.Tensor]] = {i: [] for i in indices}
    tok_cnt: Dict[int, int] = {i: 0 for i in indices}
    try:
        for batch in iter_batches(chunks, torch.arange(chunks.shape[0]), batch_size=batch_size):
            if all(tok_cnt[i] >= max_tokens for i in indices):
                break
            cap.reset()
            model.eval()
            model(batch.to(device), use_cache=False)
            u_by = cap.collect()
            for i in indices:
                if i not in u_by or tok_cnt[i] >= max_tokens:
                    continue
                gamma, beta = affine[i]
                y = u_by[i] * gamma.detach().float().cpu()
                if beta is not None:
                    y = y + beta.detach().float().cpu()
                take = max_tokens - tok_cnt[i]
                if take < y.shape[0]:
                    y = y[:take]
                store[i].append(y)
                tok_cnt[i] += y.shape[0]
    finally:
        cap.detach()
    return {i: (torch.cat(store[i], 0) if store[i]
                else torch.zeros(1, ln2[i].weight.numel())) for i in indices}


def _get_mlp(model, i):
    from qlot_rms.model_integration import find_decoder_layers
    return find_decoder_layers(model)[i].mlp


def _output_aware_score(delta_tilde, mlp, device):
    wg = mlp.gate_proj.weight.data.float().norm(dim=0).cpu()
    wu = mlp.up_proj.weight.data.float().norm(dim=0).cpu()
    score = sadnd_cap.compute_output_aware_sadnd_score(delta_tilde.cpu(), wg, wu)
    return score.to(device)
