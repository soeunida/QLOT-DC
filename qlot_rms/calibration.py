"""SADND-CAP calibration.

Produces a frozen RoutingPlan per routed layer:
  1. SADND or output-aware SADND score (proxy distortion, optionally weighted by
     gate/up weight column norms)
  2. FP/INT selection -- fixed per-layer fp_ratio OR a global FP budget reallocated
     across layers by sensitivity
  3. packing-aware INT permutation (order INT channels by activation scale so each
     W8-G128 group is more uniform)
  4. frozen per-INT-channel activation scales

No corrections (no GroupRMS / diagonal / bias / low-rank / block). No runtime
top-k/sort. No speedup is claimed.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .capture import PreAffineCapture
from .config import QLotRmsConfig, LayerRouting, RoutingPlan
from .data import build_calibration_chunks, make_subsets, iter_batches
from .quant import compute_activation_scales
from .sadnd import proxy_distortion_subset, aggregate_distortion
from . import sadnd_cap
from .model_integration import resolve_routed_layer_indices, get_ln2_modules, find_decoder_layers


@torch.no_grad()
def _run_forward(model, chunk_ids, indices, device, batch_size):
    model.eval()
    for batch in iter_batches(chunk_ids, indices, batch_size=batch_size):
        model(batch.to(device), use_cache=False)


@torch.no_grad()
def _forward_batch(model, batch, device):
    model.eval()
    model(batch.to(device), use_cache=False)


def _affine(norm):
    return norm.weight, getattr(norm, "bias", None)


@torch.no_grad()
def calibrate(model, tokenizer, cfg: QLotRmsConfig, device: str = "cpu",
              routing_method: Optional[str] = None, allow_synthetic: bool = False,
              batch_size: int = 4, verbose: bool = False) -> RoutingPlan:
    cfg.validate()
    rs = routing_method or cfg.routing_score
    if rs not in ("sadnd", "output_aware_sadnd", "magnitude"):
        raise ValueError(f"unknown routing_score {rs!r}")

    def _log(m):
        if verbose:
            print(f"[sadnd-cap:calib] {m}")

    indices = resolve_routed_layer_indices(model, cfg)
    ln2 = get_ln2_modules(model, indices)
    layer_mods = find_decoder_layers(model)
    affine = {i: _affine(ln2[i]) for i in indices}

    chunks = build_calibration_chunks(
        tokenizer, seq_len=cfg.calibration_seq_len, num_chunks=cfg.calibration_samples,
        seed=cfg.seed, allow_synthetic=allow_synthetic)
    num_chunks = chunks.shape[0]
    subsets = make_subsets(num_chunks, cfg.num_calib_subsets, cfg.subset_size, cfg.seed)
    _log(f"routed={indices} chunks={num_chunks} subsets={len(subsets)} score={rs}")

    cap = PreAffineCapture(ln2, store_device="cpu").attach()

    # ---- Pass 1: per-subset proxy distortion + per-channel abs-mean ----
    per_subset_delta: Dict[int, List[torch.Tensor]] = {i: [] for i in indices}
    absmean_sum: Dict[int, torch.Tensor] = {}
    absmean_cnt: Dict[int, int] = {i: 0 for i in indices}
    for s_i, subset in enumerate(subsets):
        cap.reset()
        _run_forward(model, chunks, subset, device, batch_size)
        u_by = cap.collect()
        for i in indices:
            u = u_by[i]
            per_subset_delta[i].append(proxy_distortion_subset(u, cfg.p_proxy, cfg.qmax, cfg.eps))
            a = u.abs().sum(dim=0)
            absmean_sum[i] = a if i not in absmean_sum else absmean_sum[i] + a
            absmean_cnt[i] += u.shape[0]
        _log(f"pass1 subset {s_i+1}/{len(subsets)}")

    # ---- scores + per-channel packing scale ----
    delta_t: Dict[int, torch.Tensor] = {}
    scores: Dict[int, torch.Tensor] = {}
    chan_scale: Dict[int, torch.Tensor] = {}   # packing-aware sort key (length C)
    for i in indices:
        dt = aggregate_distortion(torch.stack(per_subset_delta[i], 0))
        delta_t[i] = dt
        absmean = absmean_sum[i] / max(1, absmean_cnt[i])
        gamma, _ = affine[i]
        gabs = gamma.detach().float().cpu().abs()
        chan_scale[i] = absmean * gabs                # ~ scale of affine activation
        if rs == "magnitude":
            scores[i] = absmean
        elif rs == "output_aware_sadnd":
            mlp = layer_mods[i].mlp
            wg = mlp.gate_proj.weight.data.float().norm(dim=0).cpu()
            wu = mlp.up_proj.weight.data.float().norm(dim=0).cpu()
            scores[i] = sadnd_cap.compute_output_aware_sadnd_score(dt, wg, wu)
        else:
            scores[i] = dt

    # ---- FP budget allocation ----
    budget_policy = "fixed"
    cascade_diag: Dict[int, dict] = {}
    use_casc = cfg.use_cascade_aware_budget
    use_marg = cfg.use_marginal_gain_allocation
    if use_casc or use_marg:
        from . import cascade_budget as cb
        ratio = cfg.global_fp_budget_ratio if cfg.global_fp_budget_ratio is not None else cfg.fp_ratio
        total_budget = int(ratio * sum(int(scores[i].numel()) for i in indices))
        per_layer_scores = {i: scores[i] for i in indices}
        if use_casc:
            # baseline fixed-fp plan to measure residual-stream quant error
            base_cfg = QLotRmsConfig.from_dict({**cfg.to_dict(),
                "use_cascade_aware_budget": False, "use_marginal_gain_allocation": False,
                "use_fp_mask_refinement": False, "use_groupwise_clip_gain_tuning": False,
                "fp_budget_mode": "fixed", "int_permutation_mode": "original",
                "fp_ratio": float(ratio)})
            base_plan = calibrate(model, tokenizer, base_cfg, device=device,
                                  routing_method=rs, allow_synthetic=allow_synthetic,
                                  batch_size=batch_size, verbose=False)
            err = cb.capture_layer_errors(model, base_plan, cfg, chunks, indices,
                                          device=device, metric=cfg.cascade_metric)
            e_list = [err[i] for i in indices]
            casc = cb.compute_cascade_error(e_list, cfg.cascade_beta)
            amp = cb.compute_error_amplification(e_list) if cfg.use_error_amplification else None
            local_sens = [float(scores[i].sum()) for i in indices]
            bscore = cb.build_cascade_budget_scores(local_sens, casc, cfg.cascade_gamma,
                                                    amp, cfg.amplification_weight)
            for j, i in enumerate(indices):
                cascade_diag[i] = {"e": e_list[j], "cascade": casc[j], "score": float(bscore[j])}
            if use_marg:
                lw = {i: float(bscore[j]) for j, i in enumerate(indices)}
                k_by = cb.allocate_by_marginal_gain(per_layer_scores, total_budget, lw)
                budget_policy = "cascade_marginal"
            else:
                k_list = cb.allocate_fp_budget_from_scores(
                    total_budget, bscore, [int(scores[i].numel()) for i in indices])
                k_by = {indices[j]: k_list[j] for j in range(len(indices))}
                budget_policy = "cascade"
        else:  # marginal only
            k_by = cb.allocate_by_marginal_gain(per_layer_scores, total_budget)
            budget_policy = "marginal"
    elif cfg.fp_budget_mode == "global":
        k_by = sadnd_cap.allocate_global_fp_budget(scores, cfg.fp_ratio)
        budget_policy = "global"
    else:
        k_by = {i: int(cfg.fp_ratio * scores[i].numel()) for i in indices}
        budget_policy = "fixed"

    # ---- per-layer FP/INT + packing-aware permutation ----
    routing_idx = {}
    for i in indices:
        fp, int_packed, perm, mask = sadnd_cap.build_layer_fp_int_perm(
            scores[i], k_by[i], chan_scale[i], cfg.int_permutation_mode, cfg.w8_group_size)
        routing_idx[i] = {"fp": fp, "int": int_packed, "perm": perm, "mask": mask,
                          "k_fp": int(fp.numel())}

    # ---- SADND-CAP++: equal-budget FP mask refinement ----
    # Keep each layer's k_fp (FP budget) exactly; refine WHICH channels are FP by
    # boundary swaps accepted only if a measured MLP-output proxy improves. Rebuild
    # the packing-aware INT permutation afterwards. Fully static; no runtime topk.
    refine_diag: Dict[int, dict] = {}
    if cfg.use_fp_mask_refinement and cfg.fp_refine_method != "none":
        from . import fp_mask_refinement as fmr
        # capture affine activation y (capped tokens) for the proxy
        y_store: Dict[int, List[torch.Tensor]] = {i: [] for i in indices}
        y_tok: Dict[int, int] = {i: 0 for i in indices}
        for batch in iter_batches(chunks, torch.arange(num_chunks), batch_size=batch_size):
            if all(y_tok[i] >= cfg.fp_refine_max_tokens for i in indices):
                break
            cap.reset()
            _forward_batch(model, batch, device)
            u_by = cap.collect()
            for i in indices:
                if i not in u_by or y_tok[i] >= cfg.fp_refine_max_tokens:
                    continue
                gamma, beta = affine[i]
                y = u_by[i] * gamma.detach().float().cpu()
                if beta is not None:
                    y = y + beta.detach().float().cpu()
                take = cfg.fp_refine_max_tokens - y_tok[i]
                if take < y.shape[0]:
                    y = y[:take]
                y_store[i].append(y)
                y_tok[i] += y.shape[0]
        # priority: cascade budget score if available, else local sensitivity
        priority = {i: (cascade_diag[i]["score"] if i in cascade_diag
                        else float(scores[i].sum())) for i in indices}
        order = sorted(indices, key=lambda i: -float(priority[i]))
        selected = (set(order[: int(cfg.fp_refine_max_layers)])
                    if cfg.fp_refine_max_layers is not None else set(indices))
        for i in order:
            k_fp_before = routing_idx[i]["k_fp"]
            if i not in selected:
                refine_diag[i] = {"refined": False, "num_swaps": 0, "before": None,
                                  "after": None, "accepted": [], "rejected": []}
                continue
            y_i = torch.cat(y_store[i], 0) if y_store[i] else torch.zeros(1, int(scores[i].numel()))
            full_scales = compute_activation_scales(y_i, cfg.p_act, cfg.qmax, cfg.eps_scale, dim=0)
            ctx = fmr.build_refine_context(y_i, full_scales, layer_mods[i].mlp, device=device,
                                           qmax=cfg.qmax, eps=cfg.eps_scale,
                                           max_tokens=cfg.fp_refine_max_tokens)
            r = fmr.refine_layer_mask(ctx, scores[i], routing_idx[i]["fp"],
                                      routing_idx[i]["int"], cfg)
            del ctx
            fp_new = r["fp_indices"]
            int_new = r["int_indices"]                      # original-order INT set
            assert int(fp_new.numel()) == k_fp_before, "refinement must preserve k_fp"
            int_packed = sadnd_cap.build_packing_aware_int_permutation(
                int_new, chan_scale[i][int_new], cfg.int_permutation_mode, cfg.w8_group_size)
            perm = torch.cat([fp_new, int_packed], dim=0)
            mask = torch.zeros(int(scores[i].numel()), dtype=torch.bool)
            mask[fp_new] = True
            routing_idx[i] = {"fp": fp_new, "int": int_packed, "perm": perm, "mask": mask,
                              "k_fp": int(fp_new.numel())}
            refine_diag[i] = {"refined": bool(r["refined"]), "num_swaps": int(r["num_swaps"]),
                              "before": r["error_before"], "after": r["error_after"],
                              "accepted": r["accepted"], "rejected": r["rejected"]}
            _log(f"layer {i}: refine swaps={r['num_swaps']} "
                 f"err {r['error_before']}->{r['error_after']}")

    # ---- Pass 2: activation scales on the (packed) INT block ----
    # When GT is on we also retain a capped pre-affine ``u`` sample per layer for
    # the GT proxy (full-channel, since GT needs the FP branch too).
    gt_on = cfg.use_groupwise_clip_gain_tuning
    cap.reset()
    yI_store: Dict[int, List[torch.Tensor]] = {i: [] for i in indices}
    yI_tok: Dict[int, int] = {i: 0 for i in indices}
    u_gt_store: Dict[int, List[torch.Tensor]] = {i: [] for i in indices}
    u_gt_tok: Dict[int, int] = {i: 0 for i in indices}
    all_idx = torch.arange(num_chunks)
    for batch in iter_batches(chunks, all_idx, batch_size=batch_size):
        cap.reset()
        _forward_batch(model, batch, device)
        u_by = cap.collect()
        for i in indices:
            if i not in u_by:
                continue
            u = u_by[i]
            if gt_on and u_gt_tok[i] < cfg.gt_max_tokens:
                take = cfg.gt_max_tokens - u_gt_tok[i]
                uu = u[:take] if take < u.shape[0] else u
                u_gt_store[i].append(uu.cpu())
                u_gt_tok[i] += uu.shape[0]
            if yI_tok[i] >= cfg.act_scale_max_tokens:
                continue
            int_idx = routing_idx[i]["int"]
            u_I = u.index_select(-1, int_idx)
            gamma, beta = affine[i]
            gI = gamma.detach().float().cpu().index_select(0, int_idx)
            y_I = u_I * gI
            if beta is not None:
                y_I = y_I + beta.detach().float().cpu().index_select(0, int_idx)
            take = cfg.act_scale_max_tokens - yI_tok[i]
            if take < y_I.shape[0]:
                y_I = y_I[:take]
            yI_store[i].append(y_I)
            yI_tok[i] += y_I.shape[0]
    cap.detach()

    # ---- per-layer frozen INT activation scales ----
    act_scales_by: Dict[int, torch.Tensor] = {}
    for i in indices:
        c_int = int(routing_idx[i]["int"].numel())
        yI = torch.cat(yI_store[i], 0) if yI_store[i] else torch.zeros(1, max(1, c_int))
        act_scales_by[i] = compute_activation_scales(yI, cfg.p_act, cfg.qmax, cfg.eps_scale, dim=0)

    # ---- SADND-CAP-GT: static groupwise clip-gain tuning of the INT branch ----
    # tau folds into act_scales_by; eta (gains) stored per projection for patch-time
    # folding into INT weight columns. FP budget unchanged; accept-only per layer.
    gt_diag: Dict[int, dict] = {}
    if gt_on:
        from . import groupwise_clip_gain as gcg
        from types import SimpleNamespace
        gt_priority = {i: (cascade_diag[i]["score"] if i in cascade_diag
                           else float(scores[i].sum())) for i in indices}
        gt_order = sorted(indices, key=lambda i: -float(gt_priority[i]))
        gt_selected = (set(gt_order[: int(cfg.gt_max_tuned_layers)])
                       if cfg.gt_max_tuned_layers is not None else set(indices))
        for i in gt_order:
            if i not in gt_selected or int(routing_idx[i]["int"].numel()) == 0:
                gt_diag[i] = {"gt_enabled": False, "tau_values": [], "eta_gate": [],
                              "eta_up": [], "before": None, "after": None, "num_groups": 0,
                              "gain_gate": None, "gain_up": None,
                              "reason": "not selected" if i not in gt_selected else "no INT channels"}
                continue
            gamma, beta = affine[i]
            u_g = torch.cat(u_gt_store[i], 0) if u_gt_store[i] else torch.zeros(1, int(scores[i].numel()))
            y_i = u_g * gamma.detach().float().cpu()
            if beta is not None:
                y_i = y_i + beta.detach().float().cpu()
            view = SimpleNamespace(fp_indices=routing_idx[i]["fp"],
                                   int_indices=routing_idx[i]["int"],
                                   act_scales=act_scales_by[i])
            ctx = gcg.build_gt_context(y_i, view, layer_mods[i].mlp, cfg, device=device,
                                       max_tokens=cfg.gt_max_tokens)
            res = gcg.tune_layer(ctx, cfg)
            del ctx
            if res["gt_enabled"]:
                act_scales_by[i] = res["tuned_scales"]          # tau folded into scales
            gt_diag[i] = res
            _log(f"layer {i}: gt enabled={res['gt_enabled']} "
                 f"proxy {res['before']}->{res['after']} ({res['reason']})")

    # ---- assemble ----
    layers: Dict[int, LayerRouting] = {}
    for i in indices:
        r = routing_idx[i]
        act_scales = act_scales_by[i]
        gtd = gt_diag.get(i, {})
        _, beta = affine[i]
        layers[i] = LayerRouting(
            layer_index=i, num_channels=int(scores[i].numel()), k_fp=r["k_fp"],
            fp_indices=r["fp"].cpu(), int_indices=r["int"].cpu(), perm=r["perm"].cpu(),
            mask=r["mask"].cpu(), delta_tilde=delta_t[i].cpu(), act_scales=act_scales.cpu(),
            gt_enabled=bool(gtd.get("gt_enabled", False)),
            gt_clip_granularity=cfg.gt_clip_granularity if gt_on else None,
            gt_gain_granularity=cfg.gt_gain_granularity if gt_on else None,
            gt_num_groups=int(gtd.get("num_groups", 0)),
            gt_tau_values=list(gtd.get("tau_values", []) or []),
            gt_eta_gate=list(gtd.get("eta_gate", []) or []),
            gt_eta_up=list(gtd.get("eta_up", []) or []),
            gt_int_gain_gate=(gtd.get("gain_gate") if gtd.get("gt_enabled") else None),
            gt_int_gain_up=(gtd.get("gain_up") if gtd.get("gt_enabled") else None),
            gt_proxy_error_before=gtd.get("before"),
            gt_proxy_error_after=gtd.get("after"),
            gt_reason=gtd.get("reason"),
            w8_group_size=cfg.w8_group_size, routing_score=rs,
            int_permutation_mode=cfg.int_permutation_mode,
            selected_fp_ratio=r["k_fp"] / max(1, int(scores[i].numel())),
            budget_policy=budget_policy,
            cascade_local_error=cascade_diag.get(i, {}).get("e"),
            cascade_error=cascade_diag.get(i, {}).get("cascade"),
            budget_score=cascade_diag.get(i, {}).get("score"),
            refined=refine_diag.get(i, {}).get("refined", False),
            num_refine_swaps=refine_diag.get(i, {}).get("num_swaps", 0),
            refine_proxy_error_before=refine_diag.get(i, {}).get("before"),
            refine_proxy_error_after=refine_diag.get(i, {}).get("after"),
            refine_accepted_swaps=refine_diag.get(i, {}).get("accepted", []),
            refine_rejected_swaps=refine_diag.get(i, {}).get("rejected", []),
            norm_type="layernorm" if beta is not None else "rmsnorm")
        _log(f"layer {i}: C={layers[i].num_channels} K_F={r['k_fp']} "
             f"policy={budget_policy} perm={cfg.int_permutation_mode}")

    return RoutingPlan(config=cfg, layers=layers)
