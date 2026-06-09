"""SADND-CAP / CAP+ / CAP++ / CAP-GT equal-budget selection.

Compares candidates at the SAME fp_ratio (total FP budget). A candidate is
accepted only if it beats clean SADND@same-fp by ``accept_only_margin``. Some
candidates carry an EXTRA base they must also beat:
  * refined (CAP++) candidates must beat their unrefined CAP+ counterpart by
    ``fp_refine_margin``;
  * GT (CAP-GT) candidates must beat their non-GT counterpart by
    ``gt_accept_margin``.
Otherwise the strongest SADND baseline is selected and ``clear_improvement=false``
is recorded with a reason. No method wins by using more FP. PPL only; no speedup.

Outputs: sadnd_cap_selection.{json,csv}, selected_config.json,
layerwise_fp_budget.json, cascade_budget_summary.json, marginal_gain_table.csv,
int_permutation_summary.json, fp_mask_refinement_summary.json,
groupwise_clip_gain_summary.json.
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from qlot_rms.config import QLotRmsConfig
from qlot_rms.calibration import calibrate
from qlot_rms.model_integration import patch_model, unpatch_model
from qlot_rms import cascade_budget as cb
from eval.eval_perplexity import load_model, wikitext2_ppl

_OA = {"routing_score": "output_aware_sadnd", "int_permutation_mode": "packing_aware"}
_CM = {**_OA, "fp_budget_mode": "cascade", "use_cascade_aware_budget": True,
       "use_marginal_gain_allocation": True}          # CAP+ cascade+marginal base
CANDIDATES = {
    "sadnd_fixed_original": {"routing_score": "sadnd", "fp_budget_mode": "fixed",
                             "int_permutation_mode": "original"},
    "output_aware_sadnd_fixed_original": {"routing_score": "output_aware_sadnd",
                             "fp_budget_mode": "fixed", "int_permutation_mode": "original"},
    "sadnd_cap_global_packing": {**_OA, "fp_budget_mode": "global"},
    "sadnd_cap_cascade_budget": {**_OA, "fp_budget_mode": "cascade",
                                 "use_cascade_aware_budget": True},
    "sadnd_cap_marginal_gain_budget": {**_OA, "fp_budget_mode": "marginal",
                                       "use_marginal_gain_allocation": True},
    "sadnd_cap_cascade_marginal": dict(_CM),
    # --- CAP++ : equal-budget FP mask refinement ---
    "sadnd_cap_global_packing_refined": {**_OA, "fp_budget_mode": "global",
                                         "use_fp_mask_refinement": True},
    "sadnd_cap_cascade_marginal_refined": {**_CM, "use_fp_mask_refinement": True},
    # --- CAP-GT : static groupwise clip-gain tuning ---
    "sadnd_cap_cascade_marginal_gt_layer": {**_CM, "use_groupwise_clip_gain_tuning": True,
                                            "gt_gain_granularity": "layer"},
    "sadnd_cap_cascade_marginal_gt_group": {**_CM, "use_groupwise_clip_gain_tuning": True,
                                            "gt_gain_granularity": "group"},
    "sadnd_cap_cascade_marginal_refined_gt": {**_CM, "use_fp_mask_refinement": True,
                                              "use_groupwise_clip_gain_tuning": True},
}
BASELINE = "sadnd_fixed_original"
# candidate -> (extra base it must also beat, margin-key)
BASE_OF = {
    "sadnd_cap_global_packing_refined": ("sadnd_cap_global_packing", "refine"),
    "sadnd_cap_cascade_marginal_refined": ("sadnd_cap_cascade_marginal", "refine"),
    "sadnd_cap_cascade_marginal_gt_layer": ("sadnd_cap_cascade_marginal", "gt"),
    "sadnd_cap_cascade_marginal_gt_group": ("sadnd_cap_cascade_marginal", "gt"),
    "sadnd_cap_cascade_marginal_refined_gt": ("sadnd_cap_cascade_marginal_refined", "gt"),
}
REFINE_CANDS = {k for k, v in CANDIDATES.items() if v.get("use_fp_mask_refinement")}
GT_CANDS = {k for k, v in CANDIDATES.items() if v.get("use_groupwise_clip_gain_tuning")}
_RESET = {"use_cascade_aware_budget": False, "use_marginal_gain_allocation": False,
          "use_fp_mask_refinement": False, "use_groupwise_clip_gain_tuning": False,
          "fp_budget_mode": "fixed", "int_permutation_mode": "original"}


def _cfg(base, ov, fp):
    d = dict(base); d.update(enable_qlot_rms=True, qlot_scope="mlp_only", method="sadnd_cap")
    d.update(_RESET); d.update(ov)
    d["fp_ratio"] = float(fp); d["global_fp_budget_ratio"] = float(fp)
    return QLotRmsConfig.from_dict(d).validate()


def _agg(xs, fn):
    xs = [x for x in xs if x is not None]
    return fn(xs) if xs else None


def _refine_stats(plan):
    layers = list(plan.layers.values())
    refined = [lr for lr in layers if getattr(lr, "refined", False)]
    return {
        "refined_layers": len(refined),
        "accepted_swaps": sum(len(getattr(lr, "refine_accepted_swaps", []) or []) for lr in layers),
        "rejected_swaps": sum(len(getattr(lr, "refine_rejected_swaps", []) or []) for lr in layers),
        "mean_proxy_before": _agg([lr.refine_proxy_error_before for lr in refined], lambda v: sum(v)/len(v)),
        "mean_proxy_after": _agg([lr.refine_proxy_error_after for lr in refined], lambda v: sum(v)/len(v)),
        "fp_budget": sum(lr.k_fp for lr in layers),
    }


def _gt_stats(plan):
    layers = list(plan.layers.values())
    tuned = [lr for lr in layers if getattr(lr, "gt_enabled", False)]
    taus = [t for lr in tuned for t in (lr.gt_tau_values or [])]
    etas = [e for lr in tuned for e in ((lr.gt_eta_gate or []) + (lr.gt_eta_up or []))]
    return {
        "tuned_layers": len(tuned),
        "clip_granularity": _agg([lr.gt_clip_granularity for lr in layers], lambda v: v[0]),
        "gain_granularity": _agg([lr.gt_gain_granularity for lr in layers], lambda v: v[0]),
        "mean_tau": _agg(taus, lambda v: sum(v)/len(v)), "min_tau": _agg(taus, min), "max_tau": _agg(taus, max),
        "mean_eta": _agg(etas, lambda v: sum(v)/len(v)), "min_eta": _agg(etas, min), "max_eta": _agg(etas, max),
        "gt_proxy_before": _agg([lr.gt_proxy_error_before for lr in tuned], lambda v: sum(v)/len(v)),
        "gt_proxy_after": _agg([lr.gt_proxy_error_after for lr in tuned], lambda v: sum(v)/len(v)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--max_chunks", type=int, default=32)
    ap.add_argument("--fp_ratios", nargs="+", type=float, default=None)
    ap.add_argument("--candidates", nargs="+", default=None)
    ap.add_argument("--calib_samples", type=int, default=48)
    ap.add_argument("--calib_seq_len", type=int, default=512)
    ap.add_argument("--calib_subsets", type=int, default=3)
    ap.add_argument("--subset_size", type=int, default=24)
    ap.add_argument("--calib_batch_size", type=int, default=8)
    ap.add_argument("--out_dir", default="results/sadnd_cap_select")
    ap.add_argument("--allow_synthetic", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = QLotRmsConfig.load_json(args.config).to_dict()
    base.update(calibration_samples=args.calib_samples, calibration_seq_len=args.calib_seq_len,
                num_calib_subsets=args.calib_subsets, subset_size=args.subset_size)
    margin = base.get("accept_only_margin", 0.001)
    refine_margin = base.get("fp_refine_margin", 0.0005)
    gt_margin = base.get("gt_accept_margin", 0.0005)
    base_margin = {"refine": refine_margin, "gt": gt_margin}
    fp_ratios = args.fp_ratios or base.get("fp_ratio_candidates", [0.06, 0.20])
    cand = CANDIDATES if not args.candidates else {k: CANDIDATES[k] for k in args.candidates}

    model, tok = load_model(args.model, args.device)

    def eval_cfg(cfg):
        plan = calibrate(model, tok, cfg, device=args.device,
                         allow_synthetic=args.allow_synthetic, batch_size=args.calib_batch_size)
        h = patch_model(model, plan, cfg)
        try:
            ppl, _, _ = wikitext2_ppl(model, tok, args.device, args.seq_len, args.max_chunks)
        finally:
            unpatch_model(h)
        return ppl, plan

    fp16_ppl, _, _ = wikitext2_ppl(model, tok, args.device, args.seq_len, args.max_chunks)
    print(f"[cap] fp16 ppl={fp16_ppl:.4f}")
    rows = [{"variant": "fp16", "fp_ratio": None, "ppl": fp16_ppl}]
    t0 = time.time()
    int8, _ = eval_cfg(_cfg(base, CANDIDATES[BASELINE], 0.0))
    rows.append({"variant": "int8_ptq", "fp_ratio": 0.0, "ppl": int8, "seconds": round(time.time()-t0, 1)})
    print(f"[cap] int8_ptq fp=0.00 ppl={int8:.4f}")

    ppl_by, plan_by = {}, {}
    for fp in fp_ratios:
        for name, ov in cand.items():
            t0 = time.time()
            cfg = _cfg(base, ov, fp)
            ppl, plan = eval_cfg(cfg)
            ppl_by[(name, float(fp))] = ppl
            plan_by[(name, float(fp))] = (cfg, plan)
            row = {"variant": name, "fp_ratio": float(fp), "ppl": ppl,
                   "budget_policy": next(iter(plan.layers.values())).budget_policy,
                   "seconds": round(time.time()-t0, 1)}
            if name in REFINE_CANDS:
                row.update(_refine_stats(plan))
            if name in GT_CANDS:
                row.update(_gt_stats(plan))
            rows.append(row)
            extra = ""
            if name in GT_CANDS:
                extra = f" gt_layers={row.get('tuned_layers')} mean_tau={row.get('mean_tau')} mean_eta={row.get('mean_eta')}"
            elif name in REFINE_CANDS:
                extra = f" refined_layers={row.get('refined_layers')} swaps={row.get('accepted_swaps')}"
            print(f"[cap] {name:42s} fp={fp:.2f} ppl={ppl:.4f}{extra}")

    # ---- per-fp candidate analysis ----
    per_fp = {}
    for fp in fp_ratios:
        fpf = float(fp)
        sadnd_ppl = ppl_by[(BASELINE, fpf)]
        cm_ppl = ppl_by.get(("sadnd_cap_cascade_marginal", fpf))
        cands = {}
        for name in cand:
            if name == BASELINE:
                continue
            p = ppl_by[(name, fpf)]
            rec = {"ppl": p, "delta_vs_sadnd": p - sadnd_ppl,
                   "delta_vs_cascade_marginal": (p - cm_ppl) if cm_ppl is not None else None}
            beats_sadnd = p < sadnd_ppl - margin
            rec["beats_sadnd_by_margin"] = bool(beats_sadnd)
            if name in BASE_OF:
                base_name, mkey = BASE_OF[name]
                bp = ppl_by.get((base_name, fpf))
                bm = base_margin[mkey]
                rec["base_candidate"] = base_name
                rec["delta_vs_base"] = (p - bp) if bp is not None else None
                beats_base = (bp is not None) and (p < bp - bm)
                rec["beats_base_by_margin"] = bool(beats_base)
                rec["accepted"] = bool(beats_sadnd and beats_base)
            else:
                rec["accepted"] = bool(beats_sadnd)
            if name in REFINE_CANDS:
                rec.update(_refine_stats(plan_by[(name, fpf)][1]))
            if name in GT_CANDS:
                rec.update(_gt_stats(plan_by[(name, fpf)][1]))
            if not rec["accepted"]:
                reasons = []
                if not beats_sadnd:
                    reasons.append("does not beat clean SADND by accept_only_margin")
                if name in BASE_OF and not rec.get("beats_base_by_margin"):
                    base_name, mkey = BASE_OF[name]
                    reasons.append(f"does not beat {base_name} by {mkey}_margin")
                rec["fallback_reason"] = "; ".join(reasons)
            cands[name] = rec
        per_fp[f"{fpf:.2f}"] = {"baseline_sadnd_ppl": sadnd_ppl, "candidates": cands}

    # ---- winner: best accepted candidate across all fp; else SADND fallback ----
    eligible = []
    for fp in fp_ratios:
        fpf = float(fp)
        for name, rec in per_fp[f"{fpf:.2f}"]["candidates"].items():
            if rec.get("accepted"):
                eligible.append((name, fpf, rec["ppl"]))
    if eligible:
        wname, wfp, wppl = min(eligible, key=lambda x: x[2]); clear = True; reason = None
    else:
        srows = [(r["fp_ratio"], r["ppl"]) for r in rows if r["variant"] == BASELINE]
        wfp, wppl = min(srows, key=lambda x: x[1]); wname = BASELINE; clear = False
        reason = "no candidate cleared the equal-budget (refinement/GT) margins; fell back to clean SADND"

    sel_cfg, sel_plan = plan_by[(wname, wfp)]
    sel = {"model": args.model, "seq_len": args.seq_len, "val_chunks": args.max_chunks,
           "accept_only_margin": margin, "fp_refine_margin": refine_margin, "gt_accept_margin": gt_margin,
           "fp16_ppl": fp16_ppl, "int8_ptq_ppl": int8,
           "per_fp": per_fp, "winner": wname, "winner_fp": wfp, "winner_ppl": wppl,
           "clear_improvement": bool(clear), "fallback_reason": reason,
           "note": ("Equal-FP-budget selection. A candidate is accepted only if it beats clean "
                    "SADND@same-fp by accept_only_margin; refined candidates must also beat their "
                    "unrefined CAP+ counterpart by fp_refine_margin; GT candidates must also beat "
                    "their non-GT counterpart by gt_accept_margin. PPL only; no speedup."),
           "results": rows}
    json.dump(sel, open(os.path.join(args.out_dir, "sadnd_cap_selection.json"), "w"), indent=2)
    with open(os.path.join(args.out_dir, "sadnd_cap_selection.csv"), "w", newline="") as f:
        cols = ["variant", "fp_ratio", "ppl", "budget_policy", "refined_layers", "accepted_swaps",
                "tuned_layers", "mean_tau", "mean_eta", "gt_proxy_before", "gt_proxy_after", "seconds"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    sel_cfg.save_json(os.path.join(args.out_dir, "selected_config.json"))

    # layerwise FP budget + int permutation + cascade summary (from winner plan)
    json.dump({"winner": wname, "winner_fp": wfp,
               "layers": {int(i): {"k_fp": lr.k_fp, "num_channels": lr.num_channels,
                                   "selected_fp_ratio": lr.selected_fp_ratio,
                                   "budget_policy": lr.budget_policy, "refined": lr.refined,
                                   "gt_enabled": lr.gt_enabled}
                          for i, lr in sel_plan.layers.items()}},
              open(os.path.join(args.out_dir, "layerwise_fp_budget.json"), "w"), indent=2)
    json.dump({"winner": wname, "int_permutation_mode": sel_cfg.int_permutation_mode,
               "layers": {int(i): {"k_int": int(lr.int_indices.numel()),
                                   "int_permutation_mode": lr.int_permutation_mode}
                          for i, lr in sel_plan.layers.items()}},
              open(os.path.join(args.out_dir, "int_permutation_summary.json"), "w"), indent=2)
    json.dump({"winner": wname, "winner_fp": wfp,
               "cascade_beta": sel_cfg.cascade_beta, "cascade_gamma": sel_cfg.cascade_gamma,
               "layers": {int(i): {"local_error": lr.cascade_local_error,
                                   "cascade_error": lr.cascade_error,
                                   "budget_score": lr.budget_score, "k_fp": lr.k_fp}
                          for i, lr in sel_plan.layers.items()}},
              open(os.path.join(args.out_dir, "cascade_budget_summary.json"), "w"), indent=2)
    mg = cb.compute_marginal_gain_table({int(i): lr.delta_tilde for i, lr in sel_plan.layers.items()},
                                        sel_cfg.marginal_fp_candidates)
    cols2 = sorted({k for r in mg.values() for k in r})
    with open(os.path.join(args.out_dir, "marginal_gain_table.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer"] + cols2)
        for i in sorted(mg):
            w.writerow([i] + [round(mg[i][c], 6) for c in cols2])

    def _per_fp_summary(names, stats_fn, per_layer_fn):
        out = {}
        for fp in fp_ratios:
            fpf = float(fp); entry = {}
            for name in names:
                if (name, fpf) not in plan_by:
                    continue
                _, plan = plan_by[(name, fpf)]
                rec = per_fp[f"{fpf:.2f}"]["candidates"].get(name, {})
                entry[name] = {**stats_fn(plan), "ppl": ppl_by.get((name, fpf)),
                               "delta_vs_sadnd": rec.get("delta_vs_sadnd"),
                               "delta_vs_base": rec.get("delta_vs_base"),
                               "base_candidate": rec.get("base_candidate"),
                               "delta_vs_cascade_marginal": rec.get("delta_vs_cascade_marginal"),
                               "accepted": rec.get("accepted"),
                               "fallback_reason": rec.get("fallback_reason"),
                               "per_layer": per_layer_fn(plan)}
            out[f"{fpf:.2f}"] = entry
        return out

    json.dump({"fp_refine_margin": refine_margin, "accept_only_margin": margin,
               "per_fp": _per_fp_summary(REFINE_CANDS, _refine_stats, lambda plan: {
                   int(i): {"refined": lr.refined, "num_swaps": lr.num_refine_swaps,
                            "proxy_before": lr.refine_proxy_error_before,
                            "proxy_after": lr.refine_proxy_error_after, "k_fp": lr.k_fp}
                   for i, lr in plan.layers.items()})},
              open(os.path.join(args.out_dir, "fp_mask_refinement_summary.json"), "w"), indent=2)

    json.dump({"gt_accept_margin": gt_margin, "accept_only_margin": margin,
               "per_fp": _per_fp_summary(GT_CANDS, _gt_stats, lambda plan: {
                   int(i): {"gt_enabled": lr.gt_enabled, "num_groups": lr.gt_num_groups,
                            "tau_mean": (sum(lr.gt_tau_values)/len(lr.gt_tau_values)) if lr.gt_tau_values else None,
                            "eta_gate_mean": (sum(lr.gt_eta_gate)/len(lr.gt_eta_gate)) if lr.gt_eta_gate else None,
                            "eta_up_mean": (sum(lr.gt_eta_up)/len(lr.gt_eta_up)) if lr.gt_eta_up else None,
                            "proxy_before": lr.gt_proxy_error_before, "proxy_after": lr.gt_proxy_error_after,
                            "reason": lr.gt_reason}
                   for i, lr in plan.layers.items()})},
              open(os.path.join(args.out_dir, "groupwise_clip_gain_summary.json"), "w"), indent=2)

    print(f"[cap] WINNER={wname} fp={wfp} ppl={wppl:.4f} clear_improvement={clear}")
    if reason:
        print(f"[cap] fallback_reason: {reason}")
    print(f"[cap] wrote selection + selected_config + layerwise_fp_budget + cascade_budget_summary "
          f"+ marginal_gain_table + int_permutation_summary + fp_mask_refinement_summary "
          f"+ groupwise_clip_gain_summary to {args.out_dir}")


if __name__ == "__main__":
    main()
