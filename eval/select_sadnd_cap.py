"""SADND-CAP / SADND-CAP+ equal-budget selection.

Compares candidates at the SAME fp_ratio (total FP budget). A candidate is
accepted only if it beats clean SADND@same-fp by ``accept_only_margin``; else the
strongest SADND baseline is selected and ``clear_improvement=false`` recorded.
No method wins by using more FP. PPL only; no speedup.

Candidates (each at every fp in --fp_ratios):
  sadnd_fixed_original              : sadnd, fixed budget, original INT order (baseline)
  output_aware_sadnd_fixed_original: output-aware, fixed, original
  sadnd_cap_global_packing         : output-aware, global budget, packing-aware
  sadnd_cap_cascade_budget         : output-aware, cascade budget, packing-aware
  sadnd_cap_marginal_gain_budget   : output-aware, marginal-gain budget, packing-aware
  sadnd_cap_cascade_marginal       : output-aware, cascade+marginal, packing-aware
plus fp16 and int8_ptq (fp0) references.

Outputs: sadnd_cap_selection.{json,csv}, selected_config.json,
layerwise_fp_budget.json, cascade_budget_summary.json, marginal_gain_table.csv,
int_permutation_summary.json.
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
# name -> override dict (fp_ratio + global_fp_budget_ratio set per loop)
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
    "sadnd_cap_cascade_marginal": {**_OA, "fp_budget_mode": "cascade",
                                   "use_cascade_aware_budget": True,
                                   "use_marginal_gain_allocation": True},
}
BASELINE = "sadnd_fixed_original"
_RESET = {"use_cascade_aware_budget": False, "use_marginal_gain_allocation": False,
          "fp_budget_mode": "fixed", "int_permutation_mode": "original"}


def _cfg(base, ov, fp):
    d = dict(base); d.update(enable_qlot_rms=True, qlot_scope="mlp_only", method="sadnd_cap")
    d.update(_RESET); d.update(ov)
    d["fp_ratio"] = float(fp); d["global_fp_budget_ratio"] = float(fp)
    return QLotRmsConfig.from_dict(d).validate()


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
    fp_ratios = args.fp_ratios or base.get("fp_ratio_candidates", [0.04, 0.06, 0.10, 0.20])
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

    best_cfgs, best_plans, per_fp = {}, {}, {}
    for fp in fp_ratios:
        base_ppl = None
        for name, ov in cand.items():
            t0 = time.time()
            cfg = _cfg(base, ov, fp)
            ppl, plan = eval_cfg(cfg)
            rows.append({"variant": name, "fp_ratio": float(fp), "ppl": ppl,
                         "budget_policy": next(iter(plan.layers.values())).budget_policy,
                         "seconds": round(time.time()-t0, 1)})
            best_cfgs[(name, float(fp))] = cfg
            best_plans[(name, float(fp))] = plan
            if name == BASELINE:
                base_ppl = ppl
            print(f"[cap] {name:34s} fp={fp:.2f} ppl={ppl:.4f}")
        others = [(r["variant"], r["ppl"]) for r in rows
                  if r.get("fp_ratio") == float(fp) and r["variant"] != BASELINE]
        bname, bppl = min(others, key=lambda x: x[1])
        per_fp[f"{fp:.2f}"] = {"baseline_sadnd_ppl": base_ppl, "best_candidate": bname,
                               "best_ppl": bppl, "delta_vs_sadnd": bppl - base_ppl,
                               "beats_sadnd_by_margin": bppl < base_ppl - margin}

    winners = [(per_fp[f"{fp:.2f}"]["best_candidate"], float(fp), per_fp[f"{fp:.2f}"]["best_ppl"])
               for fp in fp_ratios if per_fp[f"{fp:.2f}"]["beats_sadnd_by_margin"]]
    if winners:
        wname, wfp, wppl = min(winners, key=lambda x: x[2]); clear = True
    else:
        srows = [(r["fp_ratio"], r["ppl"]) for r in rows if r["variant"] == BASELINE]
        wfp, wppl = min(srows, key=lambda x: x[1]); wname = BASELINE; clear = False

    sel_cfg = best_cfgs[(wname, wfp)]; sel_plan = best_plans[(wname, wfp)]
    sel = {"model": args.model, "seq_len": args.seq_len, "val_chunks": args.max_chunks,
           "accept_only_margin": margin, "fp16_ppl": fp16_ppl, "int8_ptq_ppl": int8,
           "per_fp": per_fp, "winner": wname, "winner_fp": wfp, "winner_ppl": wppl,
           "clear_improvement": bool(clear),
           "note": ("Equal-FP-budget selection: accept only if it beats clean SADND@same-fp "
                    "by margin, else fall back to SADND. PPL only; no speedup."),
           "results": rows}
    json.dump(sel, open(os.path.join(args.out_dir, "sadnd_cap_selection.json"), "w"), indent=2)
    with open(os.path.join(args.out_dir, "sadnd_cap_selection.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "fp_ratio", "ppl", "budget_policy", "seconds"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in ["variant", "fp_ratio", "ppl", "budget_policy", "seconds"]})
    sel_cfg.save_json(os.path.join(args.out_dir, "selected_config.json"))

    # layerwise FP budget + int permutation + cascade summary (from winner plan)
    json.dump({"winner": wname, "winner_fp": wfp,
               "layers": {int(i): {"k_fp": lr.k_fp, "num_channels": lr.num_channels,
                                   "selected_fp_ratio": lr.selected_fp_ratio,
                                   "budget_policy": lr.budget_policy}
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
    # marginal-gain table from winner per-layer SADND distortion
    mg = cb.compute_marginal_gain_table({int(i): lr.delta_tilde for i, lr in sel_plan.layers.items()},
                                        sel_cfg.marginal_fp_candidates)
    cands = sorted({k for r in mg.values() for k in r})
    with open(os.path.join(args.out_dir, "marginal_gain_table.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer"] + cands)
        for i in sorted(mg):
            w.writerow([i] + [round(mg[i][c], 6) for c in cands])

    print(f"[cap] WINNER={wname} fp={wfp} ppl={wppl:.4f} clear_improvement={clear}")
    print(f"[cap] wrote selection + selected_config + layerwise_fp_budget + "
          f"cascade_budget_summary + marginal_gain_table + int_permutation_summary to {args.out_dir}")


if __name__ == "__main__":
    main()
