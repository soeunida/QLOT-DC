"""Q-LOT-OBC equal-budget selection.

Compares all candidates at the SAME fp_ratio (so no method wins merely by using a
larger FP budget). A candidate is accepted as an improvement only if it beats
SADND at the SAME fp_ratio by ``--margin`` (default 0.001).

Candidates (routing_score=sadnd, fp_budget_mode=fixed):
  fp16, int8_ptq (fp0 ref),
  sadnd,
  sadnd_block_{bias,affine,lowrank_r2,lowrank_r4}   (routing-only + block corr)
  qlot_dc_biascorr                                  (DC + projection bias corr)
  qlot_obc_{bias,affine,lowrank_r2,lowrank_r4}      (DC + block corr)

Outputs: equal_budget_selection.{json,csv}, selected_config.json,
per_layer_block_correction_summary.json. PPL only; torch_reference; no speedup.
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
from eval.eval_perplexity import load_model, wikitext2_ppl

_NEUTRAL = {
    "use_grms": False, "use_mean_comp": False, "grms_gating": False,
    "use_static_diag_comp": False, "fp_budget_mode": "fixed",
    "use_projection_bias_correction": False, "use_lowrank_correction": False,
    "use_block_output_correction": False, "block_correction_mode": "none",
}
_DC = {"use_static_diag_comp": True, "diag_comp_mode": "median_scale"}


def _candidates():
    c = {
        "sadnd": {},
        "sadnd_block_bias": {"use_block_output_correction": True, "block_correction_mode": "bias"},
        "sadnd_block_affine": {"use_block_output_correction": True, "block_correction_mode": "affine"},
        "sadnd_block_lowrank_r2": {"use_block_output_correction": True, "block_correction_mode": "lowrank", "block_lowrank_rank": 2},
        "sadnd_block_lowrank_r4": {"use_block_output_correction": True, "block_correction_mode": "lowrank", "block_lowrank_rank": 4},
        "qlot_dc_biascorr": {**_DC, "use_projection_bias_correction": True},
        "qlot_obc_bias": {**_DC, "use_block_output_correction": True, "block_correction_mode": "bias"},
        "qlot_obc_affine": {**_DC, "use_block_output_correction": True, "block_correction_mode": "affine"},
        "qlot_obc_lowrank_r2": {**_DC, "use_block_output_correction": True, "block_correction_mode": "lowrank", "block_lowrank_rank": 2},
        "qlot_obc_lowrank_r4": {**_DC, "use_block_output_correction": True, "block_correction_mode": "lowrank", "block_lowrank_rank": 4},
    }
    return c


def _cfg(base, overrides, fp):
    d = dict(base); d["enable_qlot_rms"] = True; d["qlot_scope"] = "mlp_only"
    d["routing_score"] = "sadnd"
    d.update(_NEUTRAL); d.update(overrides); d["fp_ratio"] = float(fp)
    return QLotRmsConfig.from_dict(d).validate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--max_chunks", type=int, default=32)
    ap.add_argument("--margin", type=float, default=0.001)
    ap.add_argument("--fp_ratios", nargs="+", type=float, default=None)
    ap.add_argument("--candidates", nargs="+", default=None)
    ap.add_argument("--calib_samples", type=int, default=48)
    ap.add_argument("--calib_seq_len", type=int, default=512)
    ap.add_argument("--calib_subsets", type=int, default=3)
    ap.add_argument("--subset_size", type=int, default=24)
    ap.add_argument("--calib_batch_size", type=int, default=8)
    ap.add_argument("--out_dir", default="results/qlot_obc_select")
    ap.add_argument("--allow_synthetic", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = QLotRmsConfig.load_json(args.config).to_dict()
    base.update(calibration_samples=args.calib_samples, calibration_seq_len=args.calib_seq_len,
                num_calib_subsets=args.calib_subsets, subset_size=args.subset_size)
    fp_ratios = args.fp_ratios or base.get("fp_ratio_candidates", [0.04, 0.06, 0.10, 0.20])
    cand = _candidates()
    if args.candidates:
        cand = {k: cand[k] for k in args.candidates if k in cand}

    model, tok = load_model(args.model, args.device)
    fp16_ppl, _, _ = wikitext2_ppl(model, tok, args.device, args.seq_len, args.max_chunks)
    print(f"[obc] fp16 ppl={fp16_ppl:.4f}")

    def eval_cfg(cfg):
        plan = calibrate(model, tok, cfg, device=args.device, routing_method="sadnd",
                         allow_synthetic=args.allow_synthetic, batch_size=args.calib_batch_size)
        h = patch_model(model, plan, cfg)
        try:
            ppl, _, _ = wikitext2_ppl(model, tok, args.device, args.seq_len, args.max_chunks)
        finally:
            unpatch_model(h)
        return ppl

    rows = [{"variant": "fp16", "fp_ratio": None, "ppl": fp16_ppl}]
    # int8_ptq reference (fp0)
    t0 = time.time()
    int8 = eval_cfg(_cfg(base, {}, 0.0))
    rows.append({"variant": "int8_ptq", "fp_ratio": 0.0, "ppl": int8,
                 "seconds": round(time.time() - t0, 1)})
    print(f"[obc] int8_ptq fp=0.00 ppl={int8:.4f}")

    best_cfgs = {}
    per_fp_best = {}
    for fp in fp_ratios:
        sadnd_ppl = None
        for name, ov in cand.items():
            t0 = time.time()
            cfg = _cfg(base, ov, fp)
            ppl = eval_cfg(cfg)
            rows.append({"variant": name, "fp_ratio": float(fp), "ppl": ppl,
                         "seconds": round(time.time() - t0, 1)})
            best_cfgs[(name, float(fp))] = cfg
            if name == "sadnd":
                sadnd_ppl = ppl
            print(f"[obc] {name:24s} fp={fp:.2f} ppl={ppl:.4f}")
        # best non-sadnd candidate at this fp
        cands_fp = [(r["variant"], r["ppl"]) for r in rows
                    if r.get("fp_ratio") == float(fp) and r["variant"] != "sadnd"]
        best_name, best_ppl = min(cands_fp, key=lambda x: x[1])
        per_fp_best[f"{fp:.2f}"] = {
            "sadnd_ppl": sadnd_ppl, "best_candidate": best_name, "best_ppl": best_ppl,
            "delta_vs_sadnd": best_ppl - sadnd_ppl,
            "beats_sadnd_by_margin": best_ppl < sadnd_ppl - args.margin,
        }

    # overall winner: candidate that beats its same-fp SADND by margin, lowest ppl
    winners = []
    for fp in fp_ratios:
        info = per_fp_best[f"{fp:.2f}"]
        if info["beats_sadnd_by_margin"]:
            winners.append((info["best_candidate"], float(fp), info["best_ppl"]))
    if winners:
        wname, wfp, wppl = min(winners, key=lambda x: x[2])
        clear = True
    else:
        # no clear winner -> pick globally-lowest candidate (excluding sadnd/int8/fp16)
        pool = [(r["variant"], r["fp_ratio"], r["ppl"]) for r in rows
                if r["variant"] not in ("fp16", "int8_ptq", "sadnd") and r.get("fp_ratio") is not None]
        wname, wfp, wppl = min(pool, key=lambda x: x[2])
        clear = False

    sel_cfg = best_cfgs[(wname, wfp)]
    sel = {"model": args.model, "seq_len": args.seq_len, "val_chunks": args.max_chunks,
           "margin": args.margin, "fp16_ppl": fp16_ppl, "int8_ptq_ppl": int8,
           "per_fp_best": per_fp_best, "winner": wname, "winner_fp": wfp,
           "winner_ppl": wppl, "clear_improvement_over_sadnd_equal_budget": bool(clear),
           "note": ("Equal-FP-budget selection: a candidate is accepted only if it "
                    "beats SADND at the SAME fp_ratio by margin. PPL only; no speedup."),
           "results": rows}
    json.dump(sel, open(os.path.join(args.out_dir, "equal_budget_selection.json"), "w"), indent=2)
    with open(os.path.join(args.out_dir, "equal_budget_selection.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "fp_ratio", "ppl", "seconds"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in ["variant", "fp_ratio", "ppl", "seconds"]})
    sel_cfg.save_json(os.path.join(args.out_dir, "selected_config.json"))

    # per-layer block-correction summary for the winner (re-calibrate to capture plan)
    plan = calibrate(model, tok, sel_cfg, device=args.device, routing_method="sadnd",
                     allow_synthetic=args.allow_synthetic, batch_size=args.calib_batch_size)
    block_sum = {int(i): {k: lr.summary().get(k) for k in
                          ("block_corr_mode", "block_corr_enabled", "block_mse_before",
                           "block_mse_after", "block_corr_reason")}
                 for i, lr in plan.layers.items()}
    json.dump({"winner": wname, "winner_fp": wfp, "layers": block_sum},
              open(os.path.join(args.out_dir, "per_layer_block_correction_summary.json"), "w"), indent=2)

    print(f"[obc] WINNER={wname} fp={wfp} ppl={wppl:.4f} "
          f"clear_improvement_over_sadnd_equal_budget={clear}")
    print(f"[obc] wrote selection json/csv + selected_config.json + per-layer block summary to {args.out_dir}")


if __name__ == "__main__":
    main()
