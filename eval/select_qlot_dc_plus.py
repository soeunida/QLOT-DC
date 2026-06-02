"""Validation-PPL-guided selection for Q-LOT-DC+.

Evaluates candidate variants on a SMALL validation split (never the final test
set), then selects the best Q-LOT-DC+ variant. A variant is marked a *clear
improvement* only if its PPL beats both INT8-PTQ and SADND routing-only by a
margin (default 0.001); otherwise the best candidate is still selected but
``clear_improvement=false``.

Candidates: int8_ptq, sadnd, qlot_dc_median, qlot_dc_biascorr,
qlot_dc_output_aware, qlot_dc_adaptive_fp, qlot_dc_biascorr_adaptive_fp,
qlot_dc_lowrank_r2, qlot_dc_lowrank_r4.

Outputs: results/<out_dir>/{variant_selection.json, variant_selection.csv,
selected_config.json}.  No speedup is claimed; torch_reference only.

Example:
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python eval/select_qlot_dc_plus.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda:0 \
        --seq_len 1024 --max_chunks 32 \
        --config configs/qlot_dc_plus_select.json \
        --out_dir results/qlot_dc_plus_select_tinyllama
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
}
_DC = {"use_static_diag_comp": True, "diag_comp_mode": "median_scale"}

# name -> (routing_method, overrides, adaptive_fp)
CANDIDATES = {
    "int8_ptq":  ("sadnd", {"fp_ratio": 0.0}, False),
    "sadnd":     ("sadnd", {}, False),
    "qlot_dc_median":   ("sadnd", {**_DC}, False),
    "qlot_dc_biascorr": ("sadnd", {**_DC, "use_projection_bias_correction": True,
                                   "bias_corr_scope": "gate_up"}, False),
    "qlot_dc_output_aware": ("output_aware_sadnd", {**_DC}, False),
    "qlot_dc_adaptive_fp":  ("sadnd", {**_DC}, True),
    "qlot_dc_biascorr_adaptive_fp": ("sadnd", {**_DC, "use_projection_bias_correction": True,
                                               "bias_corr_scope": "gate_up"}, True),
    "qlot_dc_lowrank_r2": ("sadnd", {**_DC, "use_projection_bias_correction": True,
                                     "use_lowrank_correction": True, "lowrank_rank": 2}, False),
    "qlot_dc_lowrank_r4": ("sadnd", {**_DC, "use_projection_bias_correction": True,
                                     "use_lowrank_correction": True, "lowrank_rank": 4}, False),
}
DC_CANDIDATES = [k for k in CANDIDATES if k not in ("int8_ptq", "sadnd")]


def _cfg(base, overrides, method=None):
    d = dict(base)
    d["enable_qlot_rms"] = True
    d["qlot_scope"] = "mlp_only"
    d.update(_NEUTRAL)
    d.update(overrides)
    # record the ACTUAL routing used so the saved config reproduces the variant
    # (do not inherit the base config's routing_score).
    if method is not None:
        d["routing_score"] = method
    return QLotRmsConfig.from_dict(d).validate()


@torch.no_grad()
def _eval_cfg(model, tok, cfg, method, args):
    plan = calibrate(model, tok, cfg, device=args.device, routing_method=method,
                     allow_synthetic=args.allow_synthetic, batch_size=args.calib_batch_size)
    handle = patch_model(model, plan, cfg)
    try:
        ppl, _, _ = wikitext2_ppl(model, tok, args.device, args.seq_len, args.max_chunks)
    finally:
        unpatch_model(handle)
    return ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--max_chunks", type=int, default=32)
    ap.add_argument("--margin", type=float, default=0.001)
    # lighter calibration for SELECTION (final config can re-calibrate at full size)
    ap.add_argument("--calib_samples", type=int, default=48)
    ap.add_argument("--calib_seq_len", type=int, default=512)
    ap.add_argument("--calib_subsets", type=int, default=3)
    ap.add_argument("--subset_size", type=int, default=24)
    ap.add_argument("--calib_batch_size", type=int, default=8)
    ap.add_argument("--out_dir", default="results/qlot_dc_plus_select")
    ap.add_argument("--allow_synthetic", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = QLotRmsConfig.load_json(args.config).to_dict()
    # apply lighter selection-time calibration
    base.update(calibration_samples=args.calib_samples, calibration_seq_len=args.calib_seq_len,
                num_calib_subsets=args.calib_subsets, subset_size=args.subset_size)
    fp_cands = base.get("fp_ratio_candidates", [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20])

    model, tok = load_model(args.model, args.device)
    fp16_ppl, _, _ = wikitext2_ppl(model, tok, args.device, args.seq_len, args.max_chunks)
    print(f"[select+] fp16 reference ppl={fp16_ppl:.4f}")

    rows = []
    best_cfg_by_name = {}
    for name, (method, ov, adaptive) in CANDIDATES.items():
        t0 = time.time()
        if adaptive:
            best = None
            per_fp = {}
            for fr in fp_cands:
                cfg = _cfg(base, {**ov, "fp_ratio": float(fr), "fp_budget_mode": "fixed"}, method)
                ppl = _eval_cfg(model, tok, cfg, method, args)
                per_fp[f"{fr:.4f}"] = ppl
                if best is None or ppl < best[0]:
                    best = (ppl, float(fr), cfg)
            ppl, sel_fp, cfg = best
            rec = {"variant": name, "ppl": ppl, "selected_fp_ratio": sel_fp,
                   "fp_sweep": per_fp}
        else:
            cfg = _cfg(base, ov, method)
            ppl = _eval_cfg(model, tok, cfg, method, args)
            rec = {"variant": name, "ppl": ppl, "selected_fp_ratio": cfg.fp_ratio}
        rec["seconds"] = round(time.time() - t0, 1)
        rows.append(rec)
        best_cfg_by_name[name] = cfg
        print(f"[select+] {name:30s} ppl={ppl:.4f} fp={rec['selected_fp_ratio']} ({rec['seconds']}s)")

    by = {r["variant"]: r["ppl"] for r in rows}
    base_int8 = by["int8_ptq"]
    base_sadnd = by["sadnd"]
    threshold = min(base_int8, base_sadnd) - args.margin
    # best DC candidate
    dc_sorted = sorted(DC_CANDIDATES, key=lambda n: by[n])
    winner = dc_sorted[0]
    clear = by[winner] < threshold

    sel = {
        "model": args.model, "seq_len": args.seq_len, "val_chunks": args.max_chunks,
        "margin": args.margin, "fp16_reference_ppl": fp16_ppl,
        "baseline_int8_ptq": base_int8, "baseline_sadnd": base_sadnd,
        "threshold": threshold,
        "winner": winner, "winner_ppl": by[winner],
        "clear_improvement": bool(clear),
        "note": ("Small validation split; PPL only; no final-test tuning; no "
                 "speedup. selected_config.json is the winning Q-LOT-DC+ variant."),
        "results": rows,
    }
    json.dump(sel, open(os.path.join(args.out_dir, "variant_selection.json"), "w"), indent=2)
    keys = ["variant", "ppl", "selected_fp_ratio", "seconds"]
    with open(os.path.join(args.out_dir, "variant_selection.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    best_cfg_by_name[winner].save_json(os.path.join(args.out_dir, "selected_config.json"))

    print(f"[select+] WINNER={winner} ppl={by[winner]:.4f} "
          f"vs min(int8={base_int8:.4f}, sadnd={base_sadnd:.4f}) "
          f"clear_improvement={clear}")
    print(f"[select+] wrote variant_selection.json/.csv + selected_config.json to {args.out_dir}")


if __name__ == "__main__":
    main()
