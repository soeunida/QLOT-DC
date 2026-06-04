"""Equal-budget control: does Q-LOT-DC+ beat SADND at the SAME FP ratio?

Loads the model ONCE, then evaluates (variant, fp_ratio) cells:
  int8_ptq (fp=0, reference) and, for each fp in --fp_ratios:
    sadnd     (routing-only: DC off, GroupRMS off, bias-corr off)
    qlot_dc_plus (DC on per the selected config; bias-corr per selected config)
Both use routing_score=sadnd and fp_budget_mode=fixed.

PPL only; torch_reference; no speedup claim. Saves control_results.{json,csv}.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from qlot_rms.config import QLotRmsConfig
from qlot_rms.calibration import calibrate
from qlot_rms.model_integration import patch_model, unpatch_model
from eval.eval_perplexity import load_model, wikitext2_ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="selected DC+ config (for DC flags/calib)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--max_chunks", type=int, default=64)
    ap.add_argument("--fp_ratios", nargs="+", type=float, default=[0.06, 0.10, 0.20])
    ap.add_argument("--calib_batch_size", type=int, default=4)
    ap.add_argument("--out_dir", default="results/qwen25_equal_budget_control")
    ap.add_argument("--allow_synthetic", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = QLotRmsConfig.load_json(args.config).to_dict()
    base["enable_qlot_rms"] = True
    base["qlot_scope"] = "mlp_only"
    base["routing_score"] = "sadnd"
    base["fp_budget_mode"] = "fixed"
    dc_biascorr = bool(base.get("use_projection_bias_correction", False))

    def cfg_for(variant, fp):
        d = dict(base)
        d["fp_ratio"] = float(fp)
        d["use_grms"] = False
        d["use_mean_comp"] = False
        d["use_lowrank_correction"] = False
        if variant == "sadnd" or variant == "int8_ptq":
            d["use_static_diag_comp"] = False
            d["use_projection_bias_correction"] = False
        else:  # qlot_dc_plus
            d["use_static_diag_comp"] = True
            d["diag_comp_mode"] = base.get("diag_comp_mode", "median_scale")
            d["use_projection_bias_correction"] = dc_biascorr
        return QLotRmsConfig.from_dict(d).validate()

    model, tok = load_model(args.model, args.device)
    fp16_ppl, _, _ = wikitext2_ppl(model, tok, args.device, args.seq_len, args.max_chunks)
    print(f"[ctrl] fp16 ppl={fp16_ppl:.4f}")

    rows = [{"variant": "fp16", "fp_ratio": None, "ppl": fp16_ppl}]

    def run(variant, fp):
        cfg = cfg_for(variant, fp)
        plan = calibrate(model, tok, cfg, device=args.device, routing_method="sadnd",
                         allow_synthetic=args.allow_synthetic, batch_size=args.calib_batch_size)
        h = patch_model(model, plan, cfg)
        try:
            ppl, _, _ = wikitext2_ppl(model, tok, args.device, args.seq_len, args.max_chunks)
        finally:
            unpatch_model(h)
        rows.append({"variant": variant, "fp_ratio": float(fp), "ppl": ppl,
                     "dc": cfg.use_static_diag_comp,
                     "bias_corr": cfg.use_projection_bias_correction})
        print(f"[ctrl] {variant:14s} fp={fp:.2f} ppl={ppl:.4f}")
        return ppl

    run("int8_ptq", 0.0)
    for fp in args.fp_ratios:
        run("sadnd", fp)
        run("qlot_dc_plus", fp)

    out = {"model": args.model, "seq_len": args.seq_len, "max_chunks": args.max_chunks,
           "backend": "torch_reference", "fp16_ppl": fp16_ppl,
           "note": ("Equal-FP-budget control. PPL only; no speedup. SADND vs "
                    "Q-LOT-DC+ at identical fp_ratio isolates DC's contribution."),
           "rows": rows}
    json.dump(out, open(os.path.join(args.out_dir, "control_results.json"), "w"), indent=2)
    with open(os.path.join(args.out_dir, "control_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "fp_ratio", "ppl"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in ["variant", "fp_ratio", "ppl"]})
    print(f"[ctrl] wrote {args.out_dir}/control_results.json/.csv")


if __name__ == "__main__":
    main()
