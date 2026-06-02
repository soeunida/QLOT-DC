"""Auto variant selection for Q-LOT-RMS.

Runs a SMALL validation split (not the full test set) for each candidate variant,
picks the lowest-perplexity one, and writes:
  * results/<out_dir>/variant_selection.json   (per-variant PPL + winner)
  * configs/<config_out>/qlot_rms_auto_selected.json  (winning variant's config)

Candidates (same controlled matrix as eval_perplexity):
  int8_ptq, random, magnitude, sadnd, sadnd_grms, sadnd_grms_meancomp
fp16 is measured too as a reference but is not selectable (it is not a Q-LOT-RMS
config).

Example
-------
    python -m eval.select_variant --config configs/qlot_rms_full.json \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda:0 \
        --val_chunks 16 --seq_len 2048 \
        --out_dir results/qlot_rms_auto --config_out configs/auto
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from qlot_rms.config import QLotRmsConfig
from qlot_rms.calibration import calibrate
from qlot_rms.model_integration import patch_model, unpatch_model
from eval.eval_perplexity import VARIANTS, build_cfg, base_cfg_dict, load_model, wikitext2_ppl


SELECTABLE = ["int8_ptq", "random", "magnitude", "sadnd",
              "sadnd_grms", "sadnd_grms_meancomp", "sadnd_grms_gated"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp_ratio", type=float, default=0.06)
    ap.add_argument("--grms_group_size", type=int, default=128)
    ap.add_argument("--lambda_agg", type=float, default=1.0)
    ap.add_argument("--p_proxy", type=float, default=0.9995)
    ap.add_argument("--p_act", type=float, default=0.999)
    ap.add_argument("--w8_group_size", type=int, default=128)
    ap.add_argument("--routed_layers", default="all")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--val_chunks", type=int, default=16,
                    help="SMALL validation split size (chunks), not the full test set.")
    ap.add_argument("--calib_samples", type=int, default=64)
    ap.add_argument("--calib_seq_len", type=int, default=512)
    ap.add_argument("--calib_subsets", type=int, default=3)
    ap.add_argument("--subset_size", type=int, default=32)
    ap.add_argument("--calib_batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="torch_reference")
    ap.add_argument("--candidates", nargs="+", default=SELECTABLE)
    ap.add_argument("--out_dir", default="results/qlot_rms_auto")
    ap.add_argument("--config_out", default="configs/auto")
    ap.add_argument("--allow_synthetic", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.config_out, exist_ok=True)
    model, tok = load_model(args.model, args.device)

    rows = []
    # reference fp16 (not selectable)
    fp16_ppl, _, _ = wikitext2_ppl(model, tok, args.device, args.seq_len, args.val_chunks)
    rows.append({"variant": "fp16", "ppl": fp16_ppl, "selectable": False})
    print(f"[select] fp16 (reference)   ppl={fp16_ppl:.4f}")

    best = None
    for name in args.candidates:
        method, overrides = VARIANTS[name]
        cfg = build_cfg(args, method, overrides)
        plan = calibrate(model, tok, cfg, device=args.device, routing_method=method,
                         allow_synthetic=args.allow_synthetic, batch_size=args.calib_batch_size)
        handle = patch_model(model, plan, cfg)
        try:
            ppl, _, _ = wikitext2_ppl(model, tok, args.device, args.seq_len, args.val_chunks)
        finally:
            unpatch_model(handle)
        rows.append({"variant": name, "ppl": ppl, "selectable": True,
                     "fp_ratio": cfg.fp_ratio, "use_grms": cfg.use_grms,
                     "use_mean_comp": cfg.use_mean_comp})
        print(f"[select] {name:22s} ppl={ppl:.4f}")
        if best is None or ppl < best[1]:
            best = (name, ppl, method, overrides, cfg)

    winner_name, winner_ppl, _, winner_overrides, winner_cfg = best
    selection = {
        "model": args.model, "device": args.device, "seq_len": args.seq_len,
        "val_chunks": args.val_chunks, "backend": args.backend,
        "note": ("Validation is a SMALL split, not the full test set. "
                 "PPL only; no speedup. torch_reference is fake-quantized."),
        "fp16_reference_ppl": fp16_ppl,
        "results": rows,
        "winner": {"variant": winner_name, "ppl": winner_ppl,
                   "overrides": winner_overrides},
    }
    sel_path = os.path.join(args.out_dir, "variant_selection.json")
    with open(sel_path, "w") as f:
        json.dump(selection, f, indent=2)

    cfg_path = os.path.join(args.config_out, "qlot_rms_auto_selected.json")
    winner_cfg.save_json(cfg_path)

    print(f"[select] WINNER: {winner_name} (ppl={winner_ppl:.4f}, "
          f"fp16={fp16_ppl:.4f})")
    print(f"[select] wrote {sel_path} and {cfg_path}")


if __name__ == "__main__":
    main()
