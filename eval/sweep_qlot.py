"""Q-LOT-RMS hyperparameter sweep on a SMALL validation split.

Sweeps (defaults match the requested grids):
  fp_ratio        : 0.00 0.02 0.04 0.06 0.08 0.10
  grms_group_size : 32 64 128 256
  p_act           : 0.995 0.999 0.9995
  use_grms        : false true
  use_mean_comp   : false true

Saves:
  results/<out_dir>/sweep_results.json
  results/<out_dir>/sweep_results.csv
  results/<out_dir>/best_config.json

Notes
-----
* Uses a SMALL validation split (``--val_chunks``), NOT the full test set.
* The full cross product is large; combinations are de-duplicated (when
  use_grms=false, grms_group_size and use_mean_comp are irrelevant) and you can
  cap the run with ``--max_configs``.  PPL only; no speedup is reported.

Example (small, fast)
---------------------
    python -m eval.sweep_qlot --model JackFram/llama-160m --device cuda:0 \
        --val_chunks 8 --seq_len 1024 \
        --fp_ratios 0.0 0.06 0.10 --grms_group_sizes 128 --p_acts 0.999 \
        --use_grms false true --use_mean_comp false true \
        --out_dir results/qlot_rms_sweep_demo
"""

import argparse
import csv
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from qlot_rms.config import QLotRmsConfig
from qlot_rms.calibration import calibrate
from qlot_rms.model_integration import patch_model, unpatch_model
from eval.eval_perplexity import base_cfg_dict, load_model, wikitext2_ppl


def _bool_list(vals):
    out = []
    for v in vals:
        out.append(str(v).lower() in ("1", "true", "yes", "on"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--val_chunks", type=int, default=16)
    ap.add_argument("--routing_method", default="sadnd")
    ap.add_argument("--fp_ratios", nargs="+", type=float,
                    default=[0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
    ap.add_argument("--grms_group_sizes", nargs="+", type=int, default=[32, 64, 128, 256])
    ap.add_argument("--p_acts", nargs="+", type=float, default=[0.995, 0.999, 0.9995])
    ap.add_argument("--use_grms", nargs="+", default=["false", "true"])
    ap.add_argument("--use_mean_comp", nargs="+", default=["false", "true"])
    ap.add_argument("--calib_samples", type=int, default=64)
    ap.add_argument("--calib_seq_len", type=int, default=512)
    ap.add_argument("--calib_subsets", type=int, default=3)
    ap.add_argument("--subset_size", type=int, default=16)
    ap.add_argument("--calib_batch_size", type=int, default=8)
    ap.add_argument("--lambda_agg", type=float, default=1.0)
    ap.add_argument("--p_proxy", type=float, default=0.9995)
    ap.add_argument("--w8_group_size", type=int, default=128)
    ap.add_argument("--routed_layers", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="torch_reference")
    ap.add_argument("--fp_ratio", type=float, default=0.06)        # for base_cfg_dict
    ap.add_argument("--grms_group_size", type=int, default=128)
    ap.add_argument("--p_act", type=float, default=0.999)
    ap.add_argument("--max_configs", type=int, default=0, help="0 = no cap")
    ap.add_argument("--out_dir", default="results/qlot_rms_sweep")
    ap.add_argument("--allow_synthetic", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    grms_opts = _bool_list(args.use_grms)
    mc_opts = _bool_list(args.use_mean_comp)

    # build de-duplicated config grid
    seen = set()
    grid = []
    for fp, gs, pa, ug, mc in itertools.product(
        args.fp_ratios, args.grms_group_sizes, args.p_acts, grms_opts, mc_opts
    ):
        # when GroupRMS is off, group size and mean-comp don't matter -> canonicalize
        key_gs = gs if ug else 0
        key_mc = mc if ug else False
        key = (round(fp, 4), key_gs, round(pa, 5), ug, key_mc)
        if key in seen:
            continue
        seen.add(key)
        grid.append({"fp_ratio": fp, "grms_group_size": gs, "p_act": pa,
                     "use_grms": ug, "use_mean_comp": (mc if ug else False)})
    if args.max_configs and len(grid) > args.max_configs:
        print(f"[sweep] capping {len(grid)} configs to {args.max_configs}")
        grid = grid[:args.max_configs]
    print(f"[sweep] {len(grid)} unique configs on {args.val_chunks} val chunks")

    model, tok = load_model(args.model, args.device)
    base = base_cfg_dict(args)
    base["enable_qlot_rms"] = True
    base["qlot_scope"] = "mlp_only"

    fp16_ppl, _, _ = wikitext2_ppl(model, tok, args.device, args.seq_len, args.val_chunks)
    print(f"[sweep] fp16 reference ppl={fp16_ppl:.4f}")

    rows = []
    best = None
    for j, g in enumerate(grid):
        cfg = QLotRmsConfig.from_dict({**base, **g}).validate()
        plan = calibrate(model, tok, cfg, device=args.device,
                         routing_method=args.routing_method,
                         allow_synthetic=args.allow_synthetic,
                         batch_size=args.calib_batch_size)
        handle = patch_model(model, plan, cfg)
        try:
            ppl, _, _ = wikitext2_ppl(model, tok, args.device, args.seq_len, args.val_chunks)
        finally:
            unpatch_model(handle)
        rec = {**g, "ppl": ppl, "delta_vs_fp16": ppl - fp16_ppl}
        rows.append(rec)
        if best is None or ppl < best["ppl"]:
            best = rec
        print(f"[sweep] {j+1}/{len(grid)} fp={g['fp_ratio']} gs={g['grms_group_size']} "
              f"pa={g['p_act']} grms={g['use_grms']} mc={g['use_mean_comp']} -> ppl={ppl:.4f}")

    out = {"model": args.model, "seq_len": args.seq_len, "val_chunks": args.val_chunks,
           "backend": args.backend, "routing_method": args.routing_method,
           "fp16_reference_ppl": fp16_ppl,
           "note": "small validation split; PPL only; no speedup claimed.",
           "results": rows, "best": best}
    json_path = os.path.join(args.out_dir, "sweep_results.json")
    csv_path = os.path.join(args.out_dir, "sweep_results.csv")
    best_path = os.path.join(args.out_dir, "best_config.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    keys = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)
    # full QLotRmsConfig for the best point
    best_cfg = QLotRmsConfig.from_dict({**base, **{k: best[k] for k in
                ("fp_ratio", "grms_group_size", "p_act", "use_grms", "use_mean_comp")}})
    best_cfg.save_json(best_path)
    print(f"[sweep] best: {best}")
    print(f"[sweep] wrote {json_path}, {csv_path}, {best_path}")


if __name__ == "__main__":
    main()
