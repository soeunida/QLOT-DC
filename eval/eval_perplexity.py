"""Q-LOT-RMS evaluation: WikiText-2 perplexity across the variant matrix.

Variants (all share the same backend, fp_ratio, routed layers, and branch
shapes so the comparison is controlled):

    fp16                  FP16 baseline (no patch)
    int8_ptq              all FFN gate/up input channels INT8 (fp_ratio=0, no GroupRMS)
    random                random FP/INT routing (no GroupRMS)
    magnitude             magnitude FP/INT routing (no GroupRMS)
    sadnd                 SADND sensitivity routing (no GroupRMS)
    sadnd_grms            SADND routing + INT-branch GroupRMS (no mean-comp)
    sadnd_grms_meancomp   SADND routing + mean-compensated GroupRMS  (full method)

Notes
-----
* This script reports PERPLEXITY only.  It does NOT report any speedup -- the
  reference backend is fake-quantized (correctness, not speed).  Throughput must
  be measured with ``benchmark.py`` and only compared within the same backend.
* The "int8_ptq" baseline here is the controlled FFN-input INT8 path (same
  shapes as Q-LOT-RMS, fp_ratio=0), not a full-model PTQ; it is the fair
  apples-to-apples reference for the routing/GroupRMS ablation.

Example
-------
    python eval/eval_perplexity.py \
        --model meta-llama/Llama-2-7b-hf --device cuda:0 \
        --fp_ratio 0.06 --seq_len 2048 --max_chunks 64 \
        --out_dir eval/results_llama2_7b
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


VARIANTS = {
    # name: (routing_score, overrides)   -- SADND-CAP variant matrix.
    # Baselines force fixed budget + original INT order so they are CLEAN (they do
    # not inherit the config's global budget / packing-aware permutation). "config"
    # runs the full SADND-CAP method as written. All non-fp16 use the config's
    # fp_ratio => equal-FP-budget comparison.
    "fp16":                ("none",                {}),
    "int8_ptq":            ("sadnd",               {"fp_ratio": 0.0,
                                                    "fp_budget_mode": "fixed",
                                                    "int_permutation_mode": "original"}),
    "sadnd":               ("sadnd",               {"fp_budget_mode": "fixed",
                                                    "int_permutation_mode": "original"}),
    "output_aware_sadnd":  ("output_aware_sadnd",  {"fp_budget_mode": "fixed",
                                                    "int_permutation_mode": "original"}),
    # raw: run the base --config exactly as written (= the full SADND-CAP config)
    "config":              ("__raw__",             None),
}


def load_model(model_id, device, dtype=torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device).eval()
    return model, tok


@torch.no_grad()
def wikitext2_ppl(model, tok, device, seq_len=2048, max_chunks=None,
                  hf_name="wikitext-2-raw-v1"):
    from datasets import load_dataset

    ds = load_dataset("wikitext", hf_name, split="test")
    text = "\n\n".join(t for t in ds["text"] if t and t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    n_chunks = ids.numel() // seq_len
    if max_chunks is not None:
        n_chunks = min(n_chunks, max_chunks)
    ids = ids[: n_chunks * seq_len].reshape(n_chunks, seq_len)

    nll_sum, ntok = 0.0, 0
    for i in range(n_chunks):
        batch = ids[i : i + 1].to(device)
        out = model(batch, labels=batch)
        # HF returns mean NLL over (seq_len-1) shifted tokens
        nll_sum += float(out.loss) * (seq_len - 1)
        ntok += (seq_len - 1)
    ppl = float(torch.exp(torch.tensor(nll_sum / max(1, ntok))))
    return ppl, n_chunks, ntok


def base_cfg_dict(args):
    """Base config dict: from --config JSON if given, else SADND-CAP defaults."""
    if getattr(args, "config", None):
        return QLotRmsConfig.load_json(args.config).to_dict()
    d = QLotRmsConfig(enable_qlot_rms=True, qlot_scope="mlp_only").to_dict()
    for k in ("fp_ratio", "p_proxy", "p_act", "calib_samples", "calib_seq_len",
              "calib_subsets", "subset_size", "seed", "backend", "w8_group_size",
              "routed_layers"):
        if hasattr(args, k):
            key = {"calib_samples": "calibration_samples", "calib_seq_len": "calibration_seq_len",
                   "calib_subsets": "num_calib_subsets"}.get(k, k)
            d[key] = getattr(args, k)
    return d


def build_cfg(args, routing_method, overrides):
    base = base_cfg_dict(args)
    base["enable_qlot_rms"] = True
    base["qlot_scope"] = "mlp_only"
    if overrides is None:
        return QLotRmsConfig.from_dict(base).validate()      # "config" raw variant
    base["routing_score"] = routing_method
    base.update(overrides)
    return QLotRmsConfig.from_dict(base).validate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="QLotRmsConfig JSON; supplies all qlot hyper-params. "
                         "Variant overrides (fp_ratio=0 for int8_ptq, GroupRMS "
                         "toggles) are applied on top per variant.")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    help="HF model id (Llama-family). Default is the cached "
                         "TinyLlama-1.1B.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp_ratio", type=float, default=0.06)
    ap.add_argument("--grms_group_size", type=int, default=128)
    ap.add_argument("--lambda_agg", type=float, default=1.0)
    ap.add_argument("--p_proxy", type=float, default=0.9995)
    ap.add_argument("--p_act", type=float, default=0.999)
    ap.add_argument("--w8_group_size", type=int, default=128)
    ap.add_argument("--routed_layers", default="all")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--max_chunks", type=int, default=None)
    ap.add_argument("--calib_samples", type=int, default=128)
    ap.add_argument("--calib_seq_len", type=int, default=512)
    ap.add_argument("--calib_subsets", type=int, default=5)
    ap.add_argument("--subset_size", type=int, default=32)
    ap.add_argument("--calib_batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="torch_reference")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="default depends on --config method: qlot_rms -> the "
                         "GroupRMS matrix; qlot_dc -> fp16/int8_ptq/sadnd + DC set.")
    ap.add_argument("--out_dir", default="eval/results")
    ap.add_argument("--allow_synthetic", action="store_true",
                    help="fallback to synthetic calibration data if WikiText-2 "
                         "cannot be downloaded (PPL still uses real test set).")
    args = ap.parse_args()

    if isinstance(args.routed_layers, str) and args.routed_layers != "all":
        try:
            args.routed_layers = [int(x) for x in args.routed_layers.split(",")]
        except ValueError:
            pass

    os.makedirs(args.out_dir, exist_ok=True)

    # default: compare the supplied --config (run as-is) against fp16 / int8_ptq /
    # sadnd / output_aware_sadnd. sadnd inherits the config's fp_ratio -> an
    # EQUAL-FP-BUDGET comparison.
    if args.variants is None:
        args.variants = ["fp16", "int8_ptq", "sadnd", "output_aware_sadnd", "config"]

    print(f"[eval] loading {args.model} on {args.device}")
    model, tok = load_model(args.model, args.device)

    results = []
    meta_summaries = {}
    for name in args.variants:
        if name not in VARIANTS:
            print(f"[eval] skip unknown variant {name}"); continue
        method, overrides = VARIANTS[name]
        if method == "__raw__":
            method = base_cfg_dict(args).get("routing_score", "sadnd")
        t0 = time.time()
        if name == "fp16":
            ppl, n_chunks, ntok = wikitext2_ppl(
                model, tok, args.device, args.seq_len, args.max_chunks)
            rec = {"variant": name, "ppl": ppl}
        else:
            cfg = build_cfg(args, method, overrides)
            plan = calibrate(model, tok, cfg, device=args.device,
                             routing_method=method, allow_synthetic=args.allow_synthetic,
                             batch_size=args.calib_batch_size, verbose=False)
            handle = patch_model(model, plan, cfg)
            try:
                ppl, n_chunks, ntok = wikitext2_ppl(
                    model, tok, args.device, args.seq_len, args.max_chunks)
            finally:
                unpatch_model(handle)
            meta_summaries[name] = {int(i): plan.layers[i].summary()
                                    for i in plan.layers}
            ex = next(iter(plan.layers.values()))
            rec = {
                "variant": name, "ppl": ppl,
                "fp_ratio": cfg.fp_ratio, "k_fp": ex.k_fp, "C": ex.num_channels,
                "routing_score": cfg.routing_score,
                "fp_budget_mode": cfg.fp_budget_mode,
                "int_permutation_mode": cfg.int_permutation_mode,
                "selected_fp_ratio": ex.selected_fp_ratio,
                "routed_layers": len(plan.layers),
            }
        rec["seconds"] = round(time.time() - t0, 1)
        rec["seq_len"] = args.seq_len
        rec["eval_chunks"] = n_chunks
        results.append(rec)
        print(f"[eval] {name:22s} ppl={rec['ppl']:.4f} ({rec['seconds']}s)")

    meta = {
        "model": args.model, "backend": args.backend, "device": args.device,
        "fp_ratio": args.fp_ratio, "seq_len": args.seq_len,
        "note": ("PPL only. No speedup is reported: torch_reference is "
                 "fake-quantized (correctness-only). SADND-CAP method."),
        "results": results,
    }
    json_path = os.path.join(args.out_dir, "ppl_results.json")
    csv_path = os.path.join(args.out_dir, "ppl_results.csv")
    meta_path = os.path.join(args.out_dir, "metadata_summary.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    with open(meta_path, "w") as f:
        json.dump({"model": args.model, "variants": meta_summaries}, f, indent=2)
    keys = sorted({k for r in results for k in r})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"[eval] wrote {json_path}, {csv_path}, {meta_path}")


if __name__ == "__main__":
    main()
