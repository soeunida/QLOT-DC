"""StaticScale multi-seed equal-budget robustness check (public entrypoint).

Loads a model once and, for each seed, compares clean SADND vs CAP+ (cascade +
marginal-gain budget) vs StaticScale (full: + FP mask refinement + static groupwise
clip-gain tuning) at the SAME FP budget. Reports per-seed PPL and deltas, and the
aggregate decision (robust iff >=2/3 seeds clear -0.001 AND mean delta < -0.001),
both vs clean SADND and vs CAP+.

Writes ``multiseed_results.json`` and ``gt_multiseed_summary.csv``. No speedup is
claimed (``torch_reference`` is correctness-only).

Example:
    python eval/run_staticscale_multiseed.py --model Qwen/Qwen2.5-7B \\
        --fp_ratio 0.20 --seeds 0,1,2 --max_chunks 64 \\
        --out_dir results/staticscale_multiseed_qwen25_7b
"""
import argparse
import csv
import json
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from staticscale import StaticScaleConfig, calibrate, patch_model, unpatch_model  # noqa: E402
from eval.eval_perplexity import load_model, wikitext2_ppl  # noqa: E402

VARIANTS = ["clean_sadnd", "cap_plus", "staticscale"]


def make_cfg(seed, variant, fp):
    base = dict(enable_qlot_rms=True, method="sadnd_cap", qlot_scope="mlp_only",
                fp_ratio=fp, global_fp_budget_ratio=fp, calibration_samples=64,
                calibration_seq_len=512, num_calib_subsets=3, subset_size=32,
                backend="torch_reference", seed=seed)
    if variant == "clean_sadnd":
        base.update(routing_score="sadnd", fp_budget_mode="fixed",
                    int_permutation_mode="original")
    elif variant == "cap_plus":
        base.update(routing_score="output_aware_sadnd", fp_budget_mode="cascade",
                    int_permutation_mode="packing_aware", use_cascade_aware_budget=True,
                    use_marginal_gain_allocation=True, cascade_beta=0.9, cascade_gamma=1.0)
    elif variant == "staticscale":
        base.update(routing_score="output_aware_sadnd", fp_budget_mode="cascade",
                    int_permutation_mode="packing_aware", use_cascade_aware_budget=True,
                    use_marginal_gain_allocation=True, cascade_beta=0.9, cascade_gamma=1.0,
                    use_fp_mask_refinement=True, fp_refine_method="greedy_swap",
                    fp_refine_margin=0.0005,
                    use_groupwise_clip_gain_tuning=True, gt_tune_clip=True, gt_tune_gain=True,
                    gt_clip_granularity="group", gt_gain_granularity="layer", gt_group_size=128,
                    gt_clip_candidates=[0.75, 1.0, 1.25, 1.5, 2.0], gt_gain_clip_min=0.8,
                    gt_gain_clip_max=1.2, gt_metric="projection_mse", gt_accept_only=True,
                    gt_accept_margin=0.0005)
    return StaticScaleConfig(**base)


def aggregate(deltas):
    return {"mean_delta": st.mean(deltas), "std_delta": st.pstdev(deltas),
            "n_clearing": sum(d < -0.001 for d in deltas),
            "robust_better": (sum(d < -0.001 for d in deltas) >= 2 and st.mean(deltas) < -0.001)}


def main():
    ap = argparse.ArgumentParser(description="StaticScale multi-seed robustness")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp_ratio", type=float, default=0.20)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--max_chunks", type=int, default=64)
    ap.add_argument("--calib_batch_size", type=int, default=4)
    ap.add_argument("--out_dir", default="results/staticscale_multiseed")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip() != ""]
    os.makedirs(a.out_dir, exist_ok=True)

    model, tok = load_model(a.model, a.device)
    fp16, _, _ = wikitext2_ppl(model, tok, a.device, a.seq_len, a.max_chunks)
    print(f"[ms] fp16={fp16:.4f}", flush=True)
    rows = {"model": a.model, "seq_len": a.seq_len, "max_chunks": a.max_chunks,
            "fp_ratio": a.fp_ratio, "fp16": fp16, "per_seed": {}}

    def ev(cfg):
        plan = calibrate(model, tok, cfg, device=a.device, allow_synthetic=False,
                         batch_size=a.calib_batch_size)
        h = patch_model(model, plan, cfg)
        try:
            ppl, _, _ = wikitext2_ppl(model, tok, a.device, a.seq_len, a.max_chunks)
        finally:
            unpatch_model(h)
        return ppl

    for s in seeds:
        rows["per_seed"][s] = {}
        for v in VARIANTS:
            t0 = time.time()
            rows["per_seed"][s][v] = ev(make_cfg(s, v, a.fp_ratio))
            print(f"[ms] seed={s} {v:14s} ppl={rows['per_seed'][s][v]:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        sd = rows["per_seed"][s]["clean_sadnd"]; cm = rows["per_seed"][s]["cap_plus"]
        rows["per_seed"][s]["d_ss_vs_sadnd"] = rows["per_seed"][s]["staticscale"] - sd
        rows["per_seed"][s]["d_ss_vs_capplus"] = rows["per_seed"][s]["staticscale"] - cm
        print(f"[ms] seed={s} d(ss-sadnd)={rows['per_seed'][s]['d_ss_vs_sadnd']:+.4f} "
              f"d(ss-cap+)={rows['per_seed'][s]['d_ss_vs_capplus']:+.4f}", flush=True)
        json.dump(rows, open(os.path.join(a.out_dir, "multiseed_results.json"), "w"), indent=2)

    d1 = [rows["per_seed"][s]["d_ss_vs_sadnd"] for s in seeds]
    d2 = [rows["per_seed"][s]["d_ss_vs_capplus"] for s in seeds]
    rows["aggregate"] = {"staticscale_vs_sadnd": aggregate(d1),
                         "staticscale_vs_capplus": aggregate(d2), "n_seeds": len(seeds)}
    json.dump(rows, open(os.path.join(a.out_dir, "multiseed_results.json"), "w"), indent=2)

    with open(os.path.join(a.out_dir, "gt_multiseed_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "clean_sadnd", "cap_plus", "staticscale",
                    "delta_ss_vs_sadnd", "delta_ss_vs_capplus"])
        for s in seeds:
            e = rows["per_seed"][s]
            w.writerow([s, round(e["clean_sadnd"], 4), round(e["cap_plus"], 4),
                        round(e["staticscale"], 4), round(e["d_ss_vs_sadnd"], 4),
                        round(e["d_ss_vs_capplus"], 4)])
        for key, lbl in [("staticscale_vs_sadnd", "AGG_ss_vs_sadnd"),
                         ("staticscale_vs_capplus", "AGG_ss_vs_capplus")]:
            g = rows["aggregate"][key]
            w.writerow([lbl, "", "", "", f"mean={g['mean_delta']:.5f} std={g['std_delta']:.5f} "
                        f"clear={g['n_clearing']}/{len(seeds)} robust={g['robust_better']}", ""])

    a1 = rows["aggregate"]["staticscale_vs_sadnd"]; a2 = rows["aggregate"]["staticscale_vs_capplus"]
    print(f"[ms] AGG ss_vs_sadnd mean={a1['mean_delta']:+.5f} clear={a1['n_clearing']}/{len(seeds)} "
          f"robust={a1['robust_better']} | ss_vs_cap+ mean={a2['mean_delta']:+.5f} "
          f"clear={a2['n_clearing']}/{len(seeds)} robust={a2['robust_better']}", flush=True)
    print(f"[ms] wrote {a.out_dir}/multiseed_results.json + gt_multiseed_summary.csv", flush=True)


if __name__ == "__main__":
    main()
