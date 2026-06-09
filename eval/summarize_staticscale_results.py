"""Summarize StaticScale result directories into a compact CSV / printout.

Reads, if present, ``sadnd_cap_selection.json`` (equal-budget selection) and/or
``multiseed_results.json`` (multi-seed robustness) under a result dir and writes a
compact ``staticscale_summary.csv``. Reports only what was measured (no fabrication).

Example:
    python eval/summarize_staticscale_results.py --result_dir results/staticscale_qwen25_7b
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def summarize_selection(sel, out_rows):
    out_rows.append(["# selection", sel.get("model"), f"fp16={sel.get('fp16_ppl')}",
                     f"int8={sel.get('int8_ptq_ppl')}",
                     f"winner={sel.get('winner')}@{sel.get('winner_fp')}",
                     f"clear={sel.get('clear_improvement')}"])
    for fp, blk in (sel.get("per_fp") or {}).items():
        base = blk.get("baseline_sadnd_ppl")
        for name, rec in (blk.get("candidates") or {}).items():
            out_rows.append([f"fp{fp}", name, f"ppl={rec.get('ppl')}",
                             f"dSADND={rec.get('delta_vs_sadnd')}",
                             f"dBASE={rec.get('delta_vs_base')}",
                             f"accepted={rec.get('accepted')}"])


def summarize_multiseed(ms, out_rows):
    agg = ms.get("aggregate", {})
    out_rows.append(["# multiseed", ms.get("model"), f"fp_ratio={ms.get('fp_ratio')}",
                     f"chunks={ms.get('max_chunks')}", f"fp16={ms.get('fp16')}", ""])
    for key, g in agg.items():
        if isinstance(g, dict) and "mean_delta" in g:
            out_rows.append(["AGG", key, f"mean={g['mean_delta']:.5f}",
                             f"std={g['std_delta']:.5f}", f"clear={g['n_clearing']}",
                             f"robust={g['robust_better']}"])


def main():
    ap = argparse.ArgumentParser(description="Summarize StaticScale results")
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    sel = _load(os.path.join(a.result_dir, "sadnd_cap_selection.json"))
    ms = _load(os.path.join(a.result_dir, "multiseed_results.json"))
    if sel:
        summarize_selection(sel, rows)
    if ms:
        summarize_multiseed(ms, rows)
    if not rows:
        print(f"no selection/multiseed JSON found under {a.result_dir}")
        return
    out = a.out or os.path.join(a.result_dir, "staticscale_summary.csv")
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    for r in rows:
        print("  ".join(str(x) for x in r))
    print(f"# wrote {out}")


if __name__ == "__main__":
    main()
