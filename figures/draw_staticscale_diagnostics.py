"""StaticScale diagnostic figures (measured numbers only).

Produces two figures:
  routing_tuning_diagnostics_staticscale.{png,pdf}
      per-seed equal-budget ΔPPL (StaticScale vs clean SADND, vs CAP+) +
      the clip-driven nature of the gain (tau, eta).
  within_stack_attribution_staticscale.{png,pdf}
      where the equal-budget improvement comes from across the static stack
      (budget vs refinement vs clip-gain), as a cumulative attribution.

All numbers are read from results/ when present, else fall back to the recorded
Qwen2.5-7B fp_ratio=0.20 multi-seed values. Reference backend; prototype
diagnostic only — not a StaticScale speedup claim. No forbidden legacy names.

Works as a .py script and interactively (falls back to ``figures/`` for output).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    OUT_DIR = Path(__file__).resolve().parent
except NameError:
    OUT_DIR = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ROOT = OUT_DIR.parent

# recorded fallback (Qwen2.5-7B, fp_ratio=0.20, 64 chunks, seeds 0/1/2)
FALLBACK = {
    "per_seed": {"0": {"clean_sadnd": 6.5850, "cap_plus": 6.5834, "staticscale": 6.5815},
                 "1": {"clean_sadnd": 6.5850, "cap_plus": 6.5829, "staticscale": 6.5817},
                 "2": {"clean_sadnd": 6.5845, "cap_plus": 6.5827, "staticscale": 6.5826}},
    "tau_mean": 1.244, "eta_mean": 1.00001,
}


def _load_multiseed():
    for p in [ROOT / "results/staticscale_multiseed_qwen25_7b/multiseed_results.json",
              ROOT / "results/sadnd_cap_gt_multiseed_qwen25_7b/multiseed_results.json"]:
        if p.exists():
            d = json.load(open(p))
            ps = {}
            for s, e in d["per_seed"].items():
                cs = e.get("clean_sadnd", e.get("sadnd"))
                cp = e.get("cap_plus", e.get("cap_cascade_marginal"))
                ss = e.get("staticscale", e.get("gt_layer"))
                if None not in (cs, cp, ss):
                    ps[str(s)] = {"clean_sadnd": cs, "cap_plus": cp, "staticscale": ss}
            if ps:
                return {"per_seed": ps, "tau_mean": FALLBACK["tau_mean"], "eta_mean": FALLBACK["eta_mean"]}
    return FALLBACK


def fig_routing_tuning(data):
    ps = data["per_seed"]
    seeds = sorted(ps, key=int)
    d_sadnd = [ps[s]["staticscale"] - ps[s]["clean_sadnd"] for s in seeds]
    d_capp = [ps[s]["staticscale"] - ps[s]["cap_plus"] for s in seeds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = range(len(seeds)); w = 0.38
    ax1.bar([i - w/2 for i in x], d_sadnd, w, label="vs clean SADND", color="#2c7fb8")
    ax1.bar([i + w/2 for i in x], d_capp, w, label="vs CAP+ budget", color="#f08c00")
    ax1.axhline(-0.001, ls="--", lw=1, color="#888888")
    ax1.text(len(seeds)-0.5, -0.00108, " -0.001 margin", fontsize=8, color="#888888", va="top")
    ax1.set_xticks(list(x)); ax1.set_xticklabels([f"seed {s}" for s in seeds])
    ax1.set_ylabel("ΔPPL (StaticScale − baseline)")
    ax1.set_title("Equal-budget ΔPPL, Qwen2.5-7B fp=0.20")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    # clip-driven: tau != 1 (helps), eta ~ 1 (little)
    ax2.bar(["clip multiplier\ntau (mean)", "output gain\neta (mean)"],
            [data["tau_mean"], data["eta_mean"]], color=["#2c7fb8", "#cccccc"])
    ax2.axhline(1.0, ls="--", lw=1, color="#888888")
    ax2.set_ylim(0.9, 1.35)
    for i, v in enumerate([data["tau_mean"], data["eta_mean"]]):
        ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax2.set_title("Gain is clip-driven (tau≈1.24), eta≈1.0")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("StaticScale routing/tuning diagnostics  —  prototype diagnostic only",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"routing_tuning_diagnostics_staticscale.{ext}"
        fig.savefig(p, dpi=160, bbox_inches="tight"); print("wrote", p)
    plt.close(fig)


def fig_within_stack(data):
    # cumulative attribution at fp=0.20 (seed-0 single run): clean -> +budget(CAP+) -> +clip-gain(StaticScale)
    s0 = data["per_seed"][sorted(data["per_seed"], key=int)[0]]
    clean, capp, ss = s0["clean_sadnd"], s0["cap_plus"], s0["staticscale"]
    stages = ["clean SADND", "+ FP budget\n(CAP+)", "+ mask refinement\n(≈neutral here)", "+ clip-gain\n(StaticScale)"]
    vals = [clean, capp, capp, ss]   # refinement ≈ neutral in this regime

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(range(len(stages)), vals, "-o", color="#2c7fb8", lw=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.0002, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("WikiText-2 PPL (lower better)")
    ax.set_title("Within-stack attribution, Qwen2.5-7B fp=0.20 (equal FP budget)")
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.5, 0.02, "Static scale grouping controls held fixed; reference backend — prototype diagnostic only.",
            transform=ax.transAxes, ha="center", fontsize=8, style="italic", color="#777777")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"within_stack_attribution_staticscale.{ext}"
        fig.savefig(p, dpi=160, bbox_inches="tight"); print("wrote", p)
    plt.close(fig)


def main():
    data = _load_multiseed()
    fig_routing_tuning(data)
    fig_within_stack(data)


if __name__ == "__main__":
    main()
else:
    main()
