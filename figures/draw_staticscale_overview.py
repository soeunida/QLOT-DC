"""StaticScale overview figure (pipeline block diagram).

Renders the calibration-time static policy search and the static serving layout.
Saves PNG + PDF next to this script. Works as a .py script and interactively
(falls back to a ``figures/`` output dir when ``__file__`` is undefined).

No forbidden legacy names appear in the figure text. The serving block is a
packed FP/INT prototype; no backend-independent speedup is claimed.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

try:
    OUT_DIR = Path(__file__).resolve().parent
except NameError:
    OUT_DIR = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _box(ax, xy, w, h, text, fc, ec="#333333", fs=9):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.2, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def _arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                 lw=1.3, color="#444444"))


def main():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

    ax.text(6, 6.7, "StaticScale", ha="center", fontsize=16, fontweight="bold")
    ax.text(6, 6.32, "Calibration-Time FP/INT Routing and Static INT Scale Tuning",
            ha="center", fontsize=10, color="#555555")

    # Phase 1 label
    ax.text(3.0, 5.7, "Offline Calibration & Static Policy Search", ha="center",
            fontsize=11, fontweight="bold", color="#1f4e79")
    cal = "#dce9f6"
    _box(ax, (0.3, 4.55), 5.4, 0.7, "Output-aware SADND Routing\n(FP/INT split by INT8 proxy distortion × weight norm)", cal)
    _box(ax, (0.3, 3.65), 5.4, 0.7, "Cascade-aware FP Budget Allocation\n(+ marginal-gain; one fixed global FP budget)", cal)
    _box(ax, (0.3, 2.75), 5.4, 0.7, "Equal-Budget FP Mask Refinement\n(boundary swaps; k_fp preserved)", cal)
    _box(ax, (0.3, 1.85), 5.4, 0.7, "Static groupwise clip-gain tuning\n(tau_g per INT group; output gain eta)", cal)
    _box(ax, (0.3, 0.95), 5.4, 0.7, "Packing-aware static FP/INT layout\n(uniform W8-G128 groups)", cal)
    for y0, y1 in [(4.55, 4.35), (3.65, 3.45), (2.75, 2.55), (1.85, 1.65)]:
        _arrow(ax, (3.0, y0), (3.0, y1 + 0.20))

    # transition arrow
    _arrow(ax, (5.8, 3.1), (6.8, 3.1))
    ax.text(6.3, 3.35, "freeze", ha="center", fontsize=8, color="#666666")

    # Phase 2 label
    ax.text(9.4, 5.7, "Serving Phase", ha="center", fontsize=11, fontweight="bold", color="#7a4f01")
    srv = "#fdebd0"
    _box(ax, (6.9, 4.05), 4.8, 1.0, "Static Tuned FP/INT Layout\n[ FP block | INT block ]\ntuned scale multiplier tau_g per INT group", srv)
    _box(ax, (6.9, 2.75), 4.8, 0.9, "y_F @ W_F^T  +  quant(y_I) @ W_I(W8-G128)^T\n(static; no runtime top-k / sort / search)", srv)
    _box(ax, (6.9, 1.55), 4.8, 0.9, "Packed FP/INT prototype\n(reference backend; prototype diagnostic only)", "#f4f4f4")
    _arrow(ax, (9.3, 4.05), (9.3, 3.65 + 0.0))
    _arrow(ax, (9.3, 2.75), (9.3, 2.45))

    ax.text(6, 0.35, "Correctness/reference backend — not a StaticScale speedup claim.",
            ha="center", fontsize=8.5, style="italic", color="#777777")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"staticscale_overview.{ext}"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
else:
    main()
