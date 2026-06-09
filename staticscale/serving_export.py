"""Static per-layer serving-artifact export (public API).

Exports the frozen StaticScale per-layer tensors (FP/INT indices, packing-aware
permutation, INT activation scales, packed weights) exactly as the reference
backend uses them. The reference backend is correctness-only; no speedup is claimed.

    python -m staticscale.serving_export --help
"""

from qlot_rms.serving_export import export_serving_artifacts, _cli

__all__ = ["export_serving_artifacts"]

if __name__ == "__main__":
    _cli()
