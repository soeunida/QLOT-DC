"""qlot_rms — LEGACY / INTERNAL implementation package for StaticScale.

.. deprecated::
   ``qlot_rms`` is the internal implementation of **StaticScale** and is kept only
   for backward compatibility. New code should import from the public
   :mod:`staticscale` package (e.g. ``from staticscale import StaticScaleConfig,
   calibrate, patch_model``). The public StaticScale modules re-export the symbols
   defined here under stable names. This package may be removed in a future release.

SADND-CAP / SADND-CAP-GT (legacy names): calibration-time adaptive FP/INT routing,
budget, mask refinement, and static groupwise clip-gain tuning. It routes the
Pre-LN ``LN2 -> FFN`` interface to a static FP16 / INT8(W8-G128) layout using:
  1. output-aware SADND routing
  2. global layer-wise FP budget allocation
  3. packing-aware INT permutation
  4. equal-budget accept-only selection

Deprecated correction methods (Q-LOT-DC / DC+ / OBC, GroupRMS, diagonal/bias/
low-rank/block corrections) were removed; see git tag
``backup-before-final-sadnd-cap-cleanup``. The default backend is
``torch_reference`` (fake-quantized, correctness-only); no speedup is claimed.

Modules
-------
config            QLotRmsConfig + LayerRouting / RoutingPlan
sadnd             proxy distortion + aggregation (routing primitives)
sadnd_cap         output-aware score, global FP budget, packing-aware permutation,
                  equal-budget accept-only selection
quant             channel quantiles, INT8 quant, simulated W8-G128
capture           pre-affine LN2/RMSNorm activation hooks
data              WikiText-2 calibration chunks
calibration       end-to-end SADND-CAP calibration
projection        packed FP/INT reference projection + backend interface
model_integration reversible patch of the routed FFN
serving_export    static per-layer serving artifact export
"""

from .config import QLotRmsConfig, LayerRouting, RoutingPlan

__all__ = ["QLotRmsConfig", "LayerRouting", "RoutingPlan"]

__version__ = "1.0.0"
