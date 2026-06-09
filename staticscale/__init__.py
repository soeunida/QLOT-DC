"""StaticScale: calibration-time FP/INT routing and static INT scale tuning for
Transformer inference.

StaticScale keeps the FP budget fixed, improves *where* FP protection is assigned,
and tunes the remaining INT branch with static group-wise scale multipliers. It is
a calibration-time **static policy search** — every decision is frozen before
serving; inference does no runtime top-k / sort / search / RMS.

Components (internal names in parentheses):
  1. Output-aware SADND routing            (``staticscale.sadnd``)
  2. Cascade-aware & marginal-gain FP budget allocation  (``staticscale.budget``)
  3. Equal-budget FP mask refinement       (``staticscale.mask_refinement``)
  4. Static groupwise clip-gain tuning     (``staticscale.clip_gain``)
  5. Packing-aware static FP/INT layout    (``staticscale.packing``)

The current backend (``torch_reference``) is a **correctness/reference** backend
(fake-quantized, FP matmul). **No speedup is claimed**; any throughput numbers from
the older packed prototype are diagnostic only.

This is the public API. The implementation currently lives in the legacy/internal
``qlot_rms`` package; StaticScale re-exports it under stable public names. Both
import paths work during the transition.
"""

from staticscale.config import StaticScaleConfig, QLotRmsConfig, LayerRouting, RoutingPlan
from staticscale.calibration import calibrate
from staticscale.model_integration import patch_model, unpatch_model
from staticscale.selection import equal_budget_accept_only_select

from staticscale import (  # public submodules
    config, sadnd, budget, mask_refinement, clip_gain, calibration,
    projection, packing, model_integration, selection, utils,
)

__all__ = [
    "StaticScaleConfig", "QLotRmsConfig", "LayerRouting", "RoutingPlan",
    "calibrate", "patch_model", "unpatch_model", "equal_budget_accept_only_select",
    "config", "sadnd", "budget", "mask_refinement", "clip_gain", "calibration",
    "projection", "packing", "model_integration", "selection", "utils",
]

__version__ = "0.1.0"
