"""Static groupwise clip-gain tuning (public API).

Tunes the INT branch statically at calibration: a group-wise INT activation clip
multiplier ``tau_g`` (folds into the frozen activation scales) and a layer/group
INT output gain ``eta`` (folds into the INT weight columns). FP budget unchanged;
nothing dynamic at inference (this is NOT GroupRMS).
"""

from qlot_rms.groupwise_clip_gain import (
    build_int_groups, GTLayerContext, build_gt_context,
    tune_clip_multiplier, fit_int_output_gain, tune_layer,
)

__all__ = [
    "build_int_groups", "GTLayerContext", "build_gt_context",
    "tune_clip_multiplier", "fit_int_output_gain", "tune_layer",
]
