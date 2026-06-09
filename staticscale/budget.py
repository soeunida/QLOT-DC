"""Cascade-aware & marginal-gain FP budget allocation (public API).

Allocates one fixed global FP budget across layers by local sensitivity plus
accumulated residual-stream (cascade) error, or greedily by per-channel marginal
gain. The total FP budget is preserved.
"""

from qlot_rms.cascade_budget import (
    compute_layer_quant_error, compute_cascade_error, compute_error_amplification,
    build_cascade_budget_scores, allocate_fp_budget_from_scores,
    compute_marginal_gain_table, allocate_by_marginal_gain, capture_layer_errors,
)
from qlot_rms.sadnd_cap import allocate_global_fp_budget

__all__ = [
    "compute_layer_quant_error", "compute_cascade_error", "compute_error_amplification",
    "build_cascade_budget_scores", "allocate_fp_budget_from_scores",
    "compute_marginal_gain_table", "allocate_by_marginal_gain", "capture_layer_errors",
    "allocate_global_fp_budget",
]
