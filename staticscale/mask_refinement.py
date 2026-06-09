"""Equal-budget FP mask refinement (public API).

Keeps each layer's FP budget (``k_fp``) fixed and refines *which* channels are FP
via greedy boundary swaps, accepting a swap only if a measured MLP-output proxy
improves. Fully static at inference.
"""

from qlot_rms.fp_mask_refinement import (
    build_boundary_swap_candidates, build_refine_context, build_refine_context_from_weights,
    evaluate_mask_proxy, greedy_refine_fp_mask, beam_refine_fp_mask,
    refine_layer_mask, refine_policy_masks, LayerRefineContext,
)

__all__ = [
    "build_boundary_swap_candidates", "build_refine_context",
    "build_refine_context_from_weights", "evaluate_mask_proxy",
    "greedy_refine_fp_mask", "beam_refine_fp_mask", "refine_layer_mask",
    "refine_policy_masks", "LayerRefineContext",
]
