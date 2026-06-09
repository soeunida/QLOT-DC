"""Packing-aware static FP/INT layout (public API).

Reorders INT channels so each contiguous W8-G128 group has a more uniform
activation scale; the FP block stays first. No inverse permutation at inference.
"""

from qlot_rms.sadnd_cap import build_packing_aware_int_permutation, build_layer_fp_int_perm
from qlot_rms.quant import group_sizes_for

__all__ = [
    "build_packing_aware_int_permutation", "build_layer_fp_int_perm", "group_sizes_for",
]
