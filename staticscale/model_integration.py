"""Reversible model integration (public API).

``patch_model`` swaps the routed Pre-LN ``LN2 -> FFN`` interface for the static
packed FP/INT projections; ``unpatch_model`` restores the original modules exactly.
"""

from qlot_rms.model_integration import (
    patch_model, unpatch_model, build_qlot_ffn, QLotRmsFFN, PatchHandle,
    find_decoder_layers, resolve_routed_layer_indices, get_ln2_modules,
)

#: Public aliases for the routed FFN module / builder.
StaticScaleFFN = QLotRmsFFN
build_staticscale_ffn = build_qlot_ffn

__all__ = [
    "patch_model", "unpatch_model", "build_staticscale_ffn", "StaticScaleFFN",
    "build_qlot_ffn", "QLotRmsFFN",
    "PatchHandle", "find_decoder_layers", "resolve_routed_layer_indices", "get_ln2_modules",
]
