"""Packed FP/INT projection + backends (public API).

The reference backend (``torch_reference``) is correctness-only (fake-quant + FP
matmul). ``custom_packed`` is a stub (no kernel). No speedup is claimed.
"""

from qlot_rms.projection import (
    PackedProjection, branch_inputs_from_slices, get_backend,
    ProjectionBackend, TorchReferenceBackend, CustomPackedBackend,
)

__all__ = [
    "PackedProjection", "branch_inputs_from_slices", "get_backend",
    "ProjectionBackend", "TorchReferenceBackend", "CustomPackedBackend",
]
