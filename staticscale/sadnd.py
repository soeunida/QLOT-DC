"""Output-aware SADND routing (public API).

SADND = relative INT8 proxy distortion per channel; output-aware SADND weights it
by the gate/up weight-column norms. The highest-scoring channels are protected in
FP16; the rest go to INT8.
"""

from qlot_rms.sadnd import proxy_distortion_subset, aggregate_distortion
from qlot_rms.sadnd_cap import (
    compute_output_aware_sadnd_score, select_fp_int_channels,
)

__all__ = [
    "proxy_distortion_subset", "aggregate_distortion",
    "compute_output_aware_sadnd_score", "select_fp_int_channels",
]
