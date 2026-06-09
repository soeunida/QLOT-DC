"""Quantization / numeric utilities (public API).

Reference INT8 activation quant, W8-G128 weight fake-quant, per-channel quantiles.
"""

from qlot_rms.quant import (
    group_sizes_for, channel_quantile, quantize_activation_int8,
    compute_activation_scales, fake_quantize_weight_w8_g128, simulated_int8_matmul,
)

__all__ = [
    "group_sizes_for", "channel_quantile", "quantize_activation_int8",
    "compute_activation_scales", "fake_quantize_weight_w8_g128", "simulated_int8_matmul",
]
