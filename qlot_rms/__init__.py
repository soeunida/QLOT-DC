"""Q-LOT-RMS: From Quantization Sensitivity to Static FP/INT Layouts for INT8
Transformer Inference.

A faithful, testable, modular reference implementation added as a *separate*
feature path. It does NOT import, edit, or remove any existing PQ*/SFPA script.
Q-LOT-RMS only changes model behavior when ``QLotRmsConfig.enable_qlot_rms`` is
True and the model has been patched via :func:`model_integration.patch_model`.

Modules
-------
config            QLotRmsConfig + per-layer routing metadata dataclasses
quant             channel quantiles, symmetric INT8 quant, simulated W8-G128
grouprms          GroupRMS (contiguous groups, last group may be smaller)
sadnd             SADND proxy-distortion routing + [FP, INT] permutation
capture           pre-affine LN2/RMSNorm activation capture hooks
data              WikiText-2 calibration chunks and random subsets
calibration       end-to-end calibration: routing, mu_g, activation scales
projection        packed FP/INT reference projection + backend interface
model_integration patch routed FFN behind the feature flag (reversible)

The default backend is ``torch_reference`` and is correct without any custom
kernels.  See ``docs/qlot_rms.md`` for scope, limitations, and how-to-run.
"""

from .config import QLotRmsConfig, LayerRouting, RoutingPlan

__all__ = ["QLotRmsConfig", "LayerRouting", "RoutingPlan"]

__version__ = "0.1.0"
