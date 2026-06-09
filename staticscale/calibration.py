"""StaticScale calibration (public API).

``calibrate(model, tokenizer, cfg, ...)`` runs the full static policy search
(routing -> budget -> mask refinement -> clip-gain tuning -> packing) and returns
a frozen ``RoutingPlan``.
"""

from qlot_rms.calibration import calibrate

__all__ = ["calibrate"]
