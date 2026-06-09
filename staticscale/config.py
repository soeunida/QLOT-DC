"""StaticScale configuration (public API).

``StaticScaleConfig`` is the public name for the configuration dataclass; it is an
alias of the internal ``qlot_rms.config.QLotRmsConfig`` (kept for backward
compatibility during the transition). ``LayerRouting`` / ``RoutingPlan`` are the
per-layer static policy artifacts.
"""

from qlot_rms.config import QLotRmsConfig, LayerRouting, RoutingPlan

#: Public configuration class for StaticScale.
StaticScaleConfig = QLotRmsConfig

__all__ = ["StaticScaleConfig", "QLotRmsConfig", "LayerRouting", "RoutingPlan"]
