"""Configuration and per-layer routing metadata for SADND-CAP.

SADND-CAP = Calibration-time Adaptive FP/INT routing And Packing:
  1. Output-aware SADND routing
  2. Global layer-wise FP budget allocation
  3. Packing-aware INT permutation
  4. Equal-budget accept-only selection

This is the only active method. Deprecated correction methods (Q-LOT-DC,
Q-LOT-DC+, Q-LOT-OBC, GroupRMS, diagonal/bias/low-rank/block corrections) have
been removed; see git tag ``backup-before-final-sadnd-cap-cleanup`` for history.
No speedup is claimed; ``torch_reference`` is correctness-only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Union, Dict, Any, Optional

import torch


@dataclass
class QLotRmsConfig:
    """SADND-CAP configuration. Defaults match the paper-level settings."""

    # --- master switch / scope ---
    enable_qlot_rms: bool = False          # master flag (kept name for the patcher)
    method: str = "sadnd_cap"              # "sadnd_cap" (only active method)
    qlot_scope: str = "mlp_only"           # only mlp_only is implemented

    # --- routing ---
    # "sadnd"              : relative INT8 proxy distortion
    # "output_aware_sadnd" : distortion x (||W_gate[:,c]|| + ||W_up[:,c]||)
    # "magnitude"          : mean |activation| (baseline)
    routing_score: str = "sadnd"
    fp_ratio: float = 0.06                  # FP channel fraction (per layer, or global avg)

    # --- FP budget allocation ---
    # "fixed"  : every layer uses fp_ratio.
    # "global" : same TOTAL FP budget (fp_ratio * total channels) reallocated across
    #            layers by sensitivity (same_global_fp_budget keeps the total fixed).
    fp_budget_mode: str = "fixed"
    same_global_fp_budget: bool = True
    fp_ratio_candidates: List[float] = field(
        default_factory=lambda: [0.0, 0.04, 0.06, 0.10, 0.20]
    )

    # --- packing-aware INT permutation ---
    # "original"       : keep INT channels in original order
    # "scale_sorted"   : sort INT channels by activation scale (flattens W8-G128 groups)
    # "scale_clustered": coarse scale buckets, contiguous
    # "packing_aware"  : default packing strategy (currently = scale_sorted)
    int_permutation_mode: str = "packing_aware"

    # --- equal-budget accept-only selection ---
    accept_only_margin: float = 0.001       # candidate must beat SADND@same-fp by this

    # --- calibration data ---
    calibration_samples: int = 128
    calibration_seq_len: int = 512
    num_calib_subsets: int = 5
    subset_size: int = 32

    # --- quant / numerics ---
    p_proxy: float = 0.9995
    p_act: float = 0.999
    qmax: int = 127
    w8_group_size: int = 128
    backend: str = "torch_reference"        # "torch_reference" | "custom_packed" (stub)
    cache_dequant_weight: bool = True       # reference-only: cache fake-quant weight
    eps: float = 1e-6
    eps_scale: float = 1e-8
    act_scale_max_tokens: int = 16384       # per-layer token cap for scale estimation
    seed: int = 0
    dataset: str = "wikitext2"
    routed_layers: Union[str, List[int], Dict[str, Any]] = "all"

    VALID_SCOPES = ("mlp_only",)
    VALID_BACKENDS = ("torch_reference", "custom_packed")
    VALID_ROUTING = ("sadnd", "output_aware_sadnd", "magnitude")
    VALID_BUDGET = ("fixed", "global")
    VALID_PERM = ("original", "scale_sorted", "scale_clustered", "packing_aware")

    def validate(self) -> "QLotRmsConfig":
        if self.qlot_scope not in self.VALID_SCOPES:
            raise ValueError(f"qlot_scope must be 'mlp_only', got {self.qlot_scope!r}")
        if self.backend not in self.VALID_BACKENDS:
            raise ValueError(f"backend must be one of {self.VALID_BACKENDS}")
        if self.routing_score not in self.VALID_ROUTING:
            raise ValueError(f"invalid routing_score {self.routing_score!r}")
        if self.fp_budget_mode not in self.VALID_BUDGET:
            raise ValueError(f"invalid fp_budget_mode {self.fp_budget_mode!r}")
        if self.int_permutation_mode not in self.VALID_PERM:
            raise ValueError(f"invalid int_permutation_mode {self.int_permutation_mode!r}")
        if not (0.0 <= self.fp_ratio < 1.0):
            raise ValueError(f"fp_ratio must be in [0,1), got {self.fp_ratio}")
        if self.qmax <= 0 or self.w8_group_size <= 0:
            raise ValueError("qmax and w8_group_size must be positive")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QLotRmsConfig":
        # tolerant: ignores keys from deprecated methods (DC/OBC/grms/etc.)
        known = set(cls.__dataclass_fields__)  # noqa
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def load_json(cls, path: str) -> "QLotRmsConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class LayerRouting:
    """Static, frozen SADND-CAP routing artifact for one routed layer.

    Conventions
    -----------
    * ``C`` = FFN input channel count.
    * ``fp_indices`` (original order) | ``int_indices`` (PACKING-AWARE order).
    * ``perm = [fp_indices, int_indices]`` -> after permutation INT channels are
      contiguous and ordered for W8-G128 packing.
    * ``mask[c] == True`` => channel c is FP.
    * ``act_scales`` aligns with ``int_indices`` (per INT channel, permuted order).
    * Weight slices follow nn.Linear ``[out, in]``; input channels are columns.
    """

    layer_index: int
    num_channels: int
    k_fp: int
    fp_indices: torch.Tensor                 # long [K_F]
    int_indices: torch.Tensor                # long [C-K_F], packing-aware order
    perm: torch.Tensor                       # long [C] = [fp_indices, int_indices]
    mask: torch.Tensor                       # bool [C], True = FP
    delta_tilde: torch.Tensor                # float [C], SADND distortion
    act_scales: torch.Tensor                 # float [C-K_F], per INT channel
    w8_group_size: int = 128
    routing_score: str = "sadnd"
    int_permutation_mode: str = "packing_aware"
    selected_fp_ratio: Optional[float] = None
    norm_type: str = "rmsnorm"
    routed_projections: List[str] = field(
        default_factory=lambda: ["gate_proj", "up_proj"]
    )

    def summary(self) -> Dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "num_channels": self.num_channels,
            "k_fp": self.k_fp,
            "k_int": int(self.int_indices.numel()),
            "fp_ratio_effective": self.k_fp / max(1, self.num_channels),
            "selected_fp_ratio": self.selected_fp_ratio,
            "routing_score": self.routing_score,
            "int_permutation_mode": self.int_permutation_mode,
            "w8_group_size": self.w8_group_size,
            "norm_type": self.norm_type,
            "act_scales_min": float(self.act_scales.min()) if self.act_scales.numel() else None,
            "act_scales_max": float(self.act_scales.max()) if self.act_scales.numel() else None,
        }


@dataclass
class RoutingPlan:
    config: QLotRmsConfig
    layers: Dict[int, LayerRouting]

    def save(self, out_dir: str) -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        pt_path = os.path.join(out_dir, "sadnd_cap_routing.pt")
        json_path = os.path.join(out_dir, "sadnd_cap_routing.json")
        cfg_path = os.path.join(out_dir, "sadnd_cap_config.json")
        torch.save({"config": self.config.to_dict(),
                    "layers": {int(k): asdict(v) for k, v in self.layers.items()}}, pt_path)
        self.config.save_json(cfg_path)
        with open(json_path, "w") as f:
            json.dump({"config": self.config.to_dict(),
                       "layers": {int(k): v.summary() for k, v in self.layers.items()}},
                      f, indent=2)
        return {"pt": pt_path, "json": json_path, "config": cfg_path}

    @classmethod
    def load(cls, pt_path: str) -> "RoutingPlan":
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        cfg = QLotRmsConfig.from_dict(payload["config"])
        layers = {int(k): LayerRouting(**v) for k, v in payload["layers"].items()}
        return cls(config=cfg, layers=layers)
