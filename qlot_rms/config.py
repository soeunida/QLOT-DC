"""Configuration and per-layer routing metadata for Q-LOT-RMS.

Three dataclasses live here:

* :class:`QLotRmsConfig` -- all user-facing flags / hyper-parameters.
* :class:`LayerRouting`  -- the *frozen* per-layer artifact produced by
  calibration (mask, permutation, FP/INT indices, GroupRMS layout, mu_g, and
  per-channel activation scales).  Everything needed at inference time is here;
  nothing is recomputed from data at runtime.
* :class:`RoutingPlan`   -- a collection of all routed layers + the config.

All support round-trip save/load.  ``QLotRmsConfig`` serializes to JSON;
``LayerRouting`` carries tensors and serializes via torch (``.pt``), with a
parallel human-readable JSON summary written alongside.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Union, Dict, Any, Optional

import torch


# --------------------------------------------------------------------------- #
# User-facing configuration
# --------------------------------------------------------------------------- #
@dataclass
class QLotRmsConfig:
    """All Q-LOT-RMS knobs.  Defaults match the paper.

    Notes
    -----
    * ``qlot_scope="mlp_only"`` is the fully-implemented path.
    * ``qlot_scope="mlp_attn"`` is an explicit stub and raises
      ``NotImplementedError`` when integration is attempted (attention routing
      is intentionally *not* silently ignored).
    """

    # --- master switch ---
    enable_qlot_rms: bool = False

    # --- scope ---
    # "mlp_only"  -> route the Pre-LN LN2 -> FFN (gate_proj, up_proj) interface.
    # "mlp_attn"  -> additionally route attention input LN (STUB, not implemented).
    qlot_scope: str = "mlp_only"

    # --- routing / layout ---
    fp_ratio: float = 0.06          # rho_F: fraction of channels kept in FP
    grms_group_size: int = 128      # group size for INT-branch GroupRMS
    lambda_agg: float = 1.0         # delta_tilde = mean + lambda_agg * std
    p_proxy: float = 0.9995         # high quantile for the proxy INT8 scale
    p_act: float = 0.999            # quantile for frozen activation scales
    qmax: int = 127                 # INT8 symmetric max code

    # which decoder layers to route. "all", a list of ints, or a stride schedule
    # dict like {"start": 0, "stop": None, "step": 1}.
    routed_layers: Union[str, List[int], Dict[str, Any]] = "all"

    # --- calibration data ---
    calibration_samples: int = 128  # number of WikiText-2 chunks to build
    calibration_seq_len: int = 512  # tokens per chunk
    num_calib_subsets: int = 5      # number of random subsets for mean+std agg
    subset_size: int = 32           # sequences per subset

    # --- method selector ---
    # "qlot_rms" (default) -> SADND routing + (optional, gated) GroupRMS.
    # "qlot_dc"            -> SADND routing + Static Diagonal Compensation (no
    #                          token-dependent normalization). See use_static_diag_comp.
    method: str = "qlot_rms"

    # routing score used to rank channels for FP/INT assignment.
    # "sadnd" (default) | "magnitude" | "output_aware_sadnd" (STUB, not implemented).
    routing_score: str = "sadnd"

    # --- Q-LOT-DC: Static Diagonal Compensation (GroupRMS replacement) ---
    # Static, calibration-time, per-INT-channel scale alpha_c applied to the INT
    # activation, with inverse scaling folded into the INT weight columns
    # (W[:, int_indices] /= alpha_c) so the projection function is preserved
    # before quantization. No token-dependent normalization.
    use_static_diag_comp: bool = False
    diag_comp_mode: str = "none"            # "none" | "median_scale" | "smoothquant_like"
    diag_comp_alpha_min: float = 0.25
    diag_comp_alpha_max: float = 4.0
    diag_comp_beta: float = 0.5             # smoothquant_like exponent
    diag_comp_scope: str = "int_only"

    # --- Error-bounded FP budget (per-layer fp_ratio selection) ---
    fp_budget_mode: str = "fixed"           # "fixed" | "error_bounded"
    error_bound_metric: str = "activation_mse"  # "activation_mse" | "output_mse"
    error_bound: float = 0.01
    fp_ratio_candidates: List[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
    )

    # --- optional projection bias correction ---
    use_projection_bias_correction: bool = False

    # --- backend ---
    # "torch_reference" is the default correctness backend (no custom kernels).
    # "custom_packed" is a clean stub for a future CUDA/Triton branched kernel.
    backend: str = "torch_reference"

    # weight quant group size for the simulated W8-G128 path (matches paper G128)
    w8_group_size: int = 128

    # --- ablation toggles (used by eval to express the variant matrix) ---
    use_grms: bool = True           # apply INT-branch GroupRMS (False = routing only)
    use_mean_comp: bool = True      # fold mu_g into INT weight columns at packing

    # --- per-layer GroupRMS gating ---
    # When True (and use_grms True), GroupRMS is NOT applied globally; each routed
    # layer is gated by a calibration-time proxy (INT-branch output reconstruction
    # error with vs without GroupRMS). GroupRMS is enabled for a layer only if it
    # reduces that error by at least ``grms_gate_margin`` (relative). This avoids
    # the function-shift cost on layers where INT8 is already near-lossless.
    grms_gating: bool = False
    grms_gate_margin: float = 0.0       # relative error reduction required to enable
    grms_gate_max_tokens: int = 1024    # tokens used for the proxy (bounds cost)
    grms_gate_tolerance_ppl: float = 0.005  # documented PPL tolerance (for future
                                            # PPL-based gating; proxy uses error margin)
    # cap on #tokens stored (per layer, CPU) when estimating activation scales,
    # to bound calibration memory on large models. mu_g uses ALL tokens (streaming).
    act_scale_max_tokens: int = 16384

    # --- reference-backend-only performance flag ---
    # When True (and backend == "torch_reference"), the static W8-G128 weight is
    # fake-quantized ONCE at packing and cached, instead of being re-quantized on
    # every forward.  This is numerically identical to the per-forward path
    # (same fake-quant of a static weight) and is purely a speed optimization.
    # It is ignored by the custom_packed backend (which carries its own packed
    # weights).  Set False to force the original per-forward dequant path.
    cache_dequant_weight: bool = True

    # --- numerics / reproducibility ---
    eps: float = 1e-6               # GroupRMS / distortion denominator epsilon
    eps_scale: float = 1e-8         # floor for activation scales (eps_s)
    seed: int = 0                   # deterministic calibration seed

    # --- misc ---
    dataset: str = "wikitext2"

    VALID_SCOPES = ("mlp_only", "mlp_attn")
    VALID_BACKENDS = ("torch_reference", "custom_packed")

    def validate(self) -> "QLotRmsConfig":
        if self.qlot_scope not in self.VALID_SCOPES:
            raise ValueError(
                f"qlot_scope must be one of {self.VALID_SCOPES}, got {self.qlot_scope!r}"
            )
        if self.backend not in self.VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {self.VALID_BACKENDS}, got {self.backend!r}"
            )
        if not (0.0 <= self.fp_ratio < 1.0):
            raise ValueError(f"fp_ratio must be in [0, 1), got {self.fp_ratio}")
        if self.grms_group_size <= 0:
            raise ValueError("grms_group_size must be positive")
        if not (0.0 < self.p_proxy < 1.0) or not (0.0 < self.p_act < 1.0):
            raise ValueError("p_proxy and p_act must be in (0, 1)")
        if self.qmax <= 0:
            raise ValueError("qmax must be positive")
        if self.method not in ("qlot_rms", "qlot_dc"):
            raise ValueError(f"method must be qlot_rms|qlot_dc, got {self.method!r}")
        if self.routing_score not in ("sadnd", "magnitude", "output_aware_sadnd"):
            raise ValueError(f"invalid routing_score {self.routing_score!r}")
        if self.diag_comp_mode not in ("none", "median_scale", "smoothquant_like"):
            raise ValueError(f"invalid diag_comp_mode {self.diag_comp_mode!r}")
        if self.fp_budget_mode not in ("fixed", "error_bounded"):
            raise ValueError(f"invalid fp_budget_mode {self.fp_budget_mode!r}")
        if self.error_bound_metric not in ("activation_mse", "output_mse"):
            raise ValueError(f"invalid error_bound_metric {self.error_bound_metric!r}")
        if not (self.diag_comp_alpha_min > 0 and
                self.diag_comp_alpha_max >= self.diag_comp_alpha_min):
            raise ValueError("require 0 < diag_comp_alpha_min <= diag_comp_alpha_max")
        return self

    # --- serialization ---
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QLotRmsConfig":
        known = set(cls.__dataclass_fields__)  # noqa
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def load_json(cls, path: str) -> "QLotRmsConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# --------------------------------------------------------------------------- #
# Frozen per-layer routing metadata (produced by calibration)
# --------------------------------------------------------------------------- #
@dataclass
class LayerRouting:
    """Static, frozen routing artifact for one routed decoder layer.

    Everything an inference forward needs is precomputed here.  No runtime
    routing score, sort, or top-k is ever recomputed from the artifact.

    Index/orientation conventions
    ------------------------------
    * ``C`` is the FFN input channel count (LN2 hidden size).
    * ``fp_indices`` / ``int_indices`` index the *original* channel order.
    * ``perm`` = ``[fp_indices, int_indices]`` (FP first, INT second), so that
      after permutation INT channels are contiguous and GroupRMS groups are
      contiguous blocks of ``grms_group_size``.
    * ``mask[c] == True`` means channel c is routed to FP.
    * ``act_scales`` is indexed in *permuted-INT* order (length ``C - K_F``),
      i.e. it aligns with ``int_indices``.  It is shared by gate_proj & up_proj
      because both consume the same GroupRMS'd + affine INT activation.
    * Weight slices follow PyTorch ``nn.Linear`` orientation ``[out, in]``;
      input channels are the *columns* (dim=1).
    """

    layer_index: int
    num_channels: int                       # C
    k_fp: int                               # K_F = floor(fp_ratio * C)

    fp_indices: torch.Tensor                # long [K_F]   (original order)
    int_indices: torch.Tensor               # long [C-K_F] (original order)
    perm: torch.Tensor                      # long [C] = [fp_indices, int_indices]
    mask: torch.Tensor                      # bool [C], True = FP

    # SADND scores (for inspection / ablation), original channel order
    delta_tilde: torch.Tensor               # float [C]

    # GroupRMS layout over the contiguous INT block
    grms_group_size: int
    grms_num_groups: int
    grms_group_sizes: List[int]             # per-group sizes (last may be smaller)

    # mean RMS scale used for mean-scale compensation -- PER GROUP.
    #   mu_g : float tensor, length == grms_num_groups.  mu_g[g] is the mean (over
    #          calibration tokens) of the per-(token,group) RMS r_tg for group g.
    #          Per-group is required: once high-energy channels are routed to FP,
    #          the per-group RMS of the remaining INT channels varies sharply, so
    #          a single scalar mis-compensates each group.
    mu_g: torch.Tensor                      # float [grms_num_groups]

    # frozen per-channel INT activation scales (permuted-INT order), length C-K_F
    act_scales: torch.Tensor                # float [C-K_F]

    # mu_g expanded to INT channels (each channel carries its group's mu_g),
    # length C-K_F (permuted order).  This is what is folded into the INT weight
    # columns at packing.  Derived from mu_g; cached here for direct application.
    mu_g_channels: Optional[torch.Tensor] = None

    # which projections were routed (always gate_proj / up_proj for mlp_only)
    routed_projections: List[str] = field(
        default_factory=lambda: ["gate_proj", "up_proj"]
    )

    # bookkeeping
    norm_type: str = "rmsnorm"              # "rmsnorm" or "layernorm"
    mean_comp_applied: bool = False         # whether W[:, int] *= mu_g was baked in

    # --- per-layer GroupRMS gating decision (see QLotRmsConfig.grms_gating) ---
    grms_enabled: bool = True               # whether GroupRMS is applied for this layer
    grms_gate_reason: str = ""              # human-readable reason for the decision
    grms_proxy_err_ptq: Optional[float] = None   # INT-branch recon error w/o GroupRMS
    grms_proxy_err_grms: Optional[float] = None  # INT-branch recon error w/ GroupRMS
    grms_ppl_delta_if_measured: Optional[float] = None  # None unless a true PPL gate ran

    # --- Q-LOT-DC: Static Diagonal Compensation metadata ---
    diag_comp_applied: bool = False
    diag_alpha: Optional[torch.Tensor] = None   # float [k_int], packed-INT order
    # --- error-bounded FP budget (per-layer) ---
    selected_fp_ratio: Optional[float] = None
    fp_budget_errors: Optional[Dict[str, float]] = None  # {str(cand): error}
    # --- projection bias correction vectors (per routed projection) ---
    bias_corr_gate: Optional[torch.Tensor] = None   # float [O]
    bias_corr_up: Optional[torch.Tensor] = None     # float [O]

    def summary(self) -> Dict[str, Any]:
        """JSON-friendly summary (no big tensors)."""
        return {
            "layer_index": self.layer_index,
            "num_channels": self.num_channels,
            "k_fp": self.k_fp,
            "k_int": int(self.int_indices.numel()),
            "fp_ratio_effective": self.k_fp / max(1, self.num_channels),
            "grms_group_size": self.grms_group_size,
            "grms_num_groups": self.grms_num_groups,
            "grms_group_sizes": list(self.grms_group_sizes),
            "mu_g_groups": [float(x) for x in torch.as_tensor(self.mu_g).flatten().tolist()],
            "mu_g_mean": float(torch.as_tensor(self.mu_g).float().mean()),
            "mu_g_len": int(torch.as_tensor(self.mu_g).numel()),
            "routed_projections": list(self.routed_projections),
            "norm_type": self.norm_type,
            "mean_comp_applied": bool(self.mean_comp_applied),
            "grms_enabled": bool(self.grms_enabled),
            "grms_gate_reason": self.grms_gate_reason,
            "grms_proxy_err_ptq": self.grms_proxy_err_ptq,
            "grms_proxy_err_grms": self.grms_proxy_err_grms,
            "grms_ppl_delta_if_measured": self.grms_ppl_delta_if_measured,
            "diag_comp_applied": bool(self.diag_comp_applied),
            "diag_alpha_min": float(self.diag_alpha.min()) if self.diag_alpha is not None and self.diag_alpha.numel() else None,
            "diag_alpha_max": float(self.diag_alpha.max()) if self.diag_alpha is not None and self.diag_alpha.numel() else None,
            "selected_fp_ratio": self.selected_fp_ratio,
            "fp_budget_errors": self.fp_budget_errors,
            "bias_corr_applied": bool(self.bias_corr_gate is not None),
            "act_scales_min": float(self.act_scales.min()) if self.act_scales.numel() else None,
            "act_scales_max": float(self.act_scales.max()) if self.act_scales.numel() else None,
        }


# --------------------------------------------------------------------------- #
# Collection of all routed layers + the config used to produce them
# --------------------------------------------------------------------------- #
@dataclass
class RoutingPlan:
    """All per-layer routing artifacts plus the originating config."""

    config: QLotRmsConfig
    layers: Dict[int, LayerRouting]         # layer_index -> LayerRouting

    def save(self, out_dir: str) -> Dict[str, str]:
        """Save a ``.pt`` payload (tensors) + a human-readable JSON summary.

        Returns a dict of written paths.
        """
        os.makedirs(out_dir, exist_ok=True)
        pt_path = os.path.join(out_dir, "qlot_rms_routing.pt")
        json_path = os.path.join(out_dir, "qlot_rms_routing.json")
        cfg_path = os.path.join(out_dir, "qlot_rms_config.json")

        payload = {
            "config": self.config.to_dict(),
            "layers": {int(k): asdict(v) for k, v in self.layers.items()},
        }
        torch.save(payload, pt_path)

        self.config.save_json(cfg_path)
        summary = {
            "config": self.config.to_dict(),
            "layers": {int(k): v.summary() for k, v in self.layers.items()},
        }
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        return {"pt": pt_path, "json": json_path, "config": cfg_path}

    @classmethod
    def load(cls, pt_path: str) -> "RoutingPlan":
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        config = QLotRmsConfig.from_dict(payload["config"])
        layers: Dict[int, LayerRouting] = {}
        for k, v in payload["layers"].items():
            layers[int(k)] = LayerRouting(**v)
        return cls(config=config, layers=layers)
