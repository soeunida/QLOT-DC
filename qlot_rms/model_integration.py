"""Integrate Q-LOT-RMS into a Llama-family model's forward path (reversible).

Scope
-----
Only ``qlot_scope="mlp_only"`` is implemented end-to-end.  It routes the Pre-LN
``LN2 -> FFN`` interface: ``post_attention_layernorm -> mlp(gate_proj, up_proj,
down_proj)``.  ``qlot_scope="mlp_attn"`` raises ``NotImplementedError`` (attention
routing is intentionally not silently ignored).

Reversible patching contract
-----------------------------
For each routed decoder layer ``l``:

1. The ORIGINAL ``post_attention_layernorm`` (LN2) and ``mlp`` are stored.
2. A :class:`QLotRmsFFN` is constructed *first*; it captures a reference to the
   original norm (for gamma/beta/eps) and recomputes the pre-affine RMSNorm
   internally from the unnormalized hidden state.
3. ONLY AFTER QLotRmsFFN has stored the original norm do we replace
   ``layer.post_attention_layernorm`` with ``nn.Identity()`` (so the FFN module
   receives the raw residual-stream hidden) and ``layer.mlp`` with the
   QLotRmsFFN.
4. :meth:`PatchHandle.unpatch` restores the original norm and original mlp
   exactly.

Supported model families: Llama / Mistral / Qwen2-style (mlp has gate_proj,
up_proj, down_proj and a post_attention_layernorm).  Other families raise a
clear error (documented limitation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .capture import pre_affine_normalize
from .quant import quantize_activation_int8
from .config import QLotRmsConfig, LayerRouting, RoutingPlan
from .projection import (
    PackedProjection,
    compute_branch_inputs,
    branch_inputs_from_slices,
    get_backend,
)


# --------------------------------------------------------------------------- #
# Model introspection
# --------------------------------------------------------------------------- #
def _getattr_path(obj, path: str):
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def find_decoder_layers(model: nn.Module) -> List[nn.Module]:
    """Return the list of decoder layers for a Llama-family model."""
    for path in ("model.layers", "layers", "model.decoder.layers"):
        layers = _getattr_path(model, path)
        if layers is not None:
            return list(layers)
    raise ValueError(
        "could not locate decoder layers (tried model.layers / layers / "
        "model.decoder.layers); this model family may be unsupported."
    )


def _is_supported_mlp(mlp: nn.Module) -> bool:
    return all(hasattr(mlp, p) for p in ("gate_proj", "up_proj", "down_proj"))


def resolve_routed_layer_indices(model: nn.Module, cfg: QLotRmsConfig) -> List[int]:
    """Resolve ``cfg.routed_layers`` ("all" / list / schedule dict) to indices."""
    n = len(find_decoder_layers(model))
    spec = cfg.routed_layers
    if spec == "all":
        return list(range(n))
    if isinstance(spec, list):
        return [i for i in spec if 0 <= i < n]
    if isinstance(spec, dict):
        start = spec.get("start", 0) or 0
        stop = spec.get("stop", None)
        stop = n if stop is None else min(stop, n)
        step = spec.get("step", 1) or 1
        return list(range(start, stop, step))
    raise ValueError(f"invalid routed_layers spec: {spec!r}")


def get_ln2_modules(model: nn.Module, indices: List[int]) -> Dict[int, nn.Module]:
    """Map routed layer index -> its ``post_attention_layernorm`` module."""
    layers = find_decoder_layers(model)
    out: Dict[int, nn.Module] = {}
    for i in indices:
        layer = layers[i]
        if not hasattr(layer, "post_attention_layernorm"):
            raise ValueError(
                f"layer {i} has no post_attention_layernorm; unsupported family."
            )
        out[i] = layer.post_attention_layernorm
    return out


# --------------------------------------------------------------------------- #
# The routed FFN module
# --------------------------------------------------------------------------- #
class QLotRmsFFN(nn.Module):
    """Drop-in replacement for a routed layer's MLP.

    Receives the *unnormalized* residual-stream hidden state (because LN2 has
    been replaced by Identity), recomputes the pre-affine RMSNorm/LayerNorm
    internally, runs the packed FP/INT gate & up projections (sharing the branch
    inputs), applies the activation, and runs the unrouted down_proj.
    """

    def __init__(
        self,
        original_norm: nn.Module,
        packed_gate: PackedProjection,
        packed_up: PackedProjection,
        down_proj: nn.Module,
        act_fn,
        routing: LayerRouting,
        cfg: QLotRmsConfig,
    ):
        super().__init__()
        # store the ORIGINAL norm (used only to recompute pre-affine u + affine)
        self.original_norm = original_norm
        self.packed_gate = packed_gate
        self.packed_up = packed_up
        self.down_proj = down_proj
        self.act_fn = act_fn
        self.routing = routing
        self.cfg = cfg

        # cache affine params / routing tensors for branch-input computation
        gamma = getattr(original_norm, "weight")
        beta = getattr(original_norm, "bias", None)
        self._gamma = gamma
        self._beta = beta
        self._fp_idx = routing.fp_indices
        self._int_idx = routing.int_indices

        # --- Q-LOT-OBC block-output correction (applied after down_proj) ---
        dev = down_proj.weight.device
        self._block_mode = "none"
        self._block_bias = self._block_a = self._block_b = None
        self._block_A = self._block_B = None
        if getattr(routing, "block_corr_enabled", False):
            m = routing.block_corr_mode
            self._block_mode = m
            if m == "bias" and routing.block_bias is not None:
                self._block_bias = routing.block_bias.to(dev).float()
            elif m == "affine" and routing.block_affine_a is not None:
                self._block_a = routing.block_affine_a.to(dev).float()
                self._block_b = routing.block_affine_b.to(dev).float()
            elif m == "lowrank" and routing.block_lowrank_A is not None:
                self._block_A = routing.block_lowrank_A.to(dev).float()
                self._block_B = routing.block_lowrank_B.to(dev).float()
            else:
                self._block_mode = "none"  # enabled but tensors missing -> no-op

    def _apply_block_correction(self, out, hidden_states):
        """Static block-output correction (after down_proj). No-op if disabled."""
        m = self._block_mode
        if m == "none":
            return out
        o = out.float()
        if m == "bias":
            o = o + self._block_bias
        elif m == "affine":
            o = o * self._block_a + self._block_b
        elif m == "lowrank":
            o = o + torch.matmul(torch.matmul(hidden_states.float(), self._block_A), self._block_B)
        return o.to(out.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states is the raw residual-stream input (LN2 is now Identity).
        u = pre_affine_normalize(hidden_states, self.original_norm)  # fp32, orig order
        # gate & up share the same routing + affine, so reuse the gate
        # projection's PRECOMPUTED device-resident slices (no per-forward
        # index_select on gamma/beta, no host->device transfers).
        pg = self.packed_gate
        y_F, y_I = branch_inputs_from_slices(
            u,
            pg.fp_indices,
            pg.int_indices,
            pg.gamma_F,
            pg.gamma_I,
            pg.beta_F,
            pg.beta_I,
            self.routing.grms_group_size,
            self.cfg.eps,
            use_grms=pg.use_grms,   # per-layer decision (single source of truth)
            int_alpha=pg.diag_alpha,  # Q-LOT-DC static scale (None if unused)
        )
        pu = self.packed_up
        if pg.W_I_dq is not None and pu.W_I_dq is not None:
            # shared cached path: quantize y_I and cast y_F ONCE for gate & up
            # (both consume the identical y_I and act_scales). Numerically
            # identical to two separate forward_from_branches calls.
            y_F16 = y_F.to(torch.float16)
            yq = quantize_activation_int8(y_I, pg.act_scales, qmax=pg.qmax).float()
            gate = pg.matmul_shared(y_F16, yq)
            up = pu.matmul_shared(y_F16, yq)
        else:
            gate = pg.forward_from_branches(y_F, y_I)
            up = pu.forward_from_branches(y_F, y_I)
        # Q-LOT-DC+ optional low-rank residual correction (uses unquantized y_I)
        cg = pg.lowrank_correction(y_I)
        if cg is not None:
            gate = (gate.float() + cg.float()).to(torch.float16)
        cu = pu.lowrank_correction(y_I)
        if cu is not None:
            up = (up.float() + cu.float()).to(torch.float16)
        h = self.act_fn(gate) * up
        out = self.down_proj(h.to(self.down_proj.weight.dtype))
        return self._apply_block_correction(out, hidden_states)


# --------------------------------------------------------------------------- #
# Patch / unpatch
# --------------------------------------------------------------------------- #
@dataclass
class PatchHandle:
    """Holds originals so the model can be restored exactly."""

    model: nn.Module
    originals: Dict[int, Dict[str, nn.Module]]   # idx -> {"norm": ..., "mlp": ...}

    def unpatch(self) -> None:
        layers = find_decoder_layers(self.model)
        for idx, orig in self.originals.items():
            layer = layers[idx]
            layer.post_attention_layernorm = orig["norm"]
            layer.mlp = orig["mlp"]
        self.originals = {}


def patch_model(
    model: nn.Module, plan: RoutingPlan, cfg: Optional[QLotRmsConfig] = None
) -> PatchHandle:
    """Patch routed layers of ``model`` in-place; return a reversible handle.

    Raises for ``qlot_scope='mlp_attn'`` (explicit stub) and for unsupported
    model families / disabled flag.
    """
    cfg = cfg or plan.config
    cfg.validate()

    if not cfg.enable_qlot_rms:
        raise ValueError(
            "patch_model called but cfg.enable_qlot_rms is False; set it True "
            "to opt into the Q-LOT-RMS feature path."
        )
    if cfg.qlot_scope == "mlp_attn":
        raise NotImplementedError(
            "qlot_scope='mlp_attn' is not implemented. Attention routing is an "
            "explicit stub; only 'mlp_only' is supported end-to-end."
        )
    if cfg.qlot_scope != "mlp_only":
        raise ValueError(f"unsupported qlot_scope {cfg.qlot_scope!r}")

    backend = get_backend(cfg.backend)
    layers = find_decoder_layers(model)
    originals: Dict[int, Dict[str, nn.Module]] = {}

    for idx, routing in plan.layers.items():
        layer = layers[idx]
        norm = layer.post_attention_layernorm
        mlp = layer.mlp
        if not _is_supported_mlp(mlp):
            raise ValueError(
                f"layer {idx} mlp lacks gate_proj/up_proj/down_proj; unsupported."
            )

        ffn = build_qlot_ffn(layer, routing, cfg, backend)
        originals[idx] = {"norm": norm, "mlp": mlp}
        layer.post_attention_layernorm = nn.Identity()
        layer.mlp = ffn

    return PatchHandle(model=model, originals=originals)


def build_qlot_ffn(layer, routing, cfg, backend=None):
    """Construct a QLotRmsFFN for one decoder layer (does NOT mutate the model).

    Used by patch_model and by the block-correction fitter (which needs to run
    the quantized MLP output y_q on captured inputs without patching).
    """
    if backend is None:
        backend = get_backend(cfg.backend)
    norm = layer.post_attention_layernorm
    mlp = layer.mlp
    if not _is_supported_mlp(mlp):
        raise ValueError("mlp lacks gate_proj/up_proj/down_proj; unsupported.")
    gamma = norm.weight
    beta = getattr(norm, "bias", None)
    packed_gate = PackedProjection.from_linear(
        mlp.gate_proj, routing, gamma, beta, cfg, backend=backend,
        bias_corr=getattr(routing, "bias_corr_gate", None),
        lowrank_A=getattr(routing, "lowrank_gate_A", None),
        lowrank_B=getattr(routing, "lowrank_gate_B", None),
    )
    packed_up = PackedProjection.from_linear(
        mlp.up_proj, routing, gamma, beta, cfg, backend=backend,
        bias_corr=getattr(routing, "bias_corr_up", None),
        lowrank_A=getattr(routing, "lowrank_up_A", None),
        lowrank_B=getattr(routing, "lowrank_up_B", None),
    )
    return QLotRmsFFN(
        original_norm=norm, packed_gate=packed_gate, packed_up=packed_up,
        down_proj=mlp.down_proj, act_fn=getattr(mlp, "act_fn"),
        routing=routing, cfg=cfg,
    )


def unpatch_model(handle: PatchHandle) -> None:
    handle.unpatch()
