"""Integrate SADND-CAP into a Llama-family model's forward path (reversible).

Only ``qlot_scope="mlp_only"`` is supported: route the Pre-LN ``LN2 -> FFN``
interface (post_attention_layernorm -> gate_proj/up_proj). The routed FFN runs
the static packed FP/INT projections (no correction modules) and the unrouted
down_proj. Reversible: patch_model / unpatch_model restore the originals exactly.

Supported families: Llama / Mistral / Qwen2-style (mlp has gate/up/down and a
post_attention_layernorm). No runtime top-k/sort/dynamic routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .capture import pre_affine_normalize
from .quant import quantize_activation_int8
from .config import QLotRmsConfig, LayerRouting, RoutingPlan
from .projection import PackedProjection, branch_inputs_from_slices, get_backend


def _getattr_path(obj, path):
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def find_decoder_layers(model: nn.Module) -> List[nn.Module]:
    for path in ("model.layers", "layers", "model.decoder.layers"):
        layers = _getattr_path(model, path)
        if layers is not None:
            return list(layers)
    raise ValueError("could not locate decoder layers (unsupported model family).")


def _is_supported_mlp(mlp):
    return all(hasattr(mlp, p) for p in ("gate_proj", "up_proj", "down_proj"))


def resolve_routed_layer_indices(model, cfg) -> List[int]:
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


def get_ln2_modules(model, indices) -> Dict[int, nn.Module]:
    layers = find_decoder_layers(model)
    out = {}
    for i in indices:
        if not hasattr(layers[i], "post_attention_layernorm"):
            raise ValueError(f"layer {i} has no post_attention_layernorm; unsupported.")
        out[i] = layers[i].post_attention_layernorm
    return out


class QLotRmsFFN(nn.Module):
    """Drop-in routed MLP: pre-affine norm -> packed FP/INT gate&up -> act -> down."""

    def __init__(self, original_norm, packed_gate, packed_up, down_proj, act_fn,
                 routing: LayerRouting, cfg: QLotRmsConfig):
        super().__init__()
        self.original_norm = original_norm
        self.packed_gate = packed_gate
        self.packed_up = packed_up
        self.down_proj = down_proj
        self.act_fn = act_fn
        self.routing = routing
        self.cfg = cfg

    def forward(self, hidden_states):
        # LN2 was replaced by Identity; recompute pre-affine u internally.
        u = pre_affine_normalize(hidden_states, self.original_norm)   # fp32, orig order
        pg, pu = self.packed_gate, self.packed_up
        y_F, y_I = branch_inputs_from_slices(
            u, pg.fp_indices, pg.int_indices, pg.gamma_F, pg.gamma_I, pg.beta_F, pg.beta_I)
        if pg.W_I_dq is not None and pu.W_I_dq is not None:
            # shared cached path: quantize y_I + cast y_F ONCE for gate & up
            y_F16 = y_F.to(torch.float16)
            yq = quantize_activation_int8(y_I, pg.act_scales, qmax=pg.qmax).float()
            gate = pg.matmul_shared(y_F16, yq)
            up = pu.matmul_shared(y_F16, yq)
        else:
            gate = pg.forward_from_branches(y_F, y_I)
            up = pu.forward_from_branches(y_F, y_I)
        h = self.act_fn(gate) * up
        return self.down_proj(h.to(self.down_proj.weight.dtype))


def build_qlot_ffn(layer, routing, cfg, backend=None):
    if backend is None:
        backend = get_backend(cfg.backend)
    norm = layer.post_attention_layernorm
    mlp = layer.mlp
    if not _is_supported_mlp(mlp):
        raise ValueError("mlp lacks gate_proj/up_proj/down_proj; unsupported.")
    gamma = norm.weight
    beta = getattr(norm, "bias", None)
    packed_gate = PackedProjection.from_linear(mlp.gate_proj, routing, gamma, beta, cfg, backend)
    packed_up = PackedProjection.from_linear(mlp.up_proj, routing, gamma, beta, cfg, backend)
    return QLotRmsFFN(norm, packed_gate, packed_up, mlp.down_proj,
                      getattr(mlp, "act_fn"), routing, cfg)


@dataclass
class PatchHandle:
    model: nn.Module
    originals: Dict[int, Dict[str, nn.Module]]

    def unpatch(self) -> None:
        layers = find_decoder_layers(self.model)
        for idx, orig in self.originals.items():
            layers[idx].post_attention_layernorm = orig["norm"]
            layers[idx].mlp = orig["mlp"]
        self.originals = {}


def patch_model(model, plan: RoutingPlan, cfg: Optional[QLotRmsConfig] = None) -> PatchHandle:
    cfg = cfg or plan.config
    cfg.validate()
    if not cfg.enable_qlot_rms:
        raise ValueError("patch_model called but cfg.enable_qlot_rms is False.")
    if cfg.qlot_scope != "mlp_only":
        raise NotImplementedError(f"qlot_scope={cfg.qlot_scope!r} not implemented (mlp_only only).")
    backend = get_backend(cfg.backend)
    layers = find_decoder_layers(model)
    originals: Dict[int, Dict[str, nn.Module]] = {}
    for idx, routing in plan.layers.items():
        layer = layers[idx]
        ffn = build_qlot_ffn(layer, routing, cfg, backend)   # captures original norm first
        originals[idx] = {"norm": layer.post_attention_layernorm, "mlp": layer.mlp}
        layer.post_attention_layernorm = nn.Identity()
        layer.mlp = ffn
    return PatchHandle(model=model, originals=originals)


def unpatch_model(handle: PatchHandle) -> None:
    handle.unpatch()
