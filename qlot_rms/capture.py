"""Pre-affine LN2 / RMSNorm activation capture.

To route the Pre-LN ``LN2 -> FFN`` interface we need the *pre-affine*
normalized activation ``u`` (before the elementwise gamma/beta), not the module
output (which is post-affine).

We register a forward hook on each routed layer's second norm
(``post_attention_layernorm`` for Llama-family) that recomputes ``u`` from the
norm's *input* ``x``:

* RMSNorm:   ``u = x * rsqrt(mean(x^2, -1) + eps)``           (gamma = weight, beta = 0)
* LayerNorm: ``u = (x - mean(x)) * rsqrt(var(x) + eps)``      (gamma = weight, beta = bias)

The hook accumulates captured ``u`` tensors (flattened to ``[tokens, C]``) on
CPU per layer.  The calibration driver clears buffers between subsets so peak
memory is bounded by a single subset's activations.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


def detect_norm_type(norm_module: nn.Module) -> str:
    """Best-effort classification of a norm module as rmsnorm / layernorm."""
    name = norm_module.__class__.__name__.lower()
    if "rms" in name:
        return "rmsnorm"
    if isinstance(norm_module, nn.LayerNorm) or "layernorm" in name:
        return "layernorm"
    # Heuristic: RMSNorm modules have a ``weight`` but no ``bias``.
    if getattr(norm_module, "bias", None) is not None:
        return "layernorm"
    return "rmsnorm"


def get_norm_eps(norm_module: nn.Module, default: float = 1e-6) -> float:
    for attr in ("variance_epsilon", "eps"):
        if hasattr(norm_module, attr):
            return float(getattr(norm_module, attr))
    return default


def pre_affine_normalize(x: torch.Tensor, norm_module: nn.Module) -> torch.Tensor:
    """Compute the pre-affine normalized activation ``u`` from norm input ``x``.

    Returns ``u`` in float32 (the affine gamma/beta are applied later, by the
    routed projection / calibration, not here).
    """
    nt = detect_norm_type(norm_module)
    eps = get_norm_eps(norm_module)
    xf = x.float()
    if nt == "rmsnorm":
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        return xf * torch.rsqrt(var + eps)
    # layernorm
    mean = xf.mean(dim=-1, keepdim=True)
    var = (xf - mean).pow(2).mean(dim=-1, keepdim=True)
    return (xf - mean) * torch.rsqrt(var + eps)


class PreAffineCapture:
    """Registers forward hooks to capture pre-affine ``u`` per routed layer.

    Usage::

        cap = PreAffineCapture({0: norm0, 1: norm1, ...})
        cap.attach()
        cap.reset()                      # before each subset
        run_forward(model, subset_batch) # hooks fill cap.buffers
        u_by_layer = cap.collect()       # {layer_idx: [tokens, C] (cpu, fp32)}
        ...
        cap.detach()                     # restore model
    """

    def __init__(self, norm_modules: Dict[int, nn.Module], store_device: str = "cpu"):
        self.norm_modules = norm_modules
        self.store_device = store_device
        self.buffers: Dict[int, List[torch.Tensor]] = {i: [] for i in norm_modules}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._enabled = False

    def _make_hook(self, layer_idx: int, norm_module: nn.Module):
        def hook(module, inputs, output):
            if not self._enabled:
                return
            x = inputs[0]
            u = pre_affine_normalize(x, norm_module)
            u = u.reshape(-1, u.shape[-1]).to(self.store_device)
            self.buffers[layer_idx].append(u)
        return hook

    def attach(self) -> "PreAffineCapture":
        for idx, norm in self.norm_modules.items():
            h = norm.register_forward_hook(self._make_hook(idx, norm))
            self._handles.append(h)
        self._enabled = True
        return self

    def reset(self) -> None:
        for i in self.buffers:
            self.buffers[i] = []

    def collect(self) -> Dict[int, torch.Tensor]:
        out: Dict[int, torch.Tensor] = {}
        for i, chunks in self.buffers.items():
            if chunks:
                out[i] = torch.cat(chunks, dim=0)
        return out

    def set_enabled(self, flag: bool) -> None:
        self._enabled = flag

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._enabled = False
