"""Export static per-layer serving artifacts for SADND-CAP.

Reuses :class:`PackedProjection.from_linear`, so the exported tensors are exactly
what the reference backend uses:

  per routed layer:
    indices.pt : fp_indices, int_indices (packing-aware order), perm
    scales.pt  : act_scales (per INT channel)
    gate.pt    : W_F, W_I (W8-G128), W_I_dq (fake-quant), bias
    up.pt      : same for up_proj
    meta.json  : shapes, K_F, C_int, w8_group_size, qmax, int_permutation_mode
  manifest.json : config + per-layer summary

Weights are written under ``out_dir`` (git-ignored ``artifacts/`` by default).
torch_reference is correctness-only; no speedup is claimed.
"""

from __future__ import annotations

import json
import os

import torch

from .config import QLotRmsConfig, RoutingPlan
from .projection import PackedProjection, get_backend
from .model_integration import find_decoder_layers


@torch.no_grad()
def export_serving_artifacts(model, plan: RoutingPlan, cfg: QLotRmsConfig,
                             out_dir: str = "artifacts/sadnd_cap_serving") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    backend = get_backend("torch_reference")
    layers = find_decoder_layers(model)
    manifest = {"config": cfg.to_dict(), "layers": {}}

    for i, routing in plan.layers.items():
        mlp = layers[i].mlp
        norm = layers[i].post_attention_layernorm
        gamma, beta = norm.weight, getattr(norm, "bias", None)
        ldir = os.path.join(out_dir, f"layer_{i}")
        os.makedirs(ldir, exist_ok=True)

        proj_meta = {}
        for name, store in (("gate_proj", "gate"), ("up_proj", "up")):
            pp = PackedProjection.from_linear(getattr(mlp, name), routing, gamma, beta, cfg, backend)
            torch.save({
                "W_F": pp.W_F.cpu(), "W_I": pp.W_I.cpu(),
                "W_I_dq": pp.W_I_dq.cpu() if pp.W_I_dq is not None else None,
                "bias": pp.bias.cpu() if pp.bias is not None else None,
            }, os.path.join(ldir, f"{store}.pt"))
            proj_meta[store] = {"out_features": int(pp.out_features),
                                "K_F": int(pp.W_F.shape[1]), "C_int": int(pp.W_I.shape[1]),
                                "has_bias": pp.bias is not None}

        torch.save({"fp_indices": routing.fp_indices.cpu(),
                    "int_indices": routing.int_indices.cpu(),
                    "perm": routing.perm.cpu()}, os.path.join(ldir, "indices.pt"))
        torch.save({"act_scales": routing.act_scales.cpu()}, os.path.join(ldir, "scales.pt"))

        meta = {"layer_index": int(i), "num_channels": routing.num_channels,
                "k_fp": routing.k_fp, "k_int": int(routing.int_indices.numel()),
                "w8_group_size": routing.w8_group_size, "qmax": cfg.qmax,
                "routing_score": routing.routing_score,
                "int_permutation_mode": routing.int_permutation_mode,
                "static_routing": True, "projections": proj_meta,
                "files": ["indices.pt", "scales.pt", "gate.pt", "up.pt"]}
        with open(os.path.join(ldir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        manifest["layers"][int(i)] = meta

    manifest["backend_note"] = ("Static SADND-CAP serving artifacts. "
                                "torch_reference is correctness-only; no speedup claimed. "
                                "A real custom_packed kernel must pass correctness tests.")
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="artifacts/sadnd_cap_serving")
    ap.add_argument("--calib_synthetic", action="store_true")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(args.device).eval()
    if args.plan:
        plan = RoutingPlan.load(args.plan); cfg = plan.config
    else:
        from .calibration import calibrate
        cfg = QLotRmsConfig.load_json(args.config)
        tok = AutoTokenizer.from_pretrained(args.model)
        plan = calibrate(model, tok, cfg, device=args.device,
                         allow_synthetic=args.calib_synthetic, batch_size=8)
    m = export_serving_artifacts(model, plan, cfg, args.out_dir)
    print(f"[serving-export] wrote {len(m['layers'])} layers to {args.out_dir}")


if __name__ == "__main__":
    _cli()
