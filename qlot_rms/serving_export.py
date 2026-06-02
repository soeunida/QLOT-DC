"""Export static per-layer serving artifacts for the (future) custom_packed path.

Reuses the EXACT packing logic in :class:`PackedProjection.from_linear`, so the
exported tensors are bit-for-bit what the reference backend uses:

  per routed layer:
    indices.pt : fp_indices, int_indices, perm
    scales.pt  : act_scales, diag_alpha (None if no Q-LOT-DC)
    gate.pt    : W_F, W_I (effective: alpha-inversed / mean-comp'd), W_I_dq
                 (fake-quant W8-G128), bias, bias_corr
    up.pt      : same for up_proj
    meta.json  : shapes, K_F, C_int, group_size, qmax, flags
  manifest.json : config + per-layer summary

Weights are written under ``out_dir`` (default ``artifacts/qlot_dc_serving/``,
which is git-ignored) since they are model-sized; nothing large is committed.

CLI:
    python -m qlot_rms.serving_export --config configs/qlot_dc_tinyllama.json \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda:0 \
        --out_dir artifacts/qlot_dc_serving [--calib_synthetic]
or with a pre-saved plan:
    python -m qlot_rms.serving_export --plan <routing.pt> --model <id> --out_dir <dir>
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch

from .config import QLotRmsConfig, RoutingPlan
from .projection import PackedProjection, get_backend
from .model_integration import find_decoder_layers


@torch.no_grad()
def export_serving_artifacts(model, plan: RoutingPlan, cfg: QLotRmsConfig,
                             out_dir: str = "artifacts/qlot_dc_serving") -> dict:
    """Export per-layer serving artifacts; returns the manifest dict."""
    os.makedirs(out_dir, exist_ok=True)
    backend = get_backend("torch_reference")  # packing logic is backend-agnostic
    layers = find_decoder_layers(model)
    manifest = {"config": cfg.to_dict(), "layers": {}}

    for i, routing in plan.layers.items():
        mlp = layers[i].mlp
        norm = layers[i].post_attention_layernorm
        gamma = norm.weight
        beta = getattr(norm, "bias", None)
        ldir = os.path.join(out_dir, f"layer_{i}")
        os.makedirs(ldir, exist_ok=True)

        proj_meta = {}
        for name, store, bc in (("gate_proj", "gate", routing.bias_corr_gate),
                                ("up_proj", "up", routing.bias_corr_up)):
            pp = PackedProjection.from_linear(
                getattr(mlp, name), routing, gamma, beta, cfg, backend=backend,
                bias_corr=bc)
            torch.save({
                "W_F": pp.W_F.cpu(),
                "W_I": pp.W_I.cpu(),                         # effective (alpha-inversed)
                "W_I_dq": pp.W_I_dq.cpu() if pp.W_I_dq is not None else None,
                "bias": pp.bias.cpu() if pp.bias is not None else None,
                "bias_corr": pp.bias_corr.cpu() if pp.bias_corr is not None else None,
            }, os.path.join(ldir, f"{store}.pt"))
            proj_meta[store] = {
                "out_features": int(pp.out_features),
                "K_F": int(pp.W_F.shape[1]),
                "C_int": int(pp.W_I.shape[1]),
                "has_bias": pp.bias is not None,
                "has_bias_corr": pp.bias_corr is not None,
                "cached_dequant": pp.W_I_dq is not None,
            }

        torch.save({
            "fp_indices": routing.fp_indices.cpu(),
            "int_indices": routing.int_indices.cpu(),
            "perm": routing.perm.cpu(),
        }, os.path.join(ldir, "indices.pt"))
        torch.save({
            "act_scales": routing.act_scales.cpu(),
            "diag_alpha": routing.diag_alpha.cpu() if routing.diag_alpha is not None else None,
            "mu_g": torch.as_tensor(routing.mu_g).cpu(),
            "mu_g_channels": routing.mu_g_channels.cpu() if routing.mu_g_channels is not None else None,
        }, os.path.join(ldir, "scales.pt"))

        meta = {
            "layer_index": int(i),
            "num_channels": routing.num_channels,
            "k_fp": routing.k_fp,
            "k_int": int(routing.int_indices.numel()),
            "grms_group_size": routing.grms_group_size,
            "w8_group_size": cfg.w8_group_size,
            "qmax": cfg.qmax,
            "diag_comp_applied": bool(routing.diag_comp_applied),
            "diag_alpha_len": int(routing.diag_alpha.numel()) if routing.diag_alpha is not None else 0,
            "grms_enabled": bool(routing.grms_enabled),
            "selected_fp_ratio": routing.selected_fp_ratio,
            "projections": proj_meta,
            "static_routing": True,
            "files": ["indices.pt", "scales.pt", "gate.pt", "up.pt"],
        }
        with open(os.path.join(ldir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        manifest["layers"][int(i)] = meta

    manifest["backend_note"] = (
        "Static serving artifacts for the experimental custom_packed backend. "
        "torch_reference remains the default correctness backend; no speedup is "
        "claimed. custom_packed requires a real kernel that passes correctness tests.")
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--plan", default=None, help="pre-saved qlot_rms_routing.pt")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="artifacts/qlot_dc_serving")
    ap.add_argument("--calib_synthetic", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16).to(args.device).eval()
    if args.plan:
        plan = RoutingPlan.load(args.plan)
        cfg = plan.config
    else:
        from .calibration import calibrate
        cfg = QLotRmsConfig.load_json(args.config)
        tok = AutoTokenizer.from_pretrained(args.model)
        plan = calibrate(model, tok, cfg, device=args.device, routing_method=cfg.routing_score,
                         allow_synthetic=args.calib_synthetic, batch_size=8)
    manifest = export_serving_artifacts(model, plan, cfg, args.out_dir)
    print(f"[serving-export] wrote {len(manifest['layers'])} layers to {args.out_dir}")


if __name__ == "__main__":
    _cli()
