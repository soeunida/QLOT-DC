"""Verify that calibration artifacts are ready for a custom_packed kernel.

Loads a saved RoutingPlan (qlot_rms_routing.pt) and checks that every routed
layer carries the frozen artifacts a packed FP16+INT8 branched kernel needs:

  * FP indices, INT indices
  * static permutation P = [FP, INT]  (=> no runtime top-k / sort needed)
  * per-channel activation scales (positive, finite, length C_int)
  * W8-G128 packable weights (with --model: gate_proj/up_proj in_features == C)
  * per-layer metadata (group sizes summing to C_int; K_F = C - C_int)
  * GroupRMS metadata when enabled (mu_g length == n_groups; mu_g_channels == C_int)

It does NOT run a kernel. It only validates packability and writes a JSON report.

Example
-------
    python -m eval.check_custom_packed_readiness \
        --plan results/qlot_rms_full_selected_tinyllama/... # (a saved .pt)
    # or produce a plan on the fly:
    python -m eval.check_custom_packed_readiness \
        --config configs/qlot_rms_tinyllama_sadnd_only.json \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda:0 --calib_synthetic
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from qlot_rms.config import QLotRmsConfig, RoutingPlan
from qlot_rms.grouprms import group_sizes_for


def check_layer(lr, model_layer=None):
    """Return (ok, checks dict) for one LayerRouting."""
    c = {}
    C = lr.num_channels
    k_fp = int(lr.fp_indices.numel())
    c_int = int(lr.int_indices.numel())

    c["has_fp_indices"] = k_fp == lr.k_fp
    c["has_int_indices"] = c_int == (C - k_fp)
    # static permutation P = [FP, INT] (sorted within each group) => no runtime routing
    expect_perm = torch.cat([torch.sort(lr.fp_indices).values,
                             torch.sort(lr.int_indices).values])
    c["static_permutation_fp_then_int"] = (
        lr.perm.numel() == C and torch.equal(lr.perm, expect_perm)
    )
    c["perm_is_permutation"] = sorted(lr.perm.tolist()) == list(range(C))
    c["mask_matches_fp"] = bool(lr.mask.sum().item() == k_fp)

    # activation scales
    a = lr.act_scales
    c["act_scales_len"] = a.numel() == c_int
    c["act_scales_positive_finite"] = bool((a > 0).all() and torch.isfinite(a).all())

    # group layout
    gsizes = group_sizes_for(c_int, lr.grms_group_size)
    c["group_sizes_consistent"] = (
        list(lr.grms_group_sizes) == gsizes and lr.grms_num_groups == len(gsizes)
        and sum(gsizes) == c_int
    )

    # GroupRMS metadata when enabled
    if lr.grms_enabled:
        c["mu_g_len_eq_n_groups"] = torch.as_tensor(lr.mu_g).numel() == lr.grms_num_groups
        c["mu_g_channels_len"] = (lr.mu_g_channels is not None
                                  and lr.mu_g_channels.numel() == c_int)
    else:
        c["grms_disabled_ok"] = True  # routing-only: no GroupRMS metadata required

    # packable weights (optional, needs the model layer)
    if model_layer is not None:
        mlp = model_layer.mlp
        ok_w = True
        for p in ("gate_proj", "up_proj"):
            W = getattr(mlp, p).weight
            ok_w = ok_w and (W.shape[1] == C)  # in_features == LN2 hidden
        c["weights_packable_in_features_eq_C"] = bool(ok_w)

    ok = all(v for v in c.values())
    return ok, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=None, help="path to a saved qlot_rms_routing.pt")
    ap.add_argument("--config", default=None,
                    help="if no --plan, calibrate with this config to produce a plan")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--calib_synthetic", action="store_true",
                    help="use synthetic calibration data (offline; readiness only)")
    ap.add_argument("--out", default="results/custom_packed_readiness.json")
    args = ap.parse_args()

    model = None
    if args.plan:
        plan = RoutingPlan.load(args.plan)
    elif args.config:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from qlot_rms.calibration import calibrate
        cfg = QLotRmsConfig.load_json(args.config)
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16).to(args.device).eval()
        plan = calibrate(model, tok, cfg, device=args.device, routing_method="sadnd",
                         allow_synthetic=args.calib_synthetic, batch_size=8)
    else:
        raise SystemExit("provide --plan or --config")

    layers = None
    if model is not None:
        from qlot_rms.model_integration import find_decoder_layers
        layers = find_decoder_layers(model)

    per_layer = {}
    all_ok = True
    grms_enabled = 0
    for i, lr in plan.layers.items():
        ml = layers[i] if layers is not None else None
        ok, checks = check_layer(lr, ml)
        per_layer[int(i)] = {"ok": ok, "checks": checks,
                             "grms_enabled": bool(lr.grms_enabled)}
        all_ok = all_ok and ok
        grms_enabled += int(bool(lr.grms_enabled))

    report = {
        "ready": all_ok,
        "num_layers": len(plan.layers),
        "grms_enabled_layers": grms_enabled,
        "backend_note": ("Artifacts validated for packing. custom_packed is still "
                         "a stub (NotImplementedError) until a real kernel exists. "
                         "No speedup is implied by readiness."),
        "per_layer": per_layer,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[readiness] ready={all_ok}  layers={len(plan.layers)}  "
          f"grms_enabled={grms_enabled}")
    if not all_ok:
        bad = [i for i, v in per_layer.items() if not v["ok"]]
        print(f"[readiness] FAILED layers: {bad}")
    print(f"[readiness] wrote {args.out}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
