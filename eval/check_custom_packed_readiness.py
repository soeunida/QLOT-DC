"""Verify SADND-CAP calibration artifacts are ready for a custom_packed kernel.

Checks each routed layer has the static artifacts a packed FP16+INT8 kernel
needs: FP/INT indices, static permutation [FP, INT] (no runtime top-k/sort),
positive/finite per-INT-channel activation scales, W8-G128 group layout, and
(with --model) packable gate/up weights. custom_packed remains experimental;
readiness does NOT imply a kernel exists or any speedup.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from qlot_rms.config import QLotRmsConfig, RoutingPlan
from qlot_rms.quant import group_sizes_for


def check_layer(lr, model_layer=None):
    c = {}
    C = lr.num_channels
    k_fp = int(lr.fp_indices.numel())
    c_int = int(lr.int_indices.numel())
    c["has_fp_indices"] = k_fp == lr.k_fp
    c["has_int_indices"] = c_int == (C - k_fp)
    # static permutation [FP, INT]; FP block first; valid permutation
    expect = torch.cat([torch.sort(lr.fp_indices).values, lr.int_indices])
    c["perm_fp_block_first"] = (lr.perm.numel() == C and torch.equal(lr.perm, expect))
    c["perm_is_permutation"] = sorted(lr.perm.tolist()) == list(range(C))
    c["mask_matches_fp"] = bool(lr.mask.sum().item() == k_fp)
    # INT permutation only reorders INT channels (set preserved)
    c["int_set_preserved"] = sorted(lr.int_indices.tolist()) == \
        sorted(torch.nonzero(~lr.mask, as_tuple=False).squeeze(-1).tolist())
    # activation scales
    c["act_scales_len"] = lr.act_scales.numel() == c_int
    c["act_scales_pos_finite"] = bool((lr.act_scales > 0).all() and torch.isfinite(lr.act_scales).all())
    # W8-G128 group layout
    gs = group_sizes_for(c_int, lr.w8_group_size)
    c["group_layout_ok"] = sum(gs) == c_int
    if model_layer is not None:
        mlp = model_layer.mlp
        c["weights_packable_in_features_eq_C"] = all(
            getattr(mlp, p).weight.shape[1] == C for p in ("gate_proj", "up_proj"))
    return all(c.values()), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=None, help="path to sadnd_cap_routing.pt")
    ap.add_argument("--config", default=None, help="if no --plan, calibrate to produce one")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--calib_synthetic", action="store_true")
    ap.add_argument("--out", default="results/sadnd_cap_readiness.json")
    args = ap.parse_args()

    model = None
    if args.plan:
        plan = RoutingPlan.load(args.plan)
    elif args.config:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from qlot_rms.calibration import calibrate
        cfg = QLotRmsConfig.load_json(args.config)
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(args.device).eval()
        plan = calibrate(model, tok, cfg, device=args.device,
                         allow_synthetic=args.calib_synthetic, batch_size=8)
    else:
        raise SystemExit("provide --plan or --config")

    layers = None
    if model is not None:
        from qlot_rms.model_integration import find_decoder_layers
        layers = find_decoder_layers(model)

    per_layer, all_ok = {}, True
    for i, lr in plan.layers.items():
        ok, checks = check_layer(lr, layers[i] if layers is not None else None)
        per_layer[int(i)] = {"ok": ok, "checks": checks}
        all_ok = all_ok and ok

    try:
        from qlot_rms.projection import CustomPackedBackend
        cp_avail = bool(CustomPackedBackend.available())
    except Exception:
        cp_avail = False

    report = {"ready": all_ok, "num_layers": len(plan.layers),
              "custom_packed_experimental": True, "custom_packed_kernel_available": cp_avail,
              "backend_note": ("Artifacts validated for packing. custom_packed is a stub "
                               "(no kernel). torch_reference is the default; no speedup implied."),
              "per_layer": per_layer}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"[readiness] ready={all_ok} layers={len(plan.layers)} "
          f"custom_packed_experimental=True kernel_available={cp_avail}")
    if not all_ok:
        print("[readiness] FAILED:", [i for i, v in per_layer.items() if not v["ok"]])
    print(f"[readiness] wrote {args.out}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
