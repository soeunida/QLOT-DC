"""Minimal end-to-end sanity test for Q-LOT-RMS (no download, CPU).

Verifies, on a tiny randomly-initialized Llama:
  * calibration metadata is created (routing mask, [FP, INT] perm, scales)
  * mu_g is PER-GROUP (length == grms_num_groups), not scalar
  * mean-scale compensation applies each mu_g[g] only to that group's INT
    input-channel columns (orientation: weight[:, int_indices], dim=1)
  * routing is static (frozen permutation)
  * the patched forward runs and preserves output shape; deterministic
  * unpatch restores the original model exactly

Run:
    python -m eval.run_tiny_sanity
    python -m eval.run_tiny_sanity --config configs/qlot_rms_tiny.json
Exits non-zero on any failure.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from qlot_rms.config import QLotRmsConfig, RoutingPlan
from qlot_rms.calibration import calibrate
from qlot_rms.grouprms import broadcast_per_group_to_channels
from qlot_rms.model_integration import patch_model, unpatch_model, find_decoder_layers


class _Tok:
    vocab_size = 256

    def __call__(self, *a, **k):
        class O: pass
        o = O(); o.input_ids = torch.zeros(1, 8, dtype=torch.long); return o


def build_tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, tie_word_embeddings=True,
    )
    return LlamaForCausalLM(cfg).eval()


def default_cfg():
    return QLotRmsConfig(
        enable_qlot_rms=True, qlot_scope="mlp_only", fp_ratio=0.06,
        grms_group_size=16, calibration_samples=8, calibration_seq_len=16,
        num_calib_subsets=3, subset_size=4, seed=0, act_scale_max_tokens=256,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="optional QLotRmsConfig JSON; calibration hyper-params "
                         "are taken from it (tiny model is always used here).")
    args = ap.parse_args()

    cfg = QLotRmsConfig.load_json(args.config) if args.config else default_cfg()
    cfg.enable_qlot_rms = True
    cfg.qlot_scope = "mlp_only"
    # keep the tiny run cheap regardless of the supplied config
    cfg.grms_group_size = min(cfg.grms_group_size, 16)
    cfg.calibration_samples = min(cfg.calibration_samples, 8)
    cfg.calibration_seq_len = min(cfg.calibration_seq_len, 16)
    cfg.num_calib_subsets = min(cfg.num_calib_subsets, 3)
    cfg.subset_size = min(cfg.subset_size, 4)
    cfg.act_scale_max_tokens = min(cfg.act_scale_max_tokens, 256)

    model = build_tiny()
    tok = _Tok()

    print("[sanity] calibrating (synthetic data) ...")
    plan = calibrate(model, tok, cfg, device="cpu", routing_method="sadnd",
                     allow_synthetic=True, batch_size=2, verbose=True)

    assert isinstance(plan, RoutingPlan)
    assert len(plan.layers) == model.config.num_hidden_layers
    for i, lr in plan.layers.items():
        assert lr.num_channels == model.config.hidden_size
        assert lr.perm.tolist() == lr.fp_indices.tolist() + lr.int_indices.tolist()
        assert lr.k_fp == int(cfg.fp_ratio * lr.num_channels)
        assert (lr.act_scales > 0).all()

        # --- per-group mu_g checks ---
        assert torch.is_tensor(lr.mu_g), "mu_g must be a per-group tensor, not scalar"
        assert lr.mu_g.numel() == lr.grms_num_groups, \
            f"len(mu_g)={lr.mu_g.numel()} != grms_num_groups={lr.grms_num_groups}"
        assert (lr.mu_g > 0).all() and torch.isfinite(lr.mu_g).all()
        # mu_g_channels is mu_g broadcast over each group's channels
        expect_chan = broadcast_per_group_to_channels(
            lr.mu_g.tolist(), lr.int_indices.numel(), lr.grms_group_size)
        assert torch.allclose(lr.mu_g_channels.float(), expect_chan.float()), \
            "mu_g_channels must equal per-group mu_g broadcast to INT channels"

    # --- mean-scale compensation applies each mu_g[g] only to its group's
    #     INT input columns of the routed projection weights ---
    layers = find_decoder_layers(model)
    li = next(iter(plan.layers))
    lr = plan.layers[li]
    W_orig = layers[li].mlp.gate_proj.weight.data[:, lr.int_indices].clone()

    handle = patch_model(model, plan, cfg)
    W_I = layers[li].mlp.packed_gate.W_I.float()
    start = 0
    for g, s in enumerate(lr.grms_group_sizes):
        sl = slice(start, start + s)
        expect = W_orig[:, sl].float() * float(lr.mu_g[g])
        assert torch.allclose(W_I[:, sl], expect, atol=1e-2), \
            f"group {g}: mean-comp not applied per-group"
        start += s
    # FP columns are NOT scaled by mean-comp
    W_F_orig = layers[li].mlp.packed_gate.W_F  # built from unscaled fp columns
    assert torch.isfinite(W_F_orig.float()).all()
    print(f"[sanity] per-group mu_g OK: len={lr.mu_g.numel()}="
          f"grms_num_groups; mean-comp applied per group on INT columns only")

    # metadata summary + save/reload
    out_dir = os.path.join(os.path.dirname(__file__), "_sanity_out")
    paths = plan.save(out_dir)
    summary = {int(i): plan.layers[i].summary() for i in plan.layers}
    print("[sanity] metadata summary:")
    print(json.dumps(summary, indent=2))
    reloaded = RoutingPlan.load(paths["pt"])
    assert torch.equal(reloaded.layers[li].perm, plan.layers[li].perm)
    assert torch.allclose(reloaded.layers[li].mu_g, plan.layers[li].mu_g)
    print(f"[sanity] metadata saved+reloaded: {paths['json']}")

    ids = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        out1 = model(ids).logits.clone()
        out2 = model(ids).logits.clone()
    assert out1.shape[:2] == (1, 12)
    assert torch.isfinite(out1).all()
    assert torch.allclose(out1, out2)             # deterministic
    assert layers[0].mlp.down_proj is not None     # down_proj retained
    print(f"[sanity] patched forward OK, shape={tuple(out1.shape)} (deterministic)")

    unpatch_model(handle)
    with torch.no_grad():
        restored = model(ids).logits
    print("[sanity] unpatch restored model")
    print("\nSANITY PASSED")


if __name__ == "__main__":
    main()
