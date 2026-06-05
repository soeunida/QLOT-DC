"""Minimal end-to-end SADND-CAP sanity test (offline, CPU, no download).

Verifies on a tiny randomly-initialized Llama:
  * calibration metadata (FP/INT indices, [FP, INT] permutation, packing-aware
    INT order, positive activation scales)
  * static routing (no runtime top-k/sort)
  * patched forward runs, preserves output shape, deterministic
  * unpatch restores the original model exactly

Run:  python eval/run_tiny_sanity.py
Exits non-zero on any failure.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from qlot_rms.config import QLotRmsConfig, RoutingPlan
from qlot_rms.calibration import calibrate
from qlot_rms.model_integration import patch_model, unpatch_model, find_decoder_layers


class _Tok:
    vocab_size = 256


def build_tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=128, tie_word_embeddings=True)
    return LlamaForCausalLM(cfg).eval()


def main():
    model = build_tiny()
    cfg = QLotRmsConfig(enable_qlot_rms=True, method="sadnd_cap",
                        routing_score="output_aware_sadnd", fp_budget_mode="global",
                        int_permutation_mode="packing_aware", fp_ratio=0.1,
                        calibration_samples=8, calibration_seq_len=16,
                        num_calib_subsets=3, subset_size=4, act_scale_max_tokens=256)

    print("[sanity] calibrating (synthetic data) ...")
    plan = calibrate(model, _Tok(), cfg, device="cpu", allow_synthetic=True,
                     batch_size=2, verbose=True)
    assert isinstance(plan, RoutingPlan)
    assert len(plan.layers) == model.config.num_hidden_layers
    for lr in plan.layers.values():
        assert lr.num_channels == model.config.hidden_size
        assert lr.perm.tolist() == lr.fp_indices.tolist() + lr.int_indices.tolist()
        assert bool(lr.mask[lr.fp_indices].all())
        assert sorted(lr.perm.tolist()) == list(range(lr.num_channels))
        assert lr.act_scales.numel() == lr.int_indices.numel()
        assert bool((lr.act_scales > 0).all())
    print(f"[sanity] routing OK for {len(plan.layers)} layers; "
          f"K_F={next(iter(plan.layers.values())).k_fp}")

    out_dir = os.path.join(os.path.dirname(__file__), "_sanity_out")
    paths = plan.save(out_dir)
    reloaded = RoutingPlan.load(paths["pt"])
    assert torch.equal(reloaded.layers[0].perm, plan.layers[0].perm)
    print(f"[sanity] metadata saved+reloaded: {paths['json']}")

    ids = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        base = model(ids).logits.clone()
    handle = patch_model(model, plan, cfg)
    layers = find_decoder_layers(model)
    assert layers[0].mlp.down_proj is not None
    with torch.no_grad():
        out1 = model(ids).logits.clone()
        out2 = model(ids).logits.clone()
    assert out1.shape == base.shape and torch.isfinite(out1).all()
    assert torch.allclose(out1, out2)
    print(f"[sanity] patched forward OK, shape={tuple(out1.shape)} (deterministic)")

    unpatch_model(handle)
    with torch.no_grad():
        restored = model(ids).logits
    assert torch.allclose(restored, base, atol=1e-5)
    print("[sanity] unpatch restored baseline exactly")
    print("\nSANITY PASSED")


if __name__ == "__main__":
    main()
