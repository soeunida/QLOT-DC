"""Determinism + artifact validity: same seed + data => identical SADND-CAP plan."""

import torch

from qlot_rms.calibration import calibrate
from qlot_rms.config import RoutingPlan


def _run(model, tok, cfg):
    return calibrate(model, tok, cfg, device="cpu", allow_synthetic=True, batch_size=2)


def test_calibration_deterministic(tiny_model, fake_tokenizer, small_config):
    p1 = _run(tiny_model, fake_tokenizer, small_config)
    p2 = _run(tiny_model, fake_tokenizer, small_config)
    assert set(p1.layers) == set(p2.layers)
    for i in p1.layers:
        a, b = p1.layers[i], p2.layers[i]
        assert torch.equal(a.perm, b.perm)
        assert torch.equal(a.fp_indices, b.fp_indices)
        assert torch.equal(a.int_indices, b.int_indices)
        assert torch.allclose(a.delta_tilde, b.delta_tilde)
        assert torch.allclose(a.act_scales, b.act_scales)


def test_artifacts_valid(tiny_model, fake_tokenizer, small_config):
    plan = _run(tiny_model, fake_tokenizer, small_config)
    for lr in plan.layers.values():
        assert lr.act_scales.numel() == lr.int_indices.numel()
        assert bool((lr.act_scales > 0).all()) and torch.isfinite(lr.act_scales).all()
        # static [FP, INT] permutation; valid permutation; FP block first
        assert lr.perm.tolist() == lr.fp_indices.tolist() + lr.int_indices.tolist()
        assert sorted(lr.perm.tolist()) == list(range(lr.num_channels))
        assert bool(lr.mask[lr.fp_indices].all())


def test_save_load_roundtrip(tiny_model, fake_tokenizer, small_config, tmp_path):
    plan = _run(tiny_model, fake_tokenizer, small_config)
    paths = plan.save(str(tmp_path))
    loaded = RoutingPlan.load(paths["pt"])
    for i in plan.layers:
        assert torch.equal(plan.layers[i].perm, loaded.layers[i].perm)
        assert torch.allclose(plan.layers[i].act_scales, loaded.layers[i].act_scales)
