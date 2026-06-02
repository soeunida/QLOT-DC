"""Determinism: same seed + same data => identical routing artifacts."""

import torch

from qlot_rms.calibration import calibrate


def _run(model, tok, cfg, method="sadnd"):
    return calibrate(model, tok, cfg, device="cpu", routing_method=method,
                     allow_synthetic=True, batch_size=2)


def test_calibration_deterministic_sadnd(tiny_model, fake_tokenizer, small_config):
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
        assert torch.allclose(a.mu_g, b.mu_g)
        assert torch.allclose(a.mu_g_channels, b.mu_g_channels)


def test_calibration_deterministic_random(tiny_model, fake_tokenizer, small_config):
    p1 = _run(tiny_model, fake_tokenizer, small_config, method="random")
    p2 = _run(tiny_model, fake_tokenizer, small_config, method="random")
    for i in p1.layers:
        assert torch.equal(p1.layers[i].perm, p2.layers[i].perm)


def test_act_scales_positive_and_frozen(tiny_model, fake_tokenizer, small_config):
    plan = _run(tiny_model, fake_tokenizer, small_config)
    for lr in plan.layers.values():
        assert lr.act_scales.numel() == lr.int_indices.numel()
        assert torch.isfinite(lr.act_scales).all()
        assert (lr.act_scales > 0).all()
        # mu_g is PER-GROUP: length must equal grms_num_groups, all positive
        assert lr.mu_g.numel() == lr.grms_num_groups
        assert (lr.mu_g > 0).all()
        assert torch.isfinite(lr.mu_g).all()
        # mu_g_channels broadcasts mu_g across the INT channels
        assert lr.mu_g_channels.numel() == lr.int_indices.numel()


def test_save_load_roundtrip(tiny_model, fake_tokenizer, small_config, tmp_path):
    from qlot_rms.config import RoutingPlan

    plan = _run(tiny_model, fake_tokenizer, small_config)
    paths = plan.save(str(tmp_path))
    loaded = RoutingPlan.load(paths["pt"])
    for i in plan.layers:
        assert torch.equal(plan.layers[i].perm, loaded.layers[i].perm)
        assert torch.allclose(plan.layers[i].act_scales, loaded.layers[i].act_scales)
