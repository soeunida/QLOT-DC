"""Packed projection tests: mean-comp orientation, output shape, backends."""

import pytest
import torch
import torch.nn as nn

from qlot_rms.config import QLotRmsConfig, LayerRouting
from qlot_rms.projection import (
    apply_mean_scale_compensation,
    PackedProjection,
    get_backend,
    CustomPackedBackend,
)
from qlot_rms.sadnd import assign_channels


def _make_routing(C, fp_ratio=0.06, group_size=16):
    d = torch.rand(C)
    fp, intc, perm, mask = assign_channels(d, fp_ratio)
    from qlot_rms.grouprms import group_sizes_for

    gs = group_sizes_for(intc.numel(), group_size)
    from qlot_rms.grouprms import broadcast_per_group_to_channels

    mu_groups = [1.7] * len(gs)            # constant per group -> equals scalar 1.7
    mu_chan = broadcast_per_group_to_channels(mu_groups, intc.numel(), group_size)
    return LayerRouting(
        layer_index=0, num_channels=C, k_fp=fp.numel(),
        fp_indices=fp, int_indices=intc, perm=perm, mask=mask, delta_tilde=d,
        grms_group_size=group_size, grms_num_groups=len(gs), grms_group_sizes=gs,
        mu_g=torch.tensor(mu_groups), mu_g_channels=mu_chan,
        act_scales=torch.ones(intc.numel()),
    )


def test_mean_comp_modifies_only_int_input_columns():
    C = 80
    routing = _make_routing(C)
    W = torch.randn(40, C)  # [out, in]
    Wc = apply_mean_scale_compensation(W, routing.int_indices, mu_g=1.7)
    # INT *columns* scaled by mu_g
    assert torch.allclose(Wc[:, routing.int_indices], W[:, routing.int_indices] * 1.7)
    # FP columns untouched
    assert torch.allclose(Wc[:, routing.fp_indices], W[:, routing.fp_indices])
    # rows are NOT what we scale (orientation guard): scaling is along dim=1
    assert Wc.shape == W.shape


def test_packed_output_shape_equals_original_linear():
    C = 80
    routing = _make_routing(C)
    cfg = QLotRmsConfig(enable_qlot_rms=True)
    lin = nn.Linear(C, 33, bias=True)
    gamma = torch.randn(C)
    pp = PackedProjection.from_linear(lin, routing, gamma, None, cfg)
    u = torch.randn(2, 5, C)
    out = pp(u)
    ref = lin(u)
    assert out.shape == ref.shape == (2, 5, 33)
    assert torch.isfinite(out).all()


def test_mean_comp_baked_into_int_weight_slice():
    C = 64
    routing = _make_routing(C)
    cfg = QLotRmsConfig(enable_qlot_rms=True, use_grms=True, use_mean_comp=True)
    lin = nn.Linear(C, 16, bias=False)
    gamma = torch.ones(C)
    pp = PackedProjection.from_linear(lin, routing, gamma, None, cfg)
    # mu_g is per-group; apply the per-channel broadcast (all 1.7 here)
    expected = (lin.weight.data[:, routing.int_indices]
                * routing.mu_g_channels).to(torch.float16)
    assert torch.allclose(pp.W_I.float(), expected.float(), atol=1e-2)
    # FP slice is NOT scaled
    assert torch.allclose(
        pp.W_F.float(), lin.weight.data[:, routing.fp_indices].to(torch.float16).float(),
        atol=1e-2,
    )


def test_per_group_mu_g_applied_per_group():
    # Each group's mu_g must scale only that group's INT input columns.
    from qlot_rms.grouprms import group_sizes_for, broadcast_per_group_to_channels

    C = 80
    d = torch.rand(C)
    fp, intc, perm, mask = assign_channels(d, 0.06)
    gsz = 16
    gs = group_sizes_for(intc.numel(), gsz)
    mu_groups = [float(2 + g) for g in range(len(gs))]  # distinct per group
    mu_chan = broadcast_per_group_to_channels(mu_groups, intc.numel(), gsz)
    routing = LayerRouting(
        layer_index=0, num_channels=C, k_fp=fp.numel(),
        fp_indices=fp, int_indices=intc, perm=perm, mask=mask, delta_tilde=d,
        grms_group_size=gsz, grms_num_groups=len(gs), grms_group_sizes=gs,
        mu_g=torch.tensor(mu_groups), mu_g_channels=mu_chan,
        act_scales=torch.ones(intc.numel()),
    )
    assert routing.mu_g.numel() == routing.grms_num_groups
    cfg = QLotRmsConfig(enable_qlot_rms=True, use_grms=True, use_mean_comp=True,
                        grms_group_size=gsz)
    lin = nn.Linear(C, 12, bias=False)
    pp = PackedProjection.from_linear(lin, routing, torch.ones(C), None, cfg)
    # verify column g-block of W_I equals original int column * mu_groups[g]
    start = 0
    Worig = lin.weight.data[:, intc]
    for g, s in enumerate(gs):
        sl = slice(start, start + s)
        expect = (Worig[:, sl] * mu_groups[g]).to(torch.float16)
        assert torch.allclose(pp.W_I[:, sl].float(), expect.float(), atol=1e-2), g
        start += s


def test_use_grms_false_skips_grms():
    C = 64
    routing = _make_routing(C)
    cfg_on = QLotRmsConfig(enable_qlot_rms=True, use_grms=True)
    cfg_off = QLotRmsConfig(enable_qlot_rms=True, use_grms=False)
    lin = nn.Linear(C, 16, bias=False)
    gamma = torch.ones(C)
    u = torch.randn(3, C)
    out_on = PackedProjection.from_linear(lin, routing, gamma, None, cfg_on)(u)
    out_off = PackedProjection.from_linear(lin, routing, gamma, None, cfg_off)(u)
    # GroupRMS changes the function -> outputs differ
    assert not torch.allclose(out_on, out_off)


def test_custom_packed_backend_is_stub():
    be = get_backend("custom_packed")
    assert isinstance(be, CustomPackedBackend)
    with pytest.raises(NotImplementedError):
        be.fp_matmul(torch.randn(2, 4), torch.randn(8, 4))
    with pytest.raises(NotImplementedError):
        be.int_matmul(torch.randn(2, 4), torch.randn(8, 4), torch.ones(4), 128, 127)


def test_default_backend_is_torch_reference():
    cfg = QLotRmsConfig()
    assert cfg.backend == "torch_reference"
    assert get_backend(cfg.backend).name == "torch_reference"
