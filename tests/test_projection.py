"""Packed projection tests (SADND-CAP): output shape, FP/INT split, backends."""

import pytest
import torch
import torch.nn as nn

from qlot_rms.config import QLotRmsConfig, LayerRouting
from qlot_rms.projection import (
    PackedProjection, get_backend, CustomPackedBackend, branch_inputs_from_slices,
)
from qlot_rms.sadnd_cap import select_fp_int_channels


def _routing(C, fp_ratio=0.06):
    d = torch.rand(C)
    fp, intc = select_fp_int_channels(d, int(fp_ratio * C))
    perm = torch.cat([fp, intc])
    mask = torch.zeros(C, dtype=torch.bool); mask[fp] = True
    return LayerRouting(
        layer_index=0, num_channels=C, k_fp=int(fp.numel()),
        fp_indices=fp, int_indices=intc, perm=perm, mask=mask, delta_tilde=d,
        act_scales=torch.rand(intc.numel()) + 0.05, w8_group_size=128)


def test_packed_output_shape_equals_linear():
    C = 80
    r = _routing(C)
    cfg = QLotRmsConfig(enable_qlot_rms=True)
    lin = nn.Linear(C, 33, bias=True)
    pp = PackedProjection.from_linear(lin, r, torch.randn(C), None, cfg)
    u = torch.randn(2, 5, C)
    out, ref = pp(u), lin(u)
    assert out.shape == ref.shape == (2, 5, 33)
    assert torch.isfinite(out).all()


def test_fp_int_weight_slices_match_columns():
    C = 64
    r = _routing(C)
    cfg = QLotRmsConfig(enable_qlot_rms=True)
    lin = nn.Linear(C, 16, bias=False)
    pp = PackedProjection.from_linear(lin, r, torch.ones(C), None, cfg)
    assert torch.allclose(pp.W_F.float(), lin.weight.data[:, r.fp_indices].to(torch.float16).float(), atol=1e-2)
    assert torch.allclose(pp.W_I.float(), lin.weight.data[:, r.int_indices].to(torch.float16).float(), atol=1e-2)


def test_cached_dequant_equals_uncached():
    C = 96
    r = _routing(C)
    lin = nn.Linear(C, 40, bias=True)
    gamma = torch.randn(C)
    pp_c = PackedProjection.from_linear(lin, r, gamma, None,
                                        QLotRmsConfig(enable_qlot_rms=True, cache_dequant_weight=True))
    pp_n = PackedProjection.from_linear(lin, r, gamma, None,
                                        QLotRmsConfig(enable_qlot_rms=True, cache_dequant_weight=False))
    assert pp_c.W_I_dq is not None and pp_n.W_I_dq is None
    u = torch.randn(3, 11, C)
    assert torch.equal(pp_c(u), pp_n(u))   # caching is bit-identical


def test_matmul_shared_equals_forward_from_branches():
    from qlot_rms.quant import quantize_activation_int8
    C = 96
    r = _routing(C)
    cfg = QLotRmsConfig(enable_qlot_rms=True, cache_dequant_weight=True)
    pg = PackedProjection.from_linear(nn.Linear(C, 40, bias=True), r, torch.randn(C), None, cfg)
    u = torch.randn(2, 9, C)
    y_F, y_I = branch_inputs_from_slices(u, pg.fp_indices, pg.int_indices,
                                         pg.gamma_F, pg.gamma_I, pg.beta_F, pg.beta_I)
    a = pg.forward_from_branches(y_F, y_I)
    yq = quantize_activation_int8(y_I, pg.act_scales, qmax=pg.qmax).float()
    b = pg.matmul_shared(y_F.to(torch.float16), yq)
    assert torch.equal(a, b)


def test_custom_packed_is_stub():
    be = get_backend("custom_packed")
    assert isinstance(be, CustomPackedBackend) and be.available() is False
    with pytest.raises(NotImplementedError):
        be.fp_matmul(torch.randn(2, 4), torch.randn(8, 4))
    with pytest.raises(NotImplementedError):
        be.int_matmul(torch.randn(2, 4), torch.randn(8, 4), torch.ones(4), 128, 127)


def test_default_backend_is_torch_reference():
    assert QLotRmsConfig().backend == "torch_reference"
    assert get_backend("torch_reference").name == "torch_reference"
