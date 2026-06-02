"""Routing/assignment tests: bottom-K INT, FP ratio, [FP, INT] permutation."""

import torch

from qlot_rms.sadnd import assign_channels, random_routing, magnitude_routing


def test_fp_ratio_count_floor():
    d = torch.rand(100)
    fp, intc, perm, mask = assign_channels(d, fp_ratio=0.06)
    assert fp.numel() == 6              # floor(0.06 * 100)
    assert intc.numel() == 94
    assert perm.numel() == 100


def test_bottomk_int_topk_fp():
    # FP must be the K_F highest-distortion channels; INT the rest.
    d = torch.arange(10).float()       # distinct, increasing
    fp, intc, perm, mask = assign_channels(d, fp_ratio=0.3)  # K_F=3
    assert set(fp.tolist()) == {7, 8, 9}          # top-3 distortion -> FP
    assert set(intc.tolist()) == set(range(7))    # bottom-7 -> INT


def test_permutation_is_fp_then_int_original_order():
    d = torch.tensor([5.0, 0.0, 9.0, 1.0, 8.0])
    fp, intc, perm, mask = assign_channels(d, fp_ratio=0.4)  # K_F=2 -> {2,4}
    assert fp.tolist() == [2, 4]                  # original order within FP
    assert intc.tolist() == [0, 1, 3]             # original order within INT
    assert perm.tolist() == [2, 4, 0, 1, 3]       # [FP, INT]
    assert mask[2] and mask[4] and not mask[0]


def test_mask_consistency():
    d = torch.rand(64)
    fp, intc, perm, mask = assign_channels(d, fp_ratio=0.1)
    assert mask.dtype == torch.bool
    assert mask.sum().item() == fp.numel()
    assert bool(mask[fp].all())
    assert not bool(mask[intc].any())
    # FP and INT partition all channels
    assert sorted(fp.tolist() + intc.tolist()) == list(range(64))


def test_random_routing_same_k_and_partition():
    g = torch.Generator().manual_seed(0)
    fp, intc, perm, mask = random_routing(100, fp_ratio=0.06, generator=g)
    assert fp.numel() == 6 and intc.numel() == 94
    assert sorted(perm.tolist()) == list(range(100))


def test_random_routing_deterministic_with_seed():
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a = random_routing(50, 0.1, g1)[0]
    b = random_routing(50, 0.1, g2)[0]
    assert torch.equal(a, b)


def test_magnitude_routing_picks_largest():
    mag = torch.tensor([0.1, 9.0, 0.2, 8.0, 0.3])
    fp, intc, perm, mask = magnitude_routing(mag, fp_ratio=0.4)  # K_F=2
    assert set(fp.tolist()) == {1, 3}
