"""StaticScale hard-layer FP reallocation: unit tests (no model)."""
from staticscale.hard_layer_search import rank_hard_layers, reallocate_fp_budget


def test_rank_hard_layers():
    errs = {0: 0.1, 1: 0.5, 2: 0.3, 3: 0.9}
    assert rank_hard_layers(errs, 2) == [3, 1]


def test_reallocation_preserves_total_exactly():
    per_layer_k = {0: 10, 1: 10, 2: 10, 3: 10}
    channels = {i: 64 for i in range(4)}
    errors = {0: 0.9, 1: 0.8, 2: 0.1, 3: 0.05}       # 0,1 hard; 2,3 donors
    new = reallocate_fp_budget(per_layer_k, errors, channels, top_k_hard=2, move_frac=0.5)
    assert sum(new.values()) == sum(per_layer_k.values())   # exact total
    assert new[0] >= 10 and new[1] >= 10                    # hard layers gained
    assert new[2] <= 10 and new[3] <= 10                    # donors gave


def test_reallocation_respects_channel_cap():
    per_layer_k = {0: 60, 1: 5, 2: 5, 3: 5}
    channels = {0: 64, 1: 64, 2: 64, 3: 64}
    errors = {0: 0.9, 1: 0.1, 2: 0.1, 3: 0.1}        # layer 0 is hard but near cap
    new = reallocate_fp_budget(per_layer_k, errors, channels, top_k_hard=1, move_frac=0.5)
    assert sum(new.values()) == sum(per_layer_k.values())   # exact total preserved
    assert all(0 <= new[i] <= channels[i] for i in new)     # within channel bounds


def test_reallocation_zero_move_is_identity():
    per_layer_k = {0: 8, 1: 8, 2: 8}
    channels = {i: 32 for i in range(3)}
    errors = {0: 0.5, 1: 0.3, 2: 0.1}
    new = reallocate_fp_budget(per_layer_k, errors, channels, top_k_hard=1, move_frac=0.0)
    assert new == per_layer_k                                # nothing moved
