"""StaticScale public API import surface + legacy compatibility."""


def test_top_level_import():
    import staticscale as ss
    assert ss.__version__ == "0.1.0"
    for name in ["StaticScaleConfig", "LayerRouting", "RoutingPlan", "calibrate",
                 "patch_model", "unpatch_model", "equal_budget_accept_only_select"]:
        assert hasattr(ss, name), name


def test_public_submodules_import():
    from staticscale.config import StaticScaleConfig
    from staticscale.sadnd import compute_output_aware_sadnd_score, select_fp_int_channels
    from staticscale.budget import compute_cascade_error, allocate_fp_budget_from_scores
    from staticscale.mask_refinement import build_boundary_swap_candidates, greedy_refine_fp_mask
    from staticscale.clip_gain import build_int_groups, tune_layer, fit_int_output_gain
    from staticscale.packing import build_packing_aware_int_permutation
    from staticscale.projection import PackedProjection, get_backend
    from staticscale.model_integration import patch_model, unpatch_model, StaticScaleFFN
    from staticscale.selection import equal_budget_accept_only_select
    from staticscale.utils import quantize_activation_int8, fake_quantize_weight_w8_g128
    assert callable(build_int_groups) and callable(tune_layer)


def test_config_alias_is_internal_config():
    from staticscale.config import StaticScaleConfig
    from qlot_rms.config import QLotRmsConfig          # legacy path still works
    assert StaticScaleConfig is QLotRmsConfig


def test_legacy_imports_still_work():
    # legacy/internal package remains importable during the transition
    from qlot_rms.calibration import calibrate
    from qlot_rms.model_integration import patch_model, unpatch_model
    import staticscale
    assert staticscale.calibrate is calibrate
    assert staticscale.patch_model is patch_model
