# StaticScale public API

Import from the `staticscale` package. (The implementation currently lives in a
legacy/internal package and is re-exported under stable names; the legacy import path
also works during the transition.)

## Top level

```python
from staticscale import (
    StaticScaleConfig,          # configuration dataclass (alias of the internal config)
    LayerRouting, RoutingPlan,  # frozen per-layer / whole-model static policy artifacts
    calibrate,                  # run the static policy search -> RoutingPlan
    patch_model, unpatch_model, # reversibly enable / restore StaticScale
    equal_budget_accept_only_select,
)
import staticscale
staticscale.__version__         # "0.1.0"
```

Typical use:

```python
cfg = StaticScaleConfig.load_json("configs/staticscale_qwen25_7b_fp020.json")
plan = calibrate(model, tokenizer, cfg, device="cuda:0")
handle = patch_model(model, plan, cfg)
# ... inference ...
unpatch_model(handle)           # restores the original modules exactly
```

## Submodules

| module | purpose | key symbols |
|---|---|---|
| `staticscale.config` | configuration + artifacts | `StaticScaleConfig`, `LayerRouting`, `RoutingPlan` |
| `staticscale.sadnd` | output-aware SADND routing | `compute_output_aware_sadnd_score`, `select_fp_int_channels`, `proxy_distortion_subset`, `aggregate_distortion` |
| `staticscale.budget` | cascade / marginal FP budget | `compute_cascade_error`, `build_cascade_budget_scores`, `allocate_fp_budget_from_scores`, `allocate_by_marginal_gain`, `allocate_global_fp_budget`, `capture_layer_errors` |
| `staticscale.mask_refinement` | equal-budget FP mask refinement | `build_boundary_swap_candidates`, `greedy_refine_fp_mask`, `refine_policy_masks`, `evaluate_mask_proxy` |
| `staticscale.clip_gain` | static groupwise clip-gain tuning | `build_int_groups`, `tune_clip_multiplier`, `fit_int_output_gain`, `tune_layer`, `build_gt_context` |
| `staticscale.packing` | packing-aware INT layout | `build_packing_aware_int_permutation`, `build_layer_fp_int_perm`, `group_sizes_for` |
| `staticscale.projection` | packed FP/INT reference projection | `PackedProjection`, `get_backend`, `TorchReferenceBackend`, `CustomPackedBackend` |
| `staticscale.model_integration` | reversible model patching | `patch_model`, `unpatch_model`, `build_staticscale_ffn`, `StaticScaleFFN` |
| `staticscale.selection` | equal-budget accept-only rule | `equal_budget_accept_only_select` |
| `staticscale.utils` | quant primitives | `quantize_activation_int8`, `fake_quantize_weight_w8_g128`, `compute_activation_scales`, `channel_quantile`, `group_sizes_for` |
| `staticscale.serving_export` | static serving-artifact export | `export_serving_artifacts` (CLI: `python -m staticscale.serving_export`) |

## Backends

`StaticScaleConfig.backend` ∈ {`torch_reference` (default, correctness-only),
`custom_packed` (stub; raises — no kernel implemented)}. No backend-independent speedup
is claimed.

## Key config fields

- routing: `routing_score`, `fp_ratio`
- budget: `fp_budget_mode`, `use_cascade_aware_budget`, `use_marginal_gain_allocation`,
  `cascade_beta`, `cascade_gamma`, `global_fp_budget_ratio`
- refinement: `use_fp_mask_refinement`, `fp_refine_method`, `fp_refine_margin`,
  `fp_refine_candidate_pool`, `fp_refine_max_swaps_per_layer`
- clip-gain: `use_groupwise_clip_gain_tuning`, `gt_tune_clip`, `gt_tune_gain`,
  `gt_clip_granularity`, `gt_gain_granularity`, `gt_group_size`, `gt_clip_candidates`,
  `gt_gain_clip_min`, `gt_gain_clip_max`, `gt_metric`, `gt_accept_margin`
- layout: `int_permutation_mode`, `w8_group_size`
- selection: `accept_only_margin`, `fp_ratio_candidates`
