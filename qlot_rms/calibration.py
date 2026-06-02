"""End-to-end Q-LOT-RMS calibration.

Produces a :class:`RoutingPlan` (one :class:`LayerRouting` per routed layer)
containing everything the inference path needs, all frozen:

  * routing mask + [FP, INT] permutation + FP/INT indices  (SADND / random / magnitude)
  * GroupRMS group layout
  * scalar mean RMS scale ``mu_g`` (for mean-scale compensation at packing)
  * frozen per-INT-channel activation scales ``a_c``

Two passes over calibration data:

  Pass 1 (per subset): collect pre-affine ``u`` for each routed layer and compute
          the relative proxy distortion ``delta_c``; aggregate mean + lambda*std
          across subsets -> ``delta_tilde`` -> channel assignment.
  Pass 2 (streaming, batched): with routing now fixed, estimate ``mu_g`` (over all
          tokens) and the frozen activation scales (over up to
          ``cfg.act_scale_max_tokens`` tokens per layer, to bound memory).

Memory note: Pass 1 holds one subset's pre-affine activations for all routed
layers at once.  For very large models reduce ``subset_size`` / number of routed
layers.  Pass 2 is batched and bounded by ``act_scale_max_tokens``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .capture import PreAffineCapture, pre_affine_normalize
from .config import QLotRmsConfig, LayerRouting, RoutingPlan
from .data import build_calibration_chunks, make_subsets, iter_batches
from .grouprms import (
    group_rms,
    group_sizes_for,
    estimate_mu_g_per_group,
    broadcast_per_group_to_channels,
)
from .quant import (
    compute_activation_scales,
    quantize_activation_int8,
    fake_quantize_weight_w8_g128,
    channel_quantile,
)
from .diagcomp import compute_alpha
from .sadnd import (
    proxy_distortion_subset,
    aggregate_distortion,
    assign_channels,
    random_routing,
    magnitude_routing,
)
from .model_integration import (
    resolve_routed_layer_indices,
    get_ln2_modules,
    find_decoder_layers,
)


@torch.no_grad()
def _run_forward(model, chunk_ids, indices, device, batch_size):
    """Run the model over the selected chunk indices (no cache, no grad)."""
    model.eval()
    for batch in iter_batches(chunk_ids, indices, batch_size=batch_size):
        model(batch.to(device), use_cache=False)


@torch.no_grad()
def _forward_batch(model, batch, device):
    """Run the model over a single explicit batch tensor [b, seq_len]."""
    model.eval()
    model(batch.to(device), use_cache=False)


def _affine_params(norm: nn.Module):
    gamma = norm.weight
    beta = getattr(norm, "bias", None)
    return gamma, beta


@torch.no_grad()
def _grms_gate_decision(
    layer_module,
    int_idx: torch.Tensor,
    u_I: torch.Tensor,
    gamma_I: torch.Tensor,
    beta_I,
    mu_g_channels: torch.Tensor,
    cfg: QLotRmsConfig,
):
    """Proxy gating: decide whether GroupRMS helps this layer's INT branch.

    Compares the INT-branch OUTPUT reconstruction error (vs the full-precision
    INT-branch output) with vs without GroupRMS, on a small token subset, for the
    routed projections (gate_proj, up_proj).  GroupRMS is enabled only if it
    reduces that error by at least ``grms_gate_margin`` (relative), averaged over
    the two projections.  This captures both the function shift (error is measured
    against the FP reference) and the quantization-MSE reduction in one number.

    Returns ``(enabled, reason, err_ptq, err_grms)``.
    """
    n = min(u_I.shape[0], cfg.grms_gate_max_tokens)
    if n == 0:
        return cfg.use_grms, "no tokens; fell back to global use_grms", None, None
    dev = layer_module.gate_proj.weight.device
    uI = u_I[:n].to(dev).float()
    gI = gamma_I.to(dev).float()
    bI = beta_I.to(dev).float() if beta_I is not None else None
    muc = mu_g_channels.to(dev).float()

    # variant activations (pre-weight)
    y_nogrms = uI * gI + (bI if bI is not None else 0.0)
    y_grms = group_rms(uI, cfg.grms_group_size, cfg.eps) * gI + (bI if bI is not None else 0.0)
    a_nogrms = compute_activation_scales(y_nogrms, cfg.p_act, cfg.qmax, cfg.eps_scale)
    a_grms = compute_activation_scales(y_grms, cfg.p_act, cfg.qmax, cfg.eps_scale)

    errs_ptq, errs_grms = [], []
    for proj_name in ("gate_proj", "up_proj"):
        W = getattr(layer_module, proj_name).weight.data
        W_I = W[:, int_idx.to(dev)].float()                  # [O, C_int]
        z_ref = torch.matmul(y_nogrms, W_I.t())              # FP INT-branch output
        # plain INT8 (no GroupRMS)
        zq = torch.matmul(
            quantize_activation_int8(y_nogrms, a_nogrms, cfg.qmax),
            fake_quantize_weight_w8_g128(W_I, cfg.w8_group_size, cfg.qmax).t(),
        )
        # GroupRMS + per-group mean-comp folded into weight columns
        W_I_mc = W_I * muc
        zg = torch.matmul(
            quantize_activation_int8(y_grms, a_grms, cfg.qmax),
            fake_quantize_weight_w8_g128(W_I_mc, cfg.w8_group_size, cfg.qmax).t(),
        )
        denom = z_ref.norm() + 1e-8
        errs_ptq.append(float((zq - z_ref).norm() / denom))
        errs_grms.append(float((zg - z_ref).norm() / denom))

    err_ptq = sum(errs_ptq) / len(errs_ptq)
    err_grms = sum(errs_grms) / len(errs_grms)
    enabled = err_grms <= err_ptq * (1.0 - cfg.grms_gate_margin)
    if enabled:
        reason = (f"GroupRMS reduces INT-branch recon error "
                  f"{err_ptq:.4f}->{err_grms:.4f} (>= {cfg.grms_gate_margin:.3f} margin)")
    else:
        reason = (f"GroupRMS does not help: recon error "
                  f"{err_ptq:.4f}->{err_grms:.4f}; routing-only kept")
    return enabled, reason, err_ptq, err_grms


@torch.no_grad()
def _int_branch_output_mse(u_small, int_idx, weight, gamma, beta, cfg):
    """Relative INT-branch output reconstruction error (plain INT8, one projection)."""
    dev = weight.device
    uI = u_small.index_select(-1, int_idx).to(dev).float()
    gI = gamma.detach().float().to(dev).index_select(0, int_idx.to(dev))
    yI = uI * gI
    if beta is not None:
        yI = yI + beta.detach().float().to(dev).index_select(0, int_idx.to(dev))
    W_I = weight.data[:, int_idx.to(dev)].float()
    z_ref = torch.matmul(yI, W_I.t())
    a = compute_activation_scales(yI, cfg.p_act, cfg.qmax, cfg.eps_scale)
    zq = torch.matmul(
        quantize_activation_int8(yI, a, cfg.qmax),
        fake_quantize_weight_w8_g128(W_I, cfg.w8_group_size, cfg.qmax).t(),
    )
    denom = z_ref.norm() + 1e-8
    return float(((zq - z_ref).norm() / denom) ** 2)


@torch.no_grad()
def _projection_bias_corr(y_final, W_eff, act_scales, cfg):
    """Per-output bias correction b = mean_t(z_fp_ref - z_quant) for one projection.

    ``y_final`` is the (unquantized) INT activation actually fed to quant; ``W_eff``
    is the effective INT weight slice (already alpha-inversed / mean-comp'd), so
    ``y_final @ W_eff^T`` equals the full-precision INT-branch output.
    """
    z_ref = torch.matmul(y_final, W_eff.t())
    zq = torch.matmul(
        quantize_activation_int8(y_final, act_scales, cfg.qmax),
        fake_quantize_weight_w8_g128(W_eff, cfg.w8_group_size, cfg.qmax).t(),
    )
    return (z_ref - zq).mean(dim=0)  # [O]


def _select_fp_ratio_error_bounded(
    routing_method, score, delta_tilde, cfg,
    u_small=None, weight=None, gamma=None, beta=None,
):
    """Pick the smallest candidate fp_ratio whose estimated error <= error_bound.

    activation_mse: mean relative proxy distortion (delta_tilde) over INT channels.
    output_mse:     relative INT-branch output reconstruction error (one projection).
    Falls back to the largest candidate if none satisfies the bound.
    """
    C = score.numel()
    cands = sorted({round(float(c), 4) for c in cfg.fp_ratio_candidates})
    errors = {}
    chosen = None
    for c in cands:
        if routing_method == "magnitude":
            _, int_idx, _, _ = magnitude_routing(score, c)
        else:  # sadnd (score == delta_tilde)
            _, int_idx, _, _ = assign_channels(score, c)
        if (cfg.error_bound_metric == "output_mse"
                and u_small is not None and weight is not None):
            err = _int_branch_output_mse(u_small, int_idx, weight, gamma, beta, cfg)
        else:
            err = float(delta_tilde[int_idx].mean()) if int_idx.numel() else 0.0
        errors[f"{c:.4f}"] = err
        if chosen is None and err <= cfg.error_bound:
            chosen = c
    if chosen is None:
        chosen = max(cands)
    return chosen, errors


@torch.no_grad()
def calibrate(
    model: nn.Module,
    tokenizer,
    cfg: QLotRmsConfig,
    device: str = "cpu",
    routing_method: str = "sadnd",
    allow_synthetic: bool = False,
    batch_size: int = 4,
    verbose: bool = False,
) -> RoutingPlan:
    """Run calibration and return a frozen :class:`RoutingPlan`.

    ``routing_method`` in {"sadnd", "random", "magnitude"} selects how FP/INT
    channels are chosen; all methods use the same K_F and the same downstream
    GroupRMS / mu_g / activation-scale estimation.
    """
    cfg.validate()
    if routing_method == "output_aware_sadnd":
        raise NotImplementedError(
            "routing_score='output_aware_sadnd' is a stub; use 'sadnd' or 'magnitude'.")
    if routing_method not in ("sadnd", "random", "magnitude"):
        raise ValueError(f"unknown routing_method {routing_method!r}")

    def _log(m):
        if verbose:
            print(f"[qlot-rms:calib] {m}")

    indices = resolve_routed_layer_indices(model, cfg)
    ln2 = get_ln2_modules(model, indices)
    affine = {i: _affine_params(ln2[i]) for i in indices}

    chunks = build_calibration_chunks(
        tokenizer,
        seq_len=cfg.calibration_seq_len,
        num_chunks=cfg.calibration_samples,
        seed=cfg.seed,
        allow_synthetic=allow_synthetic,
    )
    num_chunks = chunks.shape[0]
    subsets = make_subsets(num_chunks, cfg.num_calib_subsets, cfg.subset_size, cfg.seed)
    _log(f"routed layers={indices}, chunks={num_chunks}, subsets={len(subsets)}")

    cap = PreAffineCapture(ln2, store_device="cpu").attach()

    # ----------------------------------------------------------------- #
    # Pass 1: per-subset proxy distortion + running channel magnitude
    # ----------------------------------------------------------------- #
    per_subset_delta: Dict[int, List[torch.Tensor]] = {i: [] for i in indices}
    absmean_sum: Dict[int, torch.Tensor] = {}
    absmean_cnt: Dict[int, int] = {i: 0 for i in indices}
    # small full-channel u subset retained only for output_mse FP-budget selection
    need_u_small = (cfg.fp_budget_mode == "error_bounded"
                    and cfg.error_bound_metric == "output_mse")
    u_small: Dict[int, torch.Tensor] = {}

    for s_i, subset in enumerate(subsets):
        cap.reset()
        _run_forward(model, chunks, subset, device, batch_size)
        u_by_layer = cap.collect()
        for i in indices:
            u = u_by_layer[i]  # [tokens, C] fp32 on cpu
            per_subset_delta[i].append(
                proxy_distortion_subset(u, cfg.p_proxy, cfg.qmax, cfg.eps)
            )
            a = u.abs().sum(dim=0)
            absmean_sum[i] = a if i not in absmean_sum else absmean_sum[i] + a
            absmean_cnt[i] += u.shape[0]
            if need_u_small and i not in u_small:
                u_small[i] = u[:512].clone()
        _log(f"pass1 subset {s_i + 1}/{len(subsets)} done")

    # build routing per layer (with optional per-layer error-bounded FP budget)
    layer_modules = find_decoder_layers(model)
    routing_idx: Dict[int, dict] = {}
    gen = torch.Generator().manual_seed(cfg.seed)
    for i in indices:
        stacked = torch.stack(per_subset_delta[i], dim=0)  # [S, C]
        delta_tilde = aggregate_distortion(stacked, cfg.lambda_agg)
        C = delta_tilde.numel()
        mag = absmean_sum[i] / max(1, absmean_cnt[i])
        score = mag if routing_method == "magnitude" else delta_tilde

        # choose fp_ratio for this layer
        fp_ratio_layer = cfg.fp_ratio
        fp_errors = None
        if cfg.fp_budget_mode == "error_bounded" and routing_method in ("sadnd", "magnitude"):
            gamma, beta = affine[i]
            fp_ratio_layer, fp_errors = _select_fp_ratio_error_bounded(
                routing_method, score, delta_tilde, cfg,
                u_small=u_small.get(i) if need_u_small else None,
                weight=layer_modules[i].mlp.gate_proj.weight if need_u_small else None,
                gamma=gamma, beta=beta,
            )

        if routing_method == "sadnd":
            fp_idx, int_idx, perm, mask = assign_channels(delta_tilde, fp_ratio_layer)
        elif routing_method == "magnitude":
            fp_idx, int_idx, perm, mask = magnitude_routing(mag, fp_ratio_layer)
        else:  # random
            fp_idx, int_idx, perm, mask = random_routing(C, fp_ratio_layer, gen)
        routing_idx[i] = {
            "delta_tilde": delta_tilde,
            "fp_idx": fp_idx,
            "int_idx": int_idx,
            "perm": perm,
            "mask": mask,
            "selected_fp_ratio": float(fp_ratio_layer),
            "fp_errors": fp_errors,
        }

    # ----------------------------------------------------------------- #
    # Pass 2: mu_g (all tokens, streaming) + activation scales (capped)
    # ----------------------------------------------------------------- #
    mu_group_sum: Dict[int, List[float]] = {}
    mu_group_cnt: Dict[int, List[int]] = {}
    # store PRE-affine INT activations u_I (capped) so the assembly can compute
    # both the routing-only and GroupRMS variants (for per-layer gating) and the
    # final activation scales after the gating decision.
    uI_store: Dict[int, List[torch.Tensor]] = {i: [] for i in indices}
    uI_tokens: Dict[int, int] = {i: 0 for i in indices}

    all_idx = torch.arange(num_chunks)
    for batch in iter_batches(chunks, all_idx, batch_size=batch_size):
        cap.reset()
        _forward_batch(model, batch, device)
        u_by_layer = cap.collect()
        for i in indices:
            if i not in u_by_layer:
                continue
            u = u_by_layer[i]                       # [tok, C] cpu fp32
            int_idx = routing_idx[i]["int_idx"]
            u_I = u.index_select(-1, int_idx)        # [tok, C_int], permuted order

            # per-group mu_g streaming over ALL tokens
            sums, counts = estimate_mu_g_per_group(u_I, cfg.grms_group_size, cfg.eps)
            if i not in mu_group_sum:
                mu_group_sum[i] = [0.0] * len(sums)
                mu_group_cnt[i] = [0] * len(sums)
            for g in range(len(sums)):
                mu_group_sum[i][g] += sums[g]
                mu_group_cnt[i][g] += counts[g]

            # store capped pre-affine INT activations
            if uI_tokens[i] < cfg.act_scale_max_tokens:
                take = cfg.act_scale_max_tokens - uI_tokens[i]
                u_I_keep = u_I[:take] if take < u_I.shape[0] else u_I
                uI_store[i].append(u_I_keep)
                uI_tokens[i] += u_I_keep.shape[0]

    cap.detach()
    layer_modules = find_decoder_layers(model)

    # ----------------------------------------------------------------- #
    # Assemble LayerRouting artifacts
    # ----------------------------------------------------------------- #
    layers: Dict[int, LayerRouting] = {}
    for i in indices:
        r = routing_idx[i]
        int_idx = r["int_idx"]
        c_int = int(int_idx.numel())
        gsizes = group_sizes_for(c_int, cfg.grms_group_size)
        if i in mu_group_sum:
            per_group_mu = [
                (mu_group_sum[i][g] / mu_group_cnt[i][g]) if mu_group_cnt[i][g] > 0 else 1.0
                for g in range(len(gsizes))
            ]
        else:
            per_group_mu = [1.0] * len(gsizes)
        mu_g_groups = torch.tensor(per_group_mu, dtype=torch.float32)  # [num_groups]
        mu_g_channels = broadcast_per_group_to_channels(
            per_group_mu, c_int, cfg.grms_group_size
        )
        uI = torch.cat(uI_store[i], dim=0) if uI_store[i] else torch.zeros(1, c_int)

        gamma, beta = affine[i]
        norm_type = "layernorm" if (beta is not None) else "rmsnorm"
        gamma_I = gamma.detach().float().cpu().index_select(0, int_idx)
        beta_I = beta.detach().float().cpu().index_select(0, int_idx) if beta is not None else None

        use_dc = bool(cfg.use_static_diag_comp and cfg.diag_comp_mode != "none")

        # --- per-layer GroupRMS gating decision ---
        err_ptq = err_grms = None
        if use_dc:
            grms_enabled, reason = False, "diag_comp replaces GroupRMS (Q-LOT-DC)"
        elif not cfg.use_grms:
            grms_enabled, reason = False, "use_grms=false (routing-only)"
        elif not cfg.grms_gating:
            grms_enabled, reason = True, "use_grms=true, global (no gating)"
        else:
            grms_enabled, reason, err_ptq, err_grms = _grms_gate_decision(
                layer_modules[i].mlp, int_idx, uI, gamma_I, beta_I, mu_g_channels, cfg
            )

        # base INT activation (pre-DC, post-affine, no token-dependent grms for DC)
        base_I = group_rms(uI, cfg.grms_group_size, cfg.eps) if grms_enabled else uI
        y_I = base_I * gamma_I
        if beta_I is not None:
            y_I = y_I + beta_I

        # --- Q-LOT-DC: static diagonal compensation ---
        diag_alpha = None
        if use_dc:
            a_c = channel_quantile(y_I.abs(), cfg.p_act, dim=0)  # per-INT-channel scale
            w_c = None
            if cfg.diag_comp_mode == "smoothquant_like":
                mlp = layer_modules[i].mlp
                dev = mlp.gate_proj.weight.device
                ii = int_idx.to(dev)
                w_g = mlp.gate_proj.weight.data[:, ii].abs().amax(0)
                w_u = mlp.up_proj.weight.data[:, ii].abs().amax(0)
                w_c = (w_g + w_u).float().cpu()
            diag_alpha = compute_alpha(
                cfg.diag_comp_mode, a_c, w_c, cfg.diag_comp_beta,
                cfg.diag_comp_alpha_min, cfg.diag_comp_alpha_max, cfg.eps_scale,
            )
            y_I = y_I * diag_alpha   # activation actually fed to quant

        # final activation scales reflect the actual inference path for this layer
        act_scales = compute_activation_scales(
            y_I, cfg.p_act, cfg.qmax, cfg.eps_scale, dim=0
        )

        # --- optional projection bias correction (gate / up) ---
        bias_corr_gate = bias_corr_up = None
        if cfg.use_projection_bias_correction:
            mlp = layer_modules[i].mlp
            dev = mlp.gate_proj.weight.device
            ii = int_idx.to(dev)
            yf = y_I.to(dev)
            a = act_scales.to(dev)
            inv = (1.0 / diag_alpha.to(dev)) if diag_alpha is not None else None
            for name, store in (("gate_proj", "g"), ("up_proj", "u")):
                W_eff = getattr(mlp, name).weight.data[:, ii].float()
                if inv is not None:
                    W_eff = W_eff * inv
                bc = _projection_bias_corr(yf, W_eff, a, cfg).float().cpu()
                if store == "g":
                    bias_corr_gate = bc
                else:
                    bias_corr_up = bc

        layers[i] = LayerRouting(
            layer_index=i,
            num_channels=int(r["delta_tilde"].numel()),
            k_fp=int(r["fp_idx"].numel()),
            fp_indices=r["fp_idx"].cpu(),
            int_indices=int_idx.cpu(),
            perm=r["perm"].cpu(),
            mask=r["mask"].cpu(),
            delta_tilde=r["delta_tilde"].cpu(),
            grms_group_size=cfg.grms_group_size,
            grms_num_groups=len(gsizes),
            grms_group_sizes=gsizes,
            mu_g=mu_g_groups.cpu(),
            mu_g_channels=mu_g_channels.cpu(),
            act_scales=act_scales.cpu(),
            routed_projections=["gate_proj", "up_proj"],
            norm_type=norm_type,
            mean_comp_applied=bool(cfg.use_mean_comp and grms_enabled),
            grms_enabled=grms_enabled,
            grms_gate_reason=reason,
            grms_proxy_err_ptq=err_ptq,
            grms_proxy_err_grms=err_grms,
            diag_comp_applied=use_dc,
            diag_alpha=diag_alpha.cpu() if diag_alpha is not None else None,
            selected_fp_ratio=r.get("selected_fp_ratio"),
            fp_budget_errors=r.get("fp_errors"),
            bias_corr_gate=bias_corr_gate,
            bias_corr_up=bias_corr_up,
        )
        _log(
            f"layer {i}: C={layers[i].num_channels} K_F={layers[i].k_fp} "
            f"fp_ratio={r.get('selected_fp_ratio')} grms={grms_enabled} dc={use_dc} "
            f"mu_g[mean]={float(mu_g_groups.mean()):.4f}"
        )

    return RoutingPlan(config=cfg, layers=layers)
