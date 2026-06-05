"""SADND-CAP calibration.

Produces a frozen RoutingPlan per routed layer:
  1. SADND or output-aware SADND score (proxy distortion, optionally weighted by
     gate/up weight column norms)
  2. FP/INT selection -- fixed per-layer fp_ratio OR a global FP budget reallocated
     across layers by sensitivity
  3. packing-aware INT permutation (order INT channels by activation scale so each
     W8-G128 group is more uniform)
  4. frozen per-INT-channel activation scales

No corrections (no GroupRMS / diagonal / bias / low-rank / block). No runtime
top-k/sort. No speedup is claimed.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .capture import PreAffineCapture
from .config import QLotRmsConfig, LayerRouting, RoutingPlan
from .data import build_calibration_chunks, make_subsets, iter_batches
from .quant import compute_activation_scales
from .sadnd import proxy_distortion_subset, aggregate_distortion
from . import sadnd_cap
from .model_integration import resolve_routed_layer_indices, get_ln2_modules, find_decoder_layers


@torch.no_grad()
def _run_forward(model, chunk_ids, indices, device, batch_size):
    model.eval()
    for batch in iter_batches(chunk_ids, indices, batch_size=batch_size):
        model(batch.to(device), use_cache=False)


@torch.no_grad()
def _forward_batch(model, batch, device):
    model.eval()
    model(batch.to(device), use_cache=False)


def _affine(norm):
    return norm.weight, getattr(norm, "bias", None)


@torch.no_grad()
def calibrate(model, tokenizer, cfg: QLotRmsConfig, device: str = "cpu",
              routing_method: Optional[str] = None, allow_synthetic: bool = False,
              batch_size: int = 4, verbose: bool = False) -> RoutingPlan:
    cfg.validate()
    rs = routing_method or cfg.routing_score
    if rs not in ("sadnd", "output_aware_sadnd", "magnitude"):
        raise ValueError(f"unknown routing_score {rs!r}")

    def _log(m):
        if verbose:
            print(f"[sadnd-cap:calib] {m}")

    indices = resolve_routed_layer_indices(model, cfg)
    ln2 = get_ln2_modules(model, indices)
    layer_mods = find_decoder_layers(model)
    affine = {i: _affine(ln2[i]) for i in indices}

    chunks = build_calibration_chunks(
        tokenizer, seq_len=cfg.calibration_seq_len, num_chunks=cfg.calibration_samples,
        seed=cfg.seed, allow_synthetic=allow_synthetic)
    num_chunks = chunks.shape[0]
    subsets = make_subsets(num_chunks, cfg.num_calib_subsets, cfg.subset_size, cfg.seed)
    _log(f"routed={indices} chunks={num_chunks} subsets={len(subsets)} score={rs}")

    cap = PreAffineCapture(ln2, store_device="cpu").attach()

    # ---- Pass 1: per-subset proxy distortion + per-channel abs-mean ----
    per_subset_delta: Dict[int, List[torch.Tensor]] = {i: [] for i in indices}
    absmean_sum: Dict[int, torch.Tensor] = {}
    absmean_cnt: Dict[int, int] = {i: 0 for i in indices}
    for s_i, subset in enumerate(subsets):
        cap.reset()
        _run_forward(model, chunks, subset, device, batch_size)
        u_by = cap.collect()
        for i in indices:
            u = u_by[i]
            per_subset_delta[i].append(proxy_distortion_subset(u, cfg.p_proxy, cfg.qmax, cfg.eps))
            a = u.abs().sum(dim=0)
            absmean_sum[i] = a if i not in absmean_sum else absmean_sum[i] + a
            absmean_cnt[i] += u.shape[0]
        _log(f"pass1 subset {s_i+1}/{len(subsets)}")

    # ---- scores + per-channel packing scale ----
    delta_t: Dict[int, torch.Tensor] = {}
    scores: Dict[int, torch.Tensor] = {}
    chan_scale: Dict[int, torch.Tensor] = {}   # packing-aware sort key (length C)
    for i in indices:
        dt = aggregate_distortion(torch.stack(per_subset_delta[i], 0))
        delta_t[i] = dt
        absmean = absmean_sum[i] / max(1, absmean_cnt[i])
        gamma, _ = affine[i]
        gabs = gamma.detach().float().cpu().abs()
        chan_scale[i] = absmean * gabs                # ~ scale of affine activation
        if rs == "magnitude":
            scores[i] = absmean
        elif rs == "output_aware_sadnd":
            mlp = layer_mods[i].mlp
            wg = mlp.gate_proj.weight.data.float().norm(dim=0).cpu()
            wu = mlp.up_proj.weight.data.float().norm(dim=0).cpu()
            scores[i] = sadnd_cap.compute_output_aware_sadnd_score(dt, wg, wu)
        else:
            scores[i] = dt

    # ---- FP budget allocation ----
    if cfg.fp_budget_mode == "global":
        k_by = sadnd_cap.allocate_global_fp_budget(scores, cfg.fp_ratio)
    else:
        k_by = {i: int(cfg.fp_ratio * scores[i].numel()) for i in indices}

    # ---- per-layer FP/INT + packing-aware permutation ----
    routing_idx = {}
    for i in indices:
        fp, int_packed, perm, mask = sadnd_cap.build_layer_fp_int_perm(
            scores[i], k_by[i], chan_scale[i], cfg.int_permutation_mode, cfg.w8_group_size)
        routing_idx[i] = {"fp": fp, "int": int_packed, "perm": perm, "mask": mask,
                          "k_fp": int(fp.numel())}

    # ---- Pass 2: activation scales on the (packed) INT block ----
    cap.reset()
    yI_store: Dict[int, List[torch.Tensor]] = {i: [] for i in indices}
    yI_tok: Dict[int, int] = {i: 0 for i in indices}
    all_idx = torch.arange(num_chunks)
    for batch in iter_batches(chunks, all_idx, batch_size=batch_size):
        cap.reset()
        _forward_batch(model, batch, device)
        u_by = cap.collect()
        for i in indices:
            if i not in u_by or yI_tok[i] >= cfg.act_scale_max_tokens:
                continue
            u = u_by[i]
            int_idx = routing_idx[i]["int"]
            u_I = u.index_select(-1, int_idx)
            gamma, beta = affine[i]
            gI = gamma.detach().float().cpu().index_select(0, int_idx)
            y_I = u_I * gI
            if beta is not None:
                y_I = y_I + beta.detach().float().cpu().index_select(0, int_idx)
            take = cfg.act_scale_max_tokens - yI_tok[i]
            if take < y_I.shape[0]:
                y_I = y_I[:take]
            yI_store[i].append(y_I)
            yI_tok[i] += y_I.shape[0]
    cap.detach()

    # ---- assemble ----
    layers: Dict[int, LayerRouting] = {}
    for i in indices:
        r = routing_idx[i]
        c_int = int(r["int"].numel())
        yI = torch.cat(yI_store[i], 0) if yI_store[i] else torch.zeros(1, max(1, c_int))
        act_scales = compute_activation_scales(yI, cfg.p_act, cfg.qmax, cfg.eps_scale, dim=0)
        _, beta = affine[i]
        layers[i] = LayerRouting(
            layer_index=i, num_channels=int(scores[i].numel()), k_fp=r["k_fp"],
            fp_indices=r["fp"].cpu(), int_indices=r["int"].cpu(), perm=r["perm"].cpu(),
            mask=r["mask"].cpu(), delta_tilde=delta_t[i].cpu(), act_scales=act_scales.cpu(),
            w8_group_size=cfg.w8_group_size, routing_score=rs,
            int_permutation_mode=cfg.int_permutation_mode,
            selected_fp_ratio=r["k_fp"] / max(1, int(scores[i].numel())),
            norm_type="layernorm" if beta is not None else "rmsnorm")
        _log(f"layer {i}: C={layers[i].num_channels} K_F={r['k_fp']} perm={cfg.int_permutation_mode}")

    return RoutingPlan(config=cfg, layers=layers)
