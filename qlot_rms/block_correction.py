"""Q-LOT-OBC: Output-aware Block Correction (final MLP-output correction).

Motivation: per-projection (gate/up) correction cannot capture the
``SiLU(gate) * up`` nonlinearity, so it was quality-neutral on Qwen2.5-7B. Here we
correct the WHOLE MLP block output (after down_proj) directly against the FP16
block output, which is more directly tied to model error.

For each routed layer, on calibration data:
  h    = MLP block input (pre-LN2 residual hidden; what QLotRmsFFN sees)
  y_fp = FP16 MLP block output (original mlp on LN2(h))
  y_q  = quantized routed MLP block output (QLotRmsFFN(h), no block correction yet)
  E    = y_fp - y_q

Fit one static correction (bias | affine | lowrank), then ACCEPT it for that
layer only if it lowers the MLP-output MSE by the margin (accept-only gating).
Everything is static and stored; inference applies it after down_proj.
No speedup is claimed; torch_reference correctness-only.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from .config import QLotRmsConfig, RoutingPlan
from .lowrank import fit_lowrank


# --------------------------------------------------------------------------- #
# Fit primitives
# --------------------------------------------------------------------------- #
def fit_bias(E: torch.Tensor) -> torch.Tensor:
    """b = mean_t(E)  -> [hidden]."""
    return E.float().mean(dim=0)


def fit_affine(y_q, y_fp, a_min=0.5, a_max=2.0, eps=1e-8):
    """Per-channel affine y_fp ~ a*y_q + b (a clamped). Returns (a [H], b [H])."""
    y_q = y_q.float(); y_fp = y_fp.float()
    mq = y_q.mean(0); mfp = y_fp.mean(0)
    var = ((y_q - mq) ** 2).mean(0) + eps
    cov = ((y_q - mq) * (y_fp - mfp)).mean(0)
    a = (cov / var).clamp(a_min, a_max)
    b = mfp - a * mq
    return a, b


def _apply(mode, y_q, h, params):
    if mode == "bias":
        return y_q + params["b"]
    if mode == "affine":
        return y_q * params["a"] + params["b"]
    if mode == "lowrank":
        return y_q + torch.matmul(torch.matmul(h, params["A"]), params["B"])
    return y_q


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _quick_ppl(model, chunks, device):
    """Mean-NLL perplexity over provided token chunks (HF loss)."""
    model.eval()
    nll, ntok = 0.0, 0
    for i in range(chunks.shape[0]):
        b = chunks[i:i + 1].to(device)
        out = model(b, labels=b)
        n = b.shape[1] - 1
        nll += float(out.loss) * n
        ntok += n
    return float(torch.exp(torch.tensor(nll / max(1, ntok))))


@torch.no_grad()
def fit_block_corrections(model, plan: RoutingPlan, cfg: QLotRmsConfig,
                          chunks: torch.Tensor, device: str = "cpu",
                          batch_size: int = 4, verbose: bool = False,
                          val_chunks: torch.Tensor = None) -> Dict[int, dict]:
    """Fit + accept-gate block corrections for every routed layer (in-place on plan).

    Returns a per-layer summary dict. Requires the model UNPATCHED (so the
    original mlp gives y_fp); builds a QLotRmsFFN per layer to get y_q.
    """
    from .model_integration import find_decoder_layers, build_qlot_ffn
    from .projection import get_backend

    def _log(m):
        if verbose:
            print(f"[qlot-obc] {m}")

    indices: List[int] = list(plan.layers.keys())
    if cfg.block_correction_scope == "selected_layers" and cfg.block_correction_max_layers:
        indices = indices[: cfg.block_correction_max_layers]
    layers = find_decoder_layers(model)
    norm_mods = {i: layers[i].post_attention_layernorm for i in indices}
    mlp_mods = {i: layers[i].mlp for i in indices}

    h_store: Dict[int, List[torch.Tensor]] = {i: [] for i in indices}
    y_store: Dict[int, List[torch.Tensor]] = {i: [] for i in indices}
    tok_cnt: Dict[int, int] = {i: 0 for i in indices}
    cap = cfg.block_correction_max_tokens
    handles = []

    def mk_norm_hook(i):
        def hook(mod, inp, out):
            if tok_cnt[i] >= cap:
                return
            h = inp[0].reshape(-1, inp[0].shape[-1])
            h_store[i].append(h.detach().to("cpu"))
        return hook

    def mk_mlp_hook(i):
        def hook(mod, inp, out):
            if tok_cnt[i] >= cap:
                return
            y = out.reshape(-1, out.shape[-1])
            y_store[i].append(y.detach().to("cpu"))
            tok_cnt[i] += y.shape[0]
        return hook

    for i in indices:
        handles.append(norm_mods[i].register_forward_hook(mk_norm_hook(i)))
        handles.append(mlp_mods[i].register_forward_hook(mk_mlp_hook(i)))

    model.eval()
    n = chunks.shape[0]
    for s in range(0, n, batch_size):
        if all(tok_cnt[i] >= cap for i in indices):
            break
        model(chunks[s:s + batch_size].to(device), use_cache=False)
    for hd in handles:
        hd.remove()

    backend = get_backend(cfg.backend)
    summary: Dict[int, dict] = {}
    mode = cfg.block_correction_mode
    for i in indices:
        if not h_store[i] or mode == "none":
            continue
        h = torch.cat(h_store[i], 0)[:cap]
        y_fp = torch.cat(y_store[i], 0)[:cap]
        ffn = build_qlot_ffn(layers[i], plan.layers[i], cfg, backend)
        dev = layers[i].down_proj.weight.device if hasattr(layers[i], "down_proj") \
            else layers[i].mlp.down_proj.weight.device
        # compute y_q in batches on device
        yq_parts = []
        for s in range(0, h.shape[0], 512):
            hb = h[s:s + 512].to(dev)
            yq_parts.append(ffn(hb).detach().float().cpu())
        y_q = torch.cat(yq_parts, 0)

        hd = layers[i].mlp.down_proj.weight.device
        hH = h.to(hd).float(); yqH = y_q.to(hd).float(); yfpH = y_fp.to(hd).float()
        before = ((yfpH - yqH) ** 2).mean().item()

        params = {}
        if mode == "bias":
            params["b"] = fit_bias(yfpH - yqH)
        elif mode == "affine":
            a, b = fit_affine(yqH, yfpH, cfg.block_affine_a_min, cfg.block_affine_a_max)
            params["a"], params["b"] = a, b
        elif mode == "lowrank":
            A, B = fit_lowrank(hH, (yfpH - yqH), cfg.block_lowrank_rank)
            params["A"], params["B"] = A.to(hd), B.to(hd)

        y_corr = _apply(mode, yqH, hH, params)
        after = ((yfpH - y_corr) ** 2).mean().item()
        enabled = after < before * (1.0 - cfg.block_correction_margin)
        reason = (f"{mode}: MSE {before:.4e}->{after:.4e} "
                  f"({'accept' if enabled else 'reject'}, margin {cfg.block_correction_margin})")

        lr = plan.layers[i]
        lr.block_corr_mode = mode
        lr.block_corr_enabled = bool(enabled)
        lr.block_mse_before = float(before)
        lr.block_mse_after = float(after)
        lr.block_corr_reason = reason
        if enabled:
            if mode == "bias":
                lr.block_bias = params["b"].cpu()
            elif mode == "affine":
                lr.block_affine_a = params["a"].cpu()
                lr.block_affine_b = params["b"].cpu()
            elif mode == "lowrank":
                lr.block_lowrank_A = params["A"].cpu()
                lr.block_lowrank_B = params["B"].cpu()
        summary[i] = {"mode": mode, "enabled": bool(enabled),
                      "mse_before": before, "mse_after": after, "reason": reason}
        _log(f"layer {i}: {reason}")

    # --- validation-PPL safeguard (catches overfitting the MSE gate misses) ---
    # The per-layer MSE gate accepts on calibration error; high-capacity
    # corrections (e.g. low-rank) can lower calib MSE yet WORSEN held-out PPL.
    # If accept_rule == "validation_ppl", verify the corrected model does not
    # worsen a small held-out PPL; otherwise disable ALL block corrections.
    if cfg.block_correction_accept_rule == "validation_ppl" and val_chunks is not None \
            and any(plan.layers[i].block_corr_enabled for i in indices):
        from .model_integration import patch_model, unpatch_model
        ppl_corr = _eval_with_block(model, plan, cfg, val_chunks, device, enabled=True)
        ppl_base = _eval_with_block(model, plan, cfg, val_chunks, device, enabled=False)
        keep = ppl_corr < ppl_base - cfg.block_correction_margin
        _log(f"validation_ppl gate: corrected={ppl_corr:.4f} baseline={ppl_base:.4f} "
             f"-> {'keep' if keep else 'DISABLE ALL (overfit/no gain)'}")
        if not keep:
            for i in indices:
                lr = plan.layers[i]
                lr.block_corr_enabled = False
                lr.block_bias = lr.block_affine_a = lr.block_affine_b = None
                lr.block_lowrank_A = lr.block_lowrank_B = None
                lr.block_corr_reason += " | global validation_ppl gate DISABLED (no PPL gain)"
                if i in summary:
                    summary[i]["enabled"] = False
        for k in summary:
            summary[k]["val_ppl_corrected"] = ppl_corr
            summary[k]["val_ppl_baseline"] = ppl_base

    return summary


@torch.no_grad()
def _eval_with_block(model, plan, cfg, val_chunks, device, enabled: bool):
    """PPL of the patched model with block corrections toggled on/off globally."""
    from .model_integration import patch_model, unpatch_model
    saved = {i: plan.layers[i].block_corr_enabled for i in plan.layers}
    for i in plan.layers:
        plan.layers[i].block_corr_enabled = bool(enabled and saved[i])
    handle = patch_model(model, plan, cfg)
    try:
        ppl = _quick_ppl(model, val_chunks, device)
    finally:
        unpatch_model(handle)
        for i in plan.layers:
            plan.layers[i].block_corr_enabled = saved[i]
    return ppl
