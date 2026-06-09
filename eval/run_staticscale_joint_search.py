"""StaticScale joint mask-scale search runner.

Tests whether *jointly* optimizing the FP mask and the static clip-scale tuning
beats the additive ``CAP+ + clip`` baseline, addressing the reviewer concern that
the additive full pipeline adds only a tiny gain over ``CAP+ + clip``.

Selection discipline (enforced in code):
  * D_calib : proxy fitting + local accept (calibration data).
  * D_sel   : candidate SELECTION only.
  * D_eval  : final REPORTING only. Never read during selection.

Stages
------
quick     : cheap per-layer proxy search (additive vs joint orderings) on D_calib;
            promote joint candidates that improve the proxy to a D_sel PPL check.
full      : evaluate the surviving candidates + all baselines on D_eval; emit the
            ablation table and delta-vs-(CAP+ + clip).
multiseed : run seeds 0,1,2 for the best joint candidate and the CAP+ + clip
            baseline; report mean/std/clear count.
ablation  : assemble the clean ablation table from existing result files only.

Example
-------
    python eval/run_staticscale_joint_search.py \\
      --config configs/staticscale_joint_qwen25_7b_fp020.json \\
      --device cuda:1 --seed 0 --stage quick \\
      --out_root results/staticscale_joint_qwen25_7b_fp020_seed0

No speedup is claimed (``torch_reference`` is correctness-only).
"""
import argparse
import csv
import json
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from staticscale import StaticScaleConfig, calibrate, patch_model, unpatch_model  # noqa: E402
from staticscale import joint_mask_scale_search as J  # noqa: E402
from eval.eval_perplexity import load_model, wikitext2_ppl  # noqa: E402

# config fields consumed by the orchestration (not part of StaticScaleConfig)
_ORCH_KEYS = {
    "_comment", "model", "no_download", "seeds", "calib_chunks", "calib_seq_len",
    "sel_chunks", "sel_seq_len", "eval_chunks", "eval_seq_len", "candidate_families",
    "joint_search", "target_delta_vs_capplus_clip_seed0",
    "target_delta_vs_capplus_clip_multiseed", "selection_split", "eval_split",
}

FAMILY_TO_MODE = {
    "joint_swap_then_retune": ("joint_swap_then_retune", "all"),
    "joint_retune_then_swap": ("joint_retune_then_swap", "all"),
    "cascade_joint_top_layers": ("joint_retune_then_swap", 12),
    "hard_layer_extra_search": ("joint_retune_then_swap", "all"),
    "group_boundary_rebuild": ("joint_retune_then_swap", "all"),
}


# --------------------------------------------------------------------------- #
def _scfg(cfg_dict, fp, seed, overrides=None):
    """Build a StaticScaleConfig from the config dict + family overrides."""
    base = {k: v for k, v in cfg_dict.items() if k not in _ORCH_KEYS}
    base["fp_ratio"] = fp
    base["global_fp_budget_ratio"] = fp
    base["seed"] = seed
    base["calibration_samples"] = cfg_dict.get("calib_chunks", base.get("calibration_samples", 128))
    base["calibration_seq_len"] = cfg_dict.get("calib_seq_len", base.get("calibration_seq_len", 512))
    if overrides:
        base.update(overrides)
    return StaticScaleConfig.from_dict(base)


def _maybe_offline(cfg_dict):
    if cfg_dict.get("no_download"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _ppl_sel(model, tok, device, cfg_dict):
    return wikitext2_ppl(model, tok, device, cfg_dict.get("sel_seq_len", 512),
                         cfg_dict.get("sel_chunks", 32))[0]


def _ppl_eval(model, tok, device, cfg_dict):
    return wikitext2_ppl(model, tok, device, cfg_dict.get("eval_seq_len", 2048),
                         cfg_dict.get("eval_chunks", 64))[0]


def _eval_plan_ppl(model, tok, device, scfg, plan, cfg_dict, split):
    """Patch the model with ``plan``, measure PPL on the requested split, unpatch."""
    h = patch_model(model, plan, scfg)
    try:
        if split == "D_sel":
            return _ppl_sel(model, tok, device, cfg_dict)
        return _ppl_eval(model, tok, device, cfg_dict)
    finally:
        unpatch_model(h)


def _baseline_plan(model, tok, device, cfg_dict, fp, seed, family, batch_size):
    ov = J.family_config_overrides(family, fp)
    scfg = _scfg(cfg_dict, fp, seed, ov)
    plan = calibrate(model, tok, scfg, device=device,
                     allow_synthetic=False, batch_size=batch_size)
    return scfg, plan


def _joint_plan(model, tok, device, cfg_dict, fp, seed, family, batch_size):
    mode, top_k = FAMILY_TO_MODE[family]
    scfg = _scfg(cfg_dict, fp, seed)  # full StaticScale fields (cascade+refine+gt)
    plan = J.build_joint_plan(model, tok, scfg, mode=mode, device=device,
                              batch_size=batch_size, allow_synthetic=False,
                              top_k_layers=top_k)
    return scfg, plan


# pure baselines (never count as an "improvement"); capplus_clip is the reference
PURE_BASELINES = ("clean_sadnd", "clean_sadnd_clip", "clean_sadnd_clip_only",
                  "capplus", "capplus_clip")


def _build_family_plan(model, tok, device, cfg_dict, fp, seed, family, batch_size):
    """Return (scfg, plan) for any family. (None, None) for the block-output
    meta-selector, which is resolved from already-evaluated candidates in stage_full."""
    if family == "block_output_proxy_selection":
        return None, None
    if family in J.JOINT_FAMILIES:
        return _joint_plan(model, tok, device, cfg_dict, fp, seed, family, batch_size)
    if family == "hard_layer_focused_refinement":
        from staticscale import hard_layer_search as HLS
        scfg = _scfg(cfg_dict, fp, seed)
        plan = HLS.build_hard_layer_plan(model, tok, scfg, top_k=8, move_frac=0.2,
                                         device=device, batch_size=batch_size)
        return scfg, plan
    if family == "harmful_int_channel_mining":
        from staticscale import harmful_int_mining as HIM
        scfg = _scfg(cfg_dict, fp, seed)
        plan = HIM.build_harmful_int_plan(model, tok, scfg, max_swaps=16, pool=64,
                                          device=device, batch_size=batch_size)
        return scfg, plan
    # baseline / group_type_tau_grid / full_staticscale via calibrate overrides
    return _baseline_plan(model, tok, device, cfg_dict, fp, seed, family, batch_size)


def _is_joint(family):
    return family in J.JOINT_FAMILIES


# --------------------------------------------------------------------------- #
def stage_quick(model, tok, args, cfg_dict, fp, seed):
    """Per-layer proxy search (D_calib) + promote improving joint candidates to D_sel."""
    os.makedirs(args.out_root, exist_ok=True)
    js = cfg_dict.get("joint_search", {})
    tau_grid = js.get("tau_grid", list(cfg_dict.get("gt_clip_candidates", [0.75, 1.0, 1.25, 1.5, 2.0])))
    n_layers = int(js.get("proxy_layer_sample", 12))
    improve_thr = float(js.get("proxy_improve_threshold", 1e-4))
    promote_n = int(js.get("promote_top_n_to_dsel", 3))
    batch_size = args.calib_batch_size

    # --- CAP+ base routing (refine OFF, clip OFF) to get fp0/int0 per layer ---
    scfg_base = _scfg(cfg_dict, fp, seed, J.family_config_overrides("capplus", fp))
    t0 = time.time()
    base_plan = calibrate(model, tok, scfg_base, device=args.device,
                          allow_synthetic=False, batch_size=batch_size)
    y_by = J.capture_affine_activations(
        model, tok, _scfg(cfg_dict, fp, seed), device=args.device,
        batch_size=batch_size, max_tokens=cfg_dict.get("gt_max_tokens", 256))

    indices = sorted(base_plan.layers.keys())
    prio = {i: (base_plan.layers[i].budget_score
                if base_plan.layers[i].budget_score is not None
                else float(base_plan.layers[i].k_fp)) for i in indices}
    sample = sorted(indices, key=lambda i: -float(prio[i]))[:n_layers]

    proxy_families = ["capplus", "capplus_clip", "capplus_refine",
                      "capplus_refine_clip", "joint_retune_then_swap",
                      "joint_swap_then_retune"]
    agg = {f: {"proxy": [], "swaps": []} for f in proxy_families}
    for i in sample:
        lr = base_plan.layers[i]
        mlp = J._get_mlp(model, i)
        score = J._output_aware_score(lr.delta_tilde, mlp, args.device)
        res = J.evaluate_layer_orderings(
            y_by[i].to(args.device), mlp, lr.fp_indices, lr.int_indices, score,
            tau_grid=tau_grid, group_size=scfg_base.gt_group_size,
            granularity=scfg_base.gt_clip_granularity,
            refine_margin=scfg_base.fp_refine_margin,
            max_swaps=scfg_base.fp_refine_max_swaps_per_layer,
            candidate_pool=scfg_base.fp_refine_candidate_pool,
            qmax=scfg_base.qmax, eps=scfg_base.eps_scale, p_act=scfg_base.p_act,
            families=proxy_families)
        for f in proxy_families:
            agg[f]["proxy"].append(res[f]["proxy"])
            agg[f]["swaps"].append(res[f]["num_swaps"])

    mean_proxy = {f: (sum(agg[f]["proxy"]) / len(agg[f]["proxy"])) for f in proxy_families}
    mean_swaps = {f: (sum(agg[f]["swaps"]) / len(agg[f]["swaps"])) for f in proxy_families}
    additive = mean_proxy[J.ADDITIVE_REFERENCE]

    # write candidates_proxy.csv
    pcsv = os.path.join(args.out_root, "candidates_proxy.csv")
    with open(pcsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "mean_proxy", "mean_swaps",
                    "delta_proxy_vs_additive_full", "is_joint"])
        for fam in proxy_families:
            w.writerow([fam, f"{mean_proxy[fam]:.8f}", f"{mean_swaps[fam]:.2f}",
                        f"{mean_proxy[fam] - additive:+.8f}", _is_joint(fam)])

    # promotion decision: joint families that beat additive proxy by threshold
    joint_proxy = [(f, mean_proxy[f]) for f in proxy_families if _is_joint(f)]
    promote = [f for f, p in sorted(joint_proxy, key=lambda x: x[1])
               if p < additive - improve_thr][:promote_n]

    lines = [
        f"# StaticScale joint quick search  fp={fp} seed={seed}",
        f"# layers sampled (by CAP+ budget priority): {len(sample)} of {len(indices)}",
        f"# elapsed_calib_capture={time.time()-t0:.1f}s",
        "",
        "Per-layer mean unified proxy (MLP-output rel-L2, lower=better):",
    ]
    for fam in proxy_families:
        lines.append(f"  {fam:28s} proxy={mean_proxy[fam]:.8f} "
                     f"swaps={mean_swaps[fam]:.2f} dVS_additive={mean_proxy[fam]-additive:+.8f}")
    lines += [
        "",
        f"Additive reference ({J.ADDITIVE_REFERENCE}) proxy = {additive:.8f}",
        f"Joint candidates beating additive by > {improve_thr:g}: "
        f"{promote if promote else 'NONE'}",
    ]

    sel_rows = []
    if not promote:
        lines += [
            "",
            "PROXY GATE: FAILED. No joint ordering improves the calibration proxy "
            "over the additive full pipeline by the threshold.",
            "Per cheap-first policy: skip D_sel/D_eval; the additional gain is "
            "scale-driven, consistent with the reviewer concern. Prepare honest "
            "reframe (StaticScale = clip-scale tuning dominant).",
        ]
    else:
        lines += ["", f"PROXY GATE: PASSED for {promote}. Evaluating on D_sel..."]
        # baselines anchored on D_sel + promoted joint candidates
        dsel_families = ["clean_sadnd", "capplus", "capplus_clip",
                         "capplus_refine_clip"] + promote
        ref_ppls = {}
        for fam in dsel_families:
            try:
                if _is_joint(fam):
                    scfg, plan = _joint_plan(model, tok, args.device, cfg_dict, fp, seed, fam, batch_size)
                else:
                    scfg, plan = _baseline_plan(model, tok, args.device, cfg_dict, fp, seed, fam, batch_size)
                ppl = _eval_plan_ppl(model, tok, args.device, scfg, plan, cfg_dict, "D_sel")
                ref_ppls[fam] = ppl
                sel_rows.append((fam, ppl, _is_joint(fam)))
                lines.append(f"  D_sel PPL  {fam:28s} = {ppl:.6f}")
            except Exception as e:  # noqa: BLE001
                lines.append(f"  D_sel PPL  {fam:28s} = FAILED ({e})")
        cc = ref_ppls.get("capplus_clip")
        if cc is not None:
            for fam in promote:
                if fam in ref_ppls:
                    lines.append(f"  delta {fam} vs capplus_clip (D_sel) = "
                                 f"{ref_ppls[fam]-cc:+.6f}")
        scsv = os.path.join(args.out_root, "candidates_sel.csv")
        with open(scsv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["family", "d_sel_ppl", "is_joint", "delta_vs_capplus_clip"])
            for fam, ppl, isj in sel_rows:
                d = (ppl - cc) if cc is not None else ""
                w.writerow([fam, f"{ppl:.6f}", isj,
                            (f"{d:+.6f}" if d != "" else "")])

    with open(os.path.join(args.out_root, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return bool(promote)


CANON_LABEL = {
    "clean_sadnd": "Clean SADND",
    "clean_sadnd_clip": "Clean SADND + clip",
    "clean_sadnd_clip_only": "Clean SADND + clip",
    "capplus": "CAP+",
    "capplus_clip": "CAP+ + clip",
    "capplus_refine": "CAP+ + mask refinement",
    "full_staticscale": "Full StaticScale (CAP+ + refine + clip)",
    "capplus_refine_clip": "Full StaticScale (CAP+ + refine + clip)",
    "hard_layer_focused_refinement": "Hard-layer focused refinement",
    "harmful_int_channel_mining": "Harmful-INT channel mining",
    "group_type_tau_grid": "Group-type tau grid",
    "block_output_proxy_selection": "Block-output-proxy selection",
}


def stage_full(model, tok, args, cfg_dict, fp, seed):
    """Evaluate the configured candidate families on D_sel (selection) and D_eval
    (reporting). Selection of the best *improvement* family uses D_sel only; D_eval is
    reported and never used to select. Emits candidates_sel/eval, ablation, best_candidate."""
    os.makedirs(args.out_root, exist_ok=True)
    batch_size = args.calib_batch_size
    fams = list(cfg_dict.get("candidate_families", [
        "clean_sadnd", "clean_sadnd_clip", "capplus", "capplus_clip", "full_staticscale",
        "hard_layer_focused_refinement", "harmful_int_channel_mining",
        "block_output_proxy_selection", "group_type_tau_grid"]))
    # ensure reference baselines are present (needed for deltas + ablation)
    for must in ("clean_sadnd", "capplus", "capplus_clip", "full_staticscale"):
        if must not in fams:
            fams.append(must)

    lines = [f"# StaticScale FULL fp={fp} seed={seed}",
             "# selection split = D_sel; reporting split = D_eval (never used to select)"]
    sel, evl = {}, {}
    for fam in fams:
        if fam == "block_output_proxy_selection":
            continue  # resolved below as a D_sel meta-selector
        try:
            scfg, plan = _build_family_plan(model, tok, args.device, cfg_dict, fp, seed, fam, batch_size)
            if plan is None:
                continue
            sel[fam] = _eval_plan_ppl(model, tok, args.device, scfg, plan, cfg_dict, "D_sel")
            evl[fam] = _eval_plan_ppl(model, tok, args.device, scfg, plan, cfg_dict, "D_eval")
            lines.append(f"  {fam:34s} D_sel={sel[fam]:.6f}  D_eval={evl[fam]:.6f}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"  {fam:34s} FAILED ({e})")

    improvement = [f for f in fams if f not in PURE_BASELINES
                   and f != "block_output_proxy_selection" and f in evl]
    # block-output-proxy meta-selector: pick best improvement family by D_sel
    meta_choice = None
    if "block_output_proxy_selection" in fams and improvement:
        meta_choice = min(improvement, key=lambda f: sel[f])
        sel["block_output_proxy_selection"] = sel[meta_choice]
        evl["block_output_proxy_selection"] = evl[meta_choice]
        lines.append(f"  block_output_proxy_selection -> {meta_choice} "
                     f"(D_sel-selected; D_eval={evl[meta_choice]:.6f})")

    cc = evl.get("capplus_clip")
    cs = evl.get("clean_sadnd")
    cp = evl.get("capplus")
    cc_sel = sel.get("capplus_clip")

    # candidates_sel.csv (selection metric)
    with open(os.path.join(args.out_root, "candidates_sel.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "d_sel_ppl", "delta_vs_capplus_clip_dsel", "is_improvement"])
        for fam in fams:
            if fam in sel:
                d = (sel[fam] - cc_sel) if cc_sel is not None else ""
                w.writerow([fam, f"{sel[fam]:.6f}",
                            (f"{d:+.6f}" if d != "" else ""), fam in improvement])

    # candidates_eval.csv (reporting metric)
    with open(os.path.join(args.out_root, "candidates_eval.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "d_eval_ppl", "delta_vs_clean_sadnd",
                    "delta_vs_capplus", "delta_vs_capplus_clip", "is_improvement"])
        for fam in fams:
            if fam in evl:
                p = evl[fam]
                w.writerow([fam, f"{p:.6f}",
                            (f"{p-cs:+.6f}" if cs is not None else ""),
                            (f"{p-cp:+.6f}" if cp is not None else ""),
                            (f"{p-cc:+.6f}" if cc is not None else ""),
                            fam in improvement])

    # ablation_table.csv (canonical, real values only)
    with open(os.path.join(args.out_root, "ablation_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "d_eval_ppl", "delta_vs_capplus_clip"])
        seen = set()
        for fam in ["clean_sadnd", "clean_sadnd_clip", "capplus", "capplus_clip",
                    "capplus_refine", "full_staticscale", "hard_layer_focused_refinement",
                    "harmful_int_channel_mining", "group_type_tau_grid"]:
            if fam in evl and CANON_LABEL[fam] not in seen:
                d = (evl[fam] - cc) if cc is not None else ""
                w.writerow([CANON_LABEL[fam], f"{evl[fam]:.6f}",
                            (f"{d:+.6f}" if d != "" else "")])
                seen.add(CANON_LABEL[fam])

    # selection (D_sel only) of the best improvement family; report its D_eval delta
    target = cfg_dict.get("target_delta_vs_capplus_clip_seed0",
                          cfg_dict.get("target_delta_vs_capplus_clip_multiseed", -0.003))
    selected, sel_delta_eval = None, None
    if improvement and cc_sel is not None:
        selected = min(improvement, key=lambda f: sel[f])  # SELECTION on D_sel only
        if cc is not None:
            sel_delta_eval = evl[selected] - cc
            lines += ["", f"Selected improvement family (by D_sel): {selected}",
                      f"  -> D_eval delta vs CAP+ + clip = {sel_delta_eval:+.6f} "
                      f"(target <= {target})",
                      f"  TARGET {'MET' if sel_delta_eval <= target else 'NOT met'}"]
    else:
        lines += ["", "No improvement family evaluated.",
                  "Conclusion so far: improvement is scale-driven (CAP+ + clip dominant)."]

    with open(os.path.join(args.out_root, "best_candidate.json"), "w") as f:
        json.dump({"fp_ratio": fp, "seed": seed, "d_eval": evl, "d_sel": sel,
                   "selected_by_dsel": selected,
                   "selected_delta_vs_capplus_clip_eval": sel_delta_eval,
                   "block_output_meta_choice": meta_choice,
                   "target_delta": target}, f, indent=2)
    with open(os.path.join(args.out_root, "summary.txt"), "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def stage_multiseed(model, tok, args, cfg_dict, fp):
    os.makedirs(args.out_root, exist_ok=True)
    seeds = cfg_dict.get("seeds", [0, 1, 2])
    batch_size = args.calib_batch_size
    # best improvement family from a prior full run's best_candidate.json if available
    best_fam = "full_staticscale"
    bc = os.path.join(args.out_root, "best_candidate.json")
    if os.path.exists(bc):
        d = json.load(open(bc))
        best_fam = d.get("selected_by_dsel") or d.get("best_joint_family") or best_fam
    rows, deltas = [], []
    for s in seeds:
        try:
            scfg_b, plan_b = _baseline_plan(model, tok, args.device, cfg_dict, fp, s, "capplus_clip", batch_size)
            cc = _eval_plan_ppl(model, tok, args.device, scfg_b, plan_b, cfg_dict, "D_eval")
            scfg_j, plan_j = _build_family_plan(model, tok, args.device, cfg_dict, fp, s, best_fam, batch_size)
            jp = _eval_plan_ppl(model, tok, args.device, scfg_j, plan_j, cfg_dict, "D_eval")
            rows.append((s, cc, jp, jp - cc))
            deltas.append(jp - cc)
        except Exception as e:  # noqa: BLE001
            rows.append((s, None, None, None))
            print(f"seed {s} FAILED: {e}")
    mcsv = os.path.join(args.out_root, "multiseed_summary.csv")
    with open(mcsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "capplus_clip", f"joint({best_fam})", "delta_vs_capplus_clip"])
        for s, cc, jp, d in rows:
            w.writerow([s, (f"{cc:.6f}" if cc else ""), (f"{jp:.6f}" if jp else ""),
                        (f"{d:+.6f}" if d is not None else "")])
        if deltas:
            target = cfg_dict.get("target_delta_vs_capplus_clip_multiseed", -0.003)
            w.writerow(["AGG", "", "",
                        f"mean={st.mean(deltas):+.6f} std={st.pstdev(deltas):.6f} "
                        f"clear={sum(d <= target for d in deltas)}/{len(deltas)}"])
    print(f"wrote {mcsv}")


def stage_ablation(args, cfg_dict):
    """Assemble the ablation table from existing result files only (no model run)."""
    src = os.path.join(args.out_root, "candidates_eval.csv")
    if not os.path.exists(src):
        print(f"No {src}; run --stage full first. Ablation uses real result files only.")
        return
    print(f"Ablation table source: {os.path.join(args.out_root, 'ablation_table.csv')}")
    with open(os.path.join(args.out_root, "ablation_table.csv")) as f:
        print(f.read())


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", choices=["quick", "full", "multiseed", "ablation"],
                    default="quick")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--calib_batch_size", type=int, default=4)
    ap.add_argument("--model", default=None, help="override config model id")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing result files in out_root")
    args = ap.parse_args()

    cfg_dict = json.load(open(args.config))
    _maybe_offline(cfg_dict)
    fp = float(cfg_dict.get("fp_ratio", 0.20))
    model_id = args.model or cfg_dict.get("model", "Qwen/Qwen2.5-7B")

    # overwrite guard
    summary = os.path.join(args.out_root, "summary.txt")
    if os.path.exists(summary) and not args.overwrite and args.stage in ("quick",):
        print(f"{summary} exists; pass --overwrite to re-run (refusing to clobber).")
        return

    if args.stage == "ablation":
        stage_ablation(args, cfg_dict)
        return

    print(f"[load] {model_id} on {args.device} (offline={cfg_dict.get('no_download')})")
    model, tok = load_model(model_id, args.device)

    if args.stage == "quick":
        stage_quick(model, tok, args, cfg_dict, fp, args.seed)
    elif args.stage == "full":
        stage_full(model, tok, args, cfg_dict, fp, args.seed)
    elif args.stage == "multiseed":
        stage_multiseed(model, tok, args, cfg_dict, fp)


if __name__ == "__main__":
    main()
