"""Optional throughput + memory benchmark for Q-LOT-RMS.

IMPORTANT honesty constraints (enforced by this script's output):
  * The default ``torch_reference`` backend is FAKE-QUANTIZED: it dequantizes
    and runs FP matmuls.  It is a CORRECTNESS reference, not a fast path.  Any
    "throughput" measured for it reflects the reference overhead, NOT a real
    INT8 speedup.  This script therefore labels every timing with its
    ``timing_backend`` and refuses to compute a speedup ratio for the reference
    backend.
  * A real speedup may only be claimed when measured with a custom packed kernel
    backend, against the FP16 baseline, with the SAME fp_ratio, branch shapes,
    routed layers, and kernel path.  Until ``custom_packed`` is implemented this
    script reports raw wall-clock + peak memory only, with a clear caveat.

Measures (configurable batch / seq_len; default batch=1, seq_len=1024):
  * average prefill latency (single forward over the full sequence)
  * tokens/sec (batch * seq_len / latency)
  * peak CUDA memory (if CUDA)

Profiling (optional, --profile):
  * CUDA-synchronized timing, optional torch.profiler table saved to
    profile_summary.txt.

Examples
--------
    python -m eval.benchmark --config configs/qlot_rms_full.json \
        --out_dir results/qlot_rms_perf/after --batch_size 1 --seq_len 1024 \
        --variants fp16 int8_ptq sadnd sadnd_grms_meancomp
    python -m eval.benchmark --config configs/qlot_rms_full.json \
        --out_dir results/qlot_rms_perf/prof --profile --profile_steps 8 \
        --variants fp16 sadnd_grms_meancomp
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from qlot_rms.calibration import calibrate
from qlot_rms.model_integration import patch_model, unpatch_model
from eval.eval_perplexity import VARIANTS, build_cfg, load_model  # reuse


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


@torch.no_grad()
def time_prefill(model, device, seq_len, vocab, batch=1, warmup=2, iters=8):
    """CUDA-synchronized average prefill latency + peak memory."""
    ids = torch.randint(0, vocab, (batch, seq_len), device=device)
    is_cuda = str(device).startswith("cuda")
    for _ in range(warmup):
        model(ids, use_cache=False)
    _sync(device)
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True) if is_cuda else None
    end = torch.cuda.Event(enable_timing=True) if is_cuda else None
    import time
    if is_cuda:
        start.record()
    else:
        t0 = time.time()
    for _ in range(iters):
        model(ids, use_cache=False)
    if is_cuda:
        end.record()
        _sync(device)
        dt = start.elapsed_time(end) / 1e3 / iters       # ms -> s, per iter
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
    else:
        dt = (time.time() - t0) / iters
        peak_mb = None
    return dt, peak_mb


@torch.no_grad()
def profile_prefill(model, device, seq_len, vocab, batch, steps, out_path):
    """Optional torch.profiler trace; writes a summary table to ``out_path``."""
    ids = torch.randint(0, vocab, (batch, seq_len), device=device)
    model(ids, use_cache=False)  # warmup
    _sync(device)
    try:
        from torch.profiler import profile, ProfilerActivity
        acts = [ProfilerActivity.CPU]
        if str(device).startswith("cuda"):
            acts.append(ProfilerActivity.CUDA)
        with profile(activities=acts, record_shapes=False) as prof:
            for _ in range(steps):
                model(ids, use_cache=False)
            _sync(device)
        sort_key = ("cuda_time_total" if str(device).startswith("cuda")
                    else "cpu_time_total")
        table = prof.key_averages().table(sort_by=sort_key, row_limit=25)
        with open(out_path, "a") as f:
            f.write(table + "\n\n")
        return True
    except Exception as e:  # noqa: BLE001
        with open(out_path, "a") as f:
            f.write(f"[profiler unavailable: {e!r}]\n\n")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="QLotRmsConfig JSON; supplies qlot hyper-params.")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--fp_ratio", type=float, default=0.06)
    ap.add_argument("--grms_group_size", type=int, default=128)
    ap.add_argument("--lambda_agg", type=float, default=1.0)
    ap.add_argument("--p_proxy", type=float, default=0.9995)
    ap.add_argument("--p_act", type=float, default=0.999)
    ap.add_argument("--w8_group_size", type=int, default=128)
    ap.add_argument("--routed_layers", default="all")
    ap.add_argument("--calib_samples", type=int, default=64)
    ap.add_argument("--calib_seq_len", type=int, default=512)
    ap.add_argument("--calib_subsets", type=int, default=3)
    ap.add_argument("--subset_size", type=int, default=16)
    ap.add_argument("--calib_batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="torch_reference")
    ap.add_argument("--variants", nargs="+",
                    default=["fp16", "sadnd_grms_meancomp"])
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--cache_dequant", choices=["auto", "on", "off"], default="auto",
                    help="override cfg.cache_dequant_weight (reference-only opt).")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile_steps", type=int, default=8)
    ap.add_argument("--out_dir", default="eval/bench_results")
    ap.add_argument("--allow_synthetic", action="store_true")
    args = ap.parse_args()

    # custom_packed is experimental: fail clearly up-front, never fake results.
    if args.backend == "custom_packed":
        from qlot_rms.projection import CustomPackedBackend
        if not CustomPackedBackend.available():
            print("[bench] ERROR: backend='custom_packed' is experimental and no "
                  "working Triton/CUDA kernel is available. Refusing to fake "
                  "results. Use --backend torch_reference.")
        else:
            print("[bench] ERROR: custom_packed is experimental and NOT wired into "
                  "the full model forward yet (only the one-projection "
                  "packed_forward prototype is correctness-tested). Refusing to "
                  "fake end-to-end results. Use --backend torch_reference.")
        sys.exit(2)

    os.makedirs(args.out_dir, exist_ok=True)
    model, tok = load_model(args.model, args.device)
    vocab = model.config.vocab_size
    dtype = str(next(model.parameters()).dtype)
    prof_path = os.path.join(args.out_dir, "profile_summary.txt")
    if args.profile and os.path.exists(prof_path):
        os.remove(prof_path)

    rows = []
    for name in args.variants:
        method, overrides = VARIANTS[name]
        n_routed, fp_ratio_eff, grms_gs = None, None, None
        timing_backend = "fp16" if name == "fp16" else args.backend
        if name != "fp16":
            cfg = build_cfg(args, method, overrides)
            if args.cache_dequant != "auto":
                cfg.cache_dequant_weight = (args.cache_dequant == "on")
            plan = calibrate(model, tok, cfg, device=args.device,
                             routing_method=method, allow_synthetic=args.allow_synthetic,
                             batch_size=args.calib_batch_size)
            handle = patch_model(model, plan, cfg)
            n_routed = len(plan.layers)
            fp_ratio_eff = cfg.fp_ratio
            grms_gs = cfg.grms_group_size
        else:
            handle = None
        try:
            if args.profile:
                with open(prof_path, "a") as f:
                    f.write(f"===== variant={name} backend={timing_backend} =====\n")
                profile_prefill(model, args.device, args.seq_len, vocab,
                                args.batch_size, args.profile_steps, prof_path)
            dt, peak = time_prefill(model, args.device, args.seq_len, vocab,
                                    batch=args.batch_size, warmup=args.warmup,
                                    iters=args.iters)
        finally:
            if handle is not None:
                unpatch_model(handle)
        tok_s = (args.batch_size * args.seq_len / dt) if dt else None
        rows.append({
            "variant": name,
            "timing_backend": timing_backend,
            "avg_latency_ms": dt * 1e3,
            "prefill_s": dt,
            "tokens_per_s": tok_s,
            "peak_mb": peak,
            "routed_layers": n_routed,
            "fp_ratio": fp_ratio_eff,
            "grms_group_size": grms_gs,
        })
        print(f"[bench] {name:22s} {dt*1e3:8.2f}ms  {tok_s and round(tok_s):>7} tok/s  "
              f"peak={peak}  ({timing_backend})")

    caveat = (
        "RAW wall-clock only. timing_backend='torch_reference' is fake-quantized "
        "(dequant + FP matmul), so its timing is NOT an INT8 speedup. A real "
        "speedup may only be claimed with a custom packed kernel vs FP16 at the "
        "SAME fp_ratio/shapes/layers/kernel path. No speedup ratio is computed "
        "for the reference backend."
    )
    common = {
        "model": args.model, "device": args.device, "dtype": dtype,
        "backend": args.backend, "batch_size": args.batch_size,
        "seq_len": args.seq_len, "iters": args.iters, "warmup": args.warmup,
        "cuda": str(args.device).startswith("cuda"), "caveat": caveat,
    }

    thr_rows = [{k: r[k] for k in
                 ("variant", "timing_backend", "avg_latency_ms", "prefill_s",
                  "tokens_per_s", "routed_layers", "fp_ratio", "grms_group_size")}
                for r in rows]
    mem_rows = [{"variant": r["variant"], "timing_backend": r["timing_backend"],
                 "peak_mb": r["peak_mb"]} for r in rows]

    thr_path = os.path.join(args.out_dir, "throughput_results.json")
    mem_path = os.path.join(args.out_dir, "memory_results.json")
    with open(thr_path, "w") as f:
        json.dump({**common, "rows": thr_rows}, f, indent=2)
    with open(mem_path, "w") as f:
        json.dump({**common, "rows": mem_rows}, f, indent=2)
    print(f"[bench] wrote {thr_path} and {mem_path}")
    if args.profile:
        print(f"[bench] wrote {prof_path}")
    print(f"[bench] CAVEAT: {caveat}")


if __name__ == "__main__":
    main()
