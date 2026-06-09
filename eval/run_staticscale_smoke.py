"""StaticScale quick smoke run (public entrypoint).

Runs the full StaticScale selection with a reduced calibration on a small model,
to confirm the pipeline end-to-end. Wraps the internal selection driver.

Example:
    python eval/run_staticscale_smoke.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    ap = argparse.ArgumentParser(description="StaticScale smoke run")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--config", default="configs/staticscale_tinyllama_smoke.json")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--max_chunks", type=int, default=32)
    ap.add_argument("--out_dir", default="results/staticscale_smoke")
    ap.add_argument("--allow_synthetic", action="store_true")
    ap.add_argument("--dry_run", action="store_true",
                    help="validate config + imports only; no model load / GPU work")
    a = ap.parse_args()

    if a.dry_run:
        from staticscale import StaticScaleConfig
        cfg = StaticScaleConfig.load_json(a.config).validate()
        print(f"[smoke:dry_run] config OK: {a.config}")
        print(f"[smoke:dry_run] method components: routing={cfg.routing_score} "
              f"budget={cfg.fp_budget_mode} refine={cfg.use_fp_mask_refinement} "
              f"clip_gain={cfg.use_groupwise_clip_gain_tuning} perm={cfg.int_permutation_mode}")
        print(f"[smoke:dry_run] would run StaticScale selection on model={a.model} "
              f"(seq_len={a.seq_len}, max_chunks={a.max_chunks}). No GPU work performed.")
        return

    from eval import select_sadnd_cap as S
    argv = ["run_staticscale_smoke", "--config", a.config, "--model", a.model,
            "--device", a.device, "--seq_len", str(a.seq_len),
            "--max_chunks", str(a.max_chunks), "--out_dir", a.out_dir,
            "--calib_samples", "48", "--calib_subsets", "3", "--subset_size", "24"]
    if a.allow_synthetic:
        argv.append("--allow_synthetic")
    sys.argv = argv
    S.main()


if __name__ == "__main__":
    main()
