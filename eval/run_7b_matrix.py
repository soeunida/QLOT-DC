"""Run the StaticScale fp-ratio matrix across a 7B model zoo (public entrypoint).

For each locally-available model in the zoo, runs StaticScale equal-budget
selection (``configs/staticscale_7b_matrix.json`` by default, fp candidates
0.06/0.10/0.20) and writes per-model results under ``--out_root``. Models not in
the local cache are skipped (logged), never downloaded.

Example:
    python eval/run_7b_matrix.py --zoo configs/model_zoo_7b.json \\
        --config configs/staticscale_7b_matrix.json --device cuda:0 \\
        --seq_len 2048 --max_chunks 64 --out_root results/staticscale_7b_matrix
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.check_model_zoo_availability import is_available  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="StaticScale 7B matrix")
    ap.add_argument("--zoo", default="configs/model_zoo_7b.json")
    ap.add_argument("--config", default="configs/staticscale_7b_matrix.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--max_chunks", type=int, default=64)
    ap.add_argument("--out_root", default="results/staticscale_7b_matrix")
    ap.add_argument("--skip_unavailable", action="store_true", default=True)
    a = ap.parse_args()

    models = json.load(open(a.zoo)).get("models", [])
    os.makedirs(a.out_root, exist_ok=True)
    from eval import select_sadnd_cap as S
    done, skipped = [], []
    for m in models:
        ok, note = is_available(m)
        if not ok and a.skip_unavailable:
            print(f"[matrix] SKIP {m} (not local: {note})")
            skipped.append(m)
            continue
        out_dir = os.path.join(a.out_root, m.replace("/", "__"))
        print(f"[matrix] RUN {m} -> {out_dir}")
        sys.argv = ["run_7b_matrix", "--config", a.config, "--model", m,
                    "--device", a.device, "--seq_len", str(a.seq_len),
                    "--max_chunks", str(a.max_chunks), "--out_dir", out_dir]
        try:
            S.main()
            done.append(m)
        except Exception as e:  # report, do not fabricate
            print(f"[matrix] FAILED {m}: {type(e).__name__}: {e}")
    print(f"[matrix] done={done} skipped={skipped}")


if __name__ == "__main__":
    main()
