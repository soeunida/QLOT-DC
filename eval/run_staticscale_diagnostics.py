"""StaticScale diagnostics runner.

Computes the five diagnostic CSVs (budget saturation, clip dominance, mask overlap,
hard layers, proxy mismatch) over a set of FP ratios, to explain *why* the full
pipeline adds little over CAP+ + clip and whether tighter budgets change that.

Example:
    python eval/run_staticscale_diagnostics.py \\
      --model Qwen/Qwen2.5-7B --device cuda:1 \\
      --fp_ratios 0.03,0.06,0.10,0.20 --seq_len 2048 --max_chunks 64 \\
      --calib_chunks 128 --sel_chunks 32 \\
      --out_root results/staticscale_diagnostics_qwen25_7b

Local models only; no download. No speedup is claimed.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from staticscale import diagnostics as D  # noqa: E402
from eval.eval_perplexity import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="optional diagnostics config json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--fp_ratios", default="0.03,0.06,0.10,0.20")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--max_chunks", type=int, default=64)
    ap.add_argument("--calib_chunks", type=int, default=128)
    ap.add_argument("--calib_seq_len", type=int, default=512)
    ap.add_argument("--sel_chunks", type=int, default=32)
    ap.add_argument("--n_layers", type=int, default=12, help="layers sampled per fp")
    ap.add_argument("--calib_batch_size", type=int, default=4)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--no_download", action="store_true", default=True)
    args = ap.parse_args()

    cfg_dict = {}
    if args.config and os.path.exists(args.config):
        cfg_dict = json.load(open(args.config))
    # CLI overrides / defaults
    cfg_dict.setdefault("model", args.model)
    cfg_dict["calib_chunks"] = args.calib_chunks
    cfg_dict["calib_seq_len"] = args.calib_seq_len
    cfg_dict["sel_chunks"] = args.sel_chunks
    cfg_dict.setdefault("gt_max_tokens", 256)
    if args.no_download or cfg_dict.get("no_download"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    fp_ratios = [float(x) for x in args.fp_ratios.split(",") if x.strip()]
    model_id = cfg_dict.get("model", args.model)
    print(f"[load] {model_id} on {args.device} (offline)")
    model, tok = load_model(model_id, args.device)

    paths = D.run_diagnostics(model, tok, cfg_dict, fp_ratios, args.device,
                              args.out_root, batch_size=args.calib_batch_size,
                              n_layers=args.n_layers)
    print("wrote:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    with open(paths["summary"]) as f:
        print("\n" + f.read())


if __name__ == "__main__":
    main()
