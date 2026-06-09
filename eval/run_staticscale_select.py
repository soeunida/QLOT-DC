"""StaticScale equal-budget policy selection (public entrypoint).

Runs the StaticScale static policy search and equal-budget accept-only selection.
This is the public, StaticScale-named front-end; it currently wraps the internal
selection driver (``eval/select_sadnd_cap.py``) — the CLI flags are identical.

Example:
    python eval/run_staticscale_select.py \\
        --config configs/staticscale_select.json \\
        --model Qwen/Qwen2.5-7B --device cuda:0 --seq_len 2048 --max_chunks 64 \\
        --out_dir results/staticscale_qwen25_7b
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.select_sadnd_cap import main  # noqa: E402

if __name__ == "__main__":
    main()
