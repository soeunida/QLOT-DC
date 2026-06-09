"""Check local availability of the StaticScale 7B model zoo (no downloads).

Reads a model-zoo JSON (default ``configs/model_zoo_7b.json``) and reports which
models are present in the local Hugging Face cache (``local_files_only=True``).
Nothing is downloaded.

Example:
    python eval/check_model_zoo_availability.py --zoo configs/model_zoo_7b.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def is_available(model_id):
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model_id, local_files_only=True)
        return True, "cached"
    except Exception as e:  # not in local cache / not accessible offline
        return False, type(e).__name__


def main():
    ap = argparse.ArgumentParser(description="StaticScale model-zoo availability")
    ap.add_argument("--zoo", default="configs/model_zoo_7b.json")
    a = ap.parse_args()
    zoo = json.load(open(a.zoo))
    models = zoo.get("models", [])
    print(f"# StaticScale model zoo: {a.zoo} ({len(models)} models)")
    avail = 0
    for m in models:
        ok, note = is_available(m)
        avail += int(ok)
        print(f"  [{'AVAILABLE' if ok else 'missing  '}] {m}  ({note})")
    print(f"# {avail}/{len(models)} available locally (no downloads attempted)")


if __name__ == "__main__":
    main()
