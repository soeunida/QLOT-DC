"""Calibration data: WikiText-2 chunks and deterministic random subsets.

Default calibration set (paper):
  * 128 WikiText-2 chunks
  * sequence length 512
  * 5 random subsets
  * 32 sequences per subset

``build_calibration_chunks`` tokenizes WikiText-2(-raw) train text into
fixed-length chunks.  If the dataset cannot be downloaded (offline), the caller
may pass ``allow_synthetic=True`` to fall back to deterministic synthetic token
sequences -- useful for tests and smoke runs (NOT for reporting perplexity).

``make_subsets`` draws ``num_subsets`` disjoint-where-possible random subsets of
``subset_size`` chunk indices using a fixed seed, for reproducible mean+std
aggregation in SADND.
"""

from __future__ import annotations

from typing import List, Optional

import torch


def build_calibration_chunks(
    tokenizer,
    seq_len: int = 512,
    num_chunks: int = 128,
    seed: int = 0,
    allow_synthetic: bool = False,
    hf_id: str = "wikitext",
    hf_name: str = "wikitext-2-raw-v1",
    split: str = "train",
    text_field: str = "text",
) -> torch.Tensor:
    """Return an int64 tensor ``[num_chunks, seq_len]`` of token ids.

    Tries to load WikiText-2 via ``datasets``.  On failure, if
    ``allow_synthetic`` is True, returns deterministic pseudo-random token ids
    (vocab-bounded) so downstream code can run without network access.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(hf_id, hf_name, split=split)
        text = "\n\n".join(t for t in ds[text_field] if t and t.strip())
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        total = num_chunks * seq_len
        if ids.numel() < total:
            raise RuntimeError(
                f"not enough calibration tokens: have {ids.numel()}, need {total}"
            )
        ids = ids[:total].reshape(num_chunks, seq_len).contiguous()
        return ids.to(torch.long)
    except Exception as e:  # noqa: BLE001
        if not allow_synthetic:
            raise RuntimeError(
                f"failed to load WikiText-2 ({e!r}); pass allow_synthetic=True "
                "for an offline (non-PPL) fallback."
            ) from e
        vocab = int(getattr(tokenizer, "vocab_size", 32000) or 32000)
        g = torch.Generator().manual_seed(seed)
        ids = torch.randint(0, max(2, vocab), (num_chunks, seq_len), generator=g)
        return ids.to(torch.long)


def make_subsets(
    num_chunks: int,
    num_subsets: int = 5,
    subset_size: int = 32,
    seed: int = 0,
) -> List[torch.Tensor]:
    """Deterministic list of ``num_subsets`` index tensors of ``subset_size``.

    Sampling is without replacement within a subset.  If ``num_chunks`` is too
    small to draw disjoint subsets, subsets may overlap (sampled independently),
    which is fine for mean+std aggregation.
    """
    g = torch.Generator().manual_seed(seed)
    subsets: List[torch.Tensor] = []
    for _ in range(num_subsets):
        if subset_size <= num_chunks:
            perm = torch.randperm(num_chunks, generator=g)
            subsets.append(perm[:subset_size].clone())
        else:
            subsets.append(
                torch.randint(0, num_chunks, (subset_size,), generator=g)
            )
    return subsets


def iter_batches(
    chunk_ids: torch.Tensor, indices: torch.Tensor, batch_size: int = 4
):
    """Yield mini-batches ``[b, seq_len]`` of the selected chunk indices."""
    sel = chunk_ids.index_select(0, indices)
    for i in range(0, sel.shape[0], batch_size):
        yield sel[i:i + batch_size]
