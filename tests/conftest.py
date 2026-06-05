"""Shared pytest fixtures for Q-LOT-RMS tests.

All tests run on CPU with a tiny, randomly-initialized Llama model (no download,
no GPU).  A fake tokenizer is provided for the synthetic calibration fallback.
"""

import os
import sys

import pytest
import torch

# make the qlot_rms package importable (repo root = parent of this tests/ dir)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(scope="session")
def torch_seed():
    torch.manual_seed(0)
    return 0


@pytest.fixture
def tiny_model():
    """A tiny LlamaForCausalLM (CPU, random init).

    hidden_size=64 (LN2 channel count C=64), so with fp_ratio=0.06, K_F=3.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(cfg).eval()
    return model


class FakeTokenizer:
    """Minimal tokenizer for the synthetic calibration fallback."""

    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size

    def __call__(self, text, return_tensors=None):
        # not used for synthetic path; present for API compatibility
        class _Out:
            pass

        out = _Out()
        out.input_ids = torch.zeros(1, 8, dtype=torch.long)
        return out


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer(vocab_size=256)


@pytest.fixture
def small_config():
    from qlot_rms.config import QLotRmsConfig

    return QLotRmsConfig(
        enable_qlot_rms=True,
        method="sadnd_cap",
        qlot_scope="mlp_only",
        routing_score="output_aware_sadnd",
        fp_budget_mode="global",
        int_permutation_mode="packing_aware",
        fp_ratio=0.06,
        w8_group_size=16,          # small so the INT block forms several groups
        calibration_samples=8,
        calibration_seq_len=16,
        num_calib_subsets=3,
        subset_size=4,
        seed=0,
        backend="torch_reference",
        act_scale_max_tokens=512,
    )
