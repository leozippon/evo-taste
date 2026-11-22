import copy
import random

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from evotaste.config import Lora, Trainer
from evotaste.trainer import Policy, Sequence

VOCAB = 97


def tiny_base(dtype=torch.float32):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=VOCAB, hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=12, max_position_embeddings=512,
                      tie_word_embeddings=True, attn_implementation="sdpa")
    return Qwen3ForCausalLM(cfg).to(dtype)


@pytest.fixture(scope="session")
def base_model():
    return tiny_base()


def make_policy(base_model, **trainer):
    return Policy("tiny", Lora(rank=4, alpha=8), Trainer(**{"grad_checkpointing": False, **trainer}), device="cpu",
                  base_model=copy.deepcopy(base_model))


def random_groups(n_prompts=3, k=3, seed=0):
    """n_prompts prompts with k completions each, of varied lengths and distinct tokens."""
    rng = random.Random(seed)
    seqs = []
    for _ in range(n_prompts):
        prompt = [rng.randrange(VOCAB) for _ in range(rng.randint(5, 12))]
        for _ in range(k):
            seqs.append(Sequence(prompt, [rng.randrange(VOCAB) for _ in range(rng.randint(1, 9))], rng.uniform(-1, 1)))
    return seqs
