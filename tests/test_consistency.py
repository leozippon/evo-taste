"""The policy-consistency check: robust to isolated near-tie tokens in short answers, strict on broad shifts."""

import random
from types import SimpleNamespace

import pytest
import torch

from evotaste import config
from evotaste.member import ConsistencyError, Lineage, Member, check_sample
from evotaste.trainer import Sequence

PROMPTS, K, LENGTH = 128, 4, 15  # terse answers: 15 tokens per rollout


class StubPolicy:
    """Returns the sampler's log-probabilities plus planted discrepancies, as the trainer would with them."""

    def __init__(self, sampler_lps, planted):
        self.sampler_lps, self.planted = sampler_lps, planted

    def logprobs(self, _weights, seqs):
        out = []
        for s in seqs:
            i = s.prompt[0]
            out.append(torch.tensor(self.sampler_lps[i], dtype=torch.float32) + self.planted[i])
        return out


def check(planted_fn, tmp_path):
    rng = random.Random(0)
    n = PROMPTS * K
    seqs = [Sequence([i], [rng.randrange(1000) for _ in range(LENGTH)], 0.0) for i in range(n)]
    sampler = [[-rng.random() for _ in range(LENGTH)] for _ in range(n)]
    noise = [torch.tensor([rng.gauss(0, 0.006) for _ in range(LENGTH)]) for _ in range(n)]
    planted = [noise[i] + planted_fn(i) for i in range(n)]
    cfg = config.parse({"name": "t", "model": "m", "seed": 0, "amplification": {"rollouts": K},
                        "seeding": {"method": "none"}, "strata": {"path": "strata.tsv", "sha256": "0" * 64},
                        "members": [{"id": "m0", "seed": 1}]}, set())
    run = SimpleNamespace(dir=tmp_path, cfg=cfg, policy=StubPolicy(sampler, planted))
    member = Member(run, Lineage("m0", "m0", 0, None, 1, status="active"))
    return member.consistency_check(seqs, sampler, K)


def one_token(i, nats=3.0):
    d = torch.zeros(LENGTH)
    if i == 0:  # the first rollout of the first prompt: always in the sample
        d[3] = nats
    return d


def test_sample_is_token_based():
    lengths = [LENGTH] * (PROMPTS * K)
    sample = check_sample(lengths, K, 3000, 8)
    assert len(sample) == 200 and sample[:3] == [0, 4, 8] and sample[128] == 1  # then second rollouts
    assert check_sample([600] * 16, 4, 3000, 8) == [0, 4, 8, 12, 1, 5, 9, 13]  # long answers: minimum rollouts
    assert len(check_sample([5] * 16, 4, 3000, 8)) == 16  # a small batch: all of it


def test_single_near_tie_token_in_short_answers_passes(tmp_path):
    stats = check(one_token, tmp_path)
    assert stats["tokens"] >= 3000 and "first_attempt" not in stats


def test_uniform_shift_fails(tmp_path):
    with pytest.raises(ConsistencyError):
        check(lambda i: torch.full((LENGTH,), 0.1), tmp_path)


def test_clustered_near_ties_are_resolved_on_the_whole_batch(tmp_path):
    stats = check(lambda i: one_token(0, 3.0) if i % K == 0 and i < 80 else torch.zeros(LENGTH), tmp_path)
    assert stats["first_attempt"]["seq_abs_per_token"] > 0.015  # 20 planted tokens fail the 3,000-token sample
    assert stats["rollouts"] == PROMPTS * K and stats["seq_abs_per_token"] <= 0.015
