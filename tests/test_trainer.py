"""Returns, advantages, the group update with self-imitation, row packing, base log-probabilities and the reverse-KL
distillation update, on a tiny random Qwen3 model."""

import copy
import random

import pytest
import torch

from evotaste import trainer
from evotaste.trainer import Sequence, advantages, build_rows, length_returns
from tests.conftest import make_policy, random_groups


def test_length_returns():
    r = length_returns([1.0, 1.0, 1.0, 0.0, 1.0], [100, 2048, 3072, 4096, 5000], cap=4096, free=2048, slope=0.25)
    assert r == [1.0, 1.0, 0.875, -0.25, 0.75]  # free up to 2048, linear to 0.25 at the cap, clipped beyond


def test_equal_returns_contribute_nothing():
    rng = random.Random(0)
    for _ in range(500):  # values whose float mean differs from each of them
        v = rng.choice([0.7, 0.1, 1.0 - 0.25 * rng.random(), -0.25 * rng.random()])
        for rule in ("centered", "standardized"):
            assert advantages([v] * 4, rule) == [0.0] * 4


def test_length_difference_alone_gives_small_proportional_weights():
    returns = length_returns([1.0] * 4, [1000, 2048, 2560, 3072], cap=4096, free=2048, slope=0.25)  # 1, 1, .9375, .875
    w = advantages(returns, "centered")
    mean = sum(returns) / 4
    assert w == pytest.approx([(r - mean) / 0.5 for r in returns])
    assert max(abs(x) for x in w) < 0.2 and abs(sum(w)) < 1e-12
    assert w[0] - w[2] == pytest.approx((w[0] - w[3]) / 2)  # proportional to the length deduction
    assert max(abs(x) for x in advantages(returns, "standardized")) > 1.0  # what standardising would make of it
    w = advantages([1.0, 0.0, 0.0, 0.0], "centered")
    assert w == [1.5, -0.5, -0.5, -0.5]  # a full success keeps a full-size weight


def test_standardized_is_round_1_rule():
    w = advantages([1, 0, 0, 0], "standardized")
    assert abs(sum(w)) < 1e-9 and abs(w[0] - 3**0.5) < 1e-5  # (1 - 0.25) / sqrt(0.1875)
    with pytest.raises(ValueError):
        advantages([1, 0], "normalized")


def _seq_logprobs(policy, member, seqs):
    return [float(lp.sum()) for lp in policy.logprobs(member, seqs)]


def test_group_update_direction(base_model):
    policy = make_policy(base_model)
    member = policy.new_member(seed=1)
    prompt = [3, 14, 15, 92, 65, 35]
    completions = [[10, 11, 12, 13], [20, 21, 22], [30, 31, 32, 33, 34], [40, 41]]
    rewards = [1.0, 0.0, 1.0, 0.0]
    seqs = [Sequence(prompt, c, w) for c, w in zip(completions, advantages(rewards, "centered"))]
    before = _seq_logprobs(policy, member, seqs)
    stats = policy.update(member, seqs, lr=1e-3)
    after = _seq_logprobs(policy, member, seqs)
    assert stats["sequences"] == 4 and stats["tokens"] == 14
    for r, b, a in zip(rewards, before, after):
        assert (a > b) if r == 1.0 else (a < b), (rewards, before, after)


def test_zero_weights_change_nothing(base_model):
    policy = make_policy(base_model)
    member = policy.new_member(seed=4)
    snapshot = {k: v.clone() for k, v in member.lora.items()}
    seqs = [Sequence(s.prompt, s.completion, 0.0, imitate=True) for s in random_groups(seed=5)]
    assert policy.update(member, seqs, lr=1e-2)["sequences"] == 0  # imitation flags without beta do nothing
    assert all(torch.equal(snapshot[k], member.lora[k]) for k in snapshot) and member.optimizer is None


def _flagged(seed):
    seqs = random_groups(n_prompts=4, k=3, seed=seed)
    for i, s in enumerate(seqs):
        s.imitate = i % 3 == 0
        if i % 4 == 0:
            s.weight = 0.0
    return seqs


def test_self_imitation_zero_is_the_advantage_only_update(base_model):
    policy = make_policy(base_model, token_budget=40)
    start = policy.new_member(seed=6, b_std=0.05)
    policy.update(start, random_groups(seed=1), lr=1e-3)  # with optimizer state
    flagged = _flagged(seed=2)
    plain = [Sequence(s.prompt, s.completion, s.weight) for s in flagged]
    a, b = copy.deepcopy(start), copy.deepcopy(start)
    stats_a = policy.update(a, flagged, lr=1e-3, self_imitation=0.0)
    stats_b = policy.update(b, plain, lr=1e-3)
    assert stats_a["loss"] == stats_b["loss"] and stats_a["si_sequences"] == 0
    for name in a.lora:
        assert torch.equal(a.lora[name], b.lora[name])
    for (_, x), (_, y) in zip(sorted(a.optimizer["state"].items()), sorted(b.optimizer["state"].items())):
        assert torch.equal(x["exp_avg"], y["exp_avg"]) and torch.equal(x["exp_avg_sq"], y["exp_avg_sq"])


def _reference_grads(policy, member, loss_fn):
    """Gradients of loss_fn(per-sequence summed log-probabilities from an ordinary full forward)."""
    policy._activate(member)
    for p in policy.params.values():
        p.grad = None
    loss = loss_fn(lambda s: policy.model(input_ids=torch.tensor([s.prompt + s.completion])).logits[
        0, len(s.prompt) - 1:-1].float().log_softmax(-1).gather(1, torch.tensor(s.completion)[:, None]).sum())
    loss.backward()
    return loss.item(), {n: p.grad.clone() for n, p in policy.params.items()}


def _engine_grads(policy, member, monkeypatch, run):
    captured = {}

    def capture(_member, _opt):
        captured.update({n: p.grad.clone() for n, p in policy.params.items()})
        return {"grad_norm": 0.0}
    monkeypatch.setattr(policy, "_step", capture)
    stats = run()
    return stats, captured


@pytest.mark.parametrize("shared_prefix", [True, False])
def test_self_imitation_loss_is_the_sum_of_both_normalised_terms(base_model, monkeypatch, shared_prefix):
    policy = make_policy(base_model, shared_prefix=shared_prefix, token_budget=40)
    member = policy.new_member(seed=7, b_std=0.1)
    seqs, beta = _flagged(seed=3), 0.5
    adv = [s for s in seqs if s.weight != 0.0]
    imitated = [s for s in seqs if s.imitate]

    def loss_fn(lp):
        a = -sum(s.weight * lp(s) for s in adv) / sum(len(s.completion) for s in adv)
        return a - beta * sum(lp(s) for s in imitated) / sum(len(s.completion) for s in imitated)
    ref_loss, ref = _reference_grads(policy, member, loss_fn)
    stats, ours = _engine_grads(policy, member, monkeypatch, lambda: policy.update(member, seqs, 1e-3, beta))
    assert stats["loss"] == pytest.approx(ref_loss, abs=1e-5)
    assert stats["si_sequences"] == len(imitated) and stats["sequences"] == len({id(s) for s in adv + imitated})
    for name in ref:
        torch.testing.assert_close(ours[name], ref[name], rtol=1e-4, atol=1e-6)
    assert max(g.abs().max() for g in ref.values()) > 1e-4


def test_self_imitation_raises_a_success_that_has_no_advantage(base_model):
    policy = make_policy(base_model)
    member = policy.new_member(seed=8, b_std=0.05)
    prompt = [5, 6, 7, 8]
    seqs = [Sequence(prompt, [11, 12, 13], 0.0, imitate=True), Sequence(prompt, [21, 22], 0.0, imitate=True)]
    before = _seq_logprobs(policy, member, seqs)
    stats = policy.update(member, seqs, lr=1e-3, self_imitation=0.5)  # an all-success group: advantages are zero
    assert stats["sequences"] == 2 and stats["si_tokens"] == 5 and stats["adv_loss"] == 0.0
    assert all(a > b for a, b in zip(_seq_logprobs(policy, member, seqs), before))


def test_rows_respect_the_token_budget():
    prompt = list(range(10))
    seqs = [Sequence(prompt, [1] * n) for n in (30, 30, 5, 5, 60)]
    rows = build_rows(seqs, shared_prefix=True, token_budget=50)
    assert [[i for i, _, _ in r.spans] for r in rows] == [[0], [1, 2, 3], [4]]
    assert all(len(r.ids) <= 50 for r in rows[:2]) and len(rows[2].ids) == 70  # one long completion stands alone
    assert [r.seg[-1] for r in rows] == [1, 3, 1]


def test_shared_prefix_equals_plain(base_model):
    """Packing a prompt's completions behind one copy of the prompt gives the plain path's log-probabilities and
    gradients, also when the token budget splits a prompt's completions over several rows."""
    plain = make_policy(base_model, shared_prefix=False, token_budget=20)
    shared = make_policy(base_model, shared_prefix=True, token_budget=20)
    member = plain.new_member(seed=6, b_std=0.1)  # non-zero B: the LoRA path contributes to the gradients
    seqs = random_groups(n_prompts=4, k=3, seed=7)
    assert len(build_rows(seqs, True, 20)) > 4  # some prompt's completions are split
    for a, b in zip(plain.logprobs(member, seqs), shared.logprobs(member, seqs)):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    grads = []
    for policy in (plain, shared):
        captured = {}
        policy._step = lambda _m, _o, policy=policy, captured=captured: captured.update(
            {n: p.grad.clone() for n, p in policy.params.items()}) or {"grad_norm": 0.0}
        grads.append((policy.update(member, seqs, 1e-3)["loss"], captured))
    assert grads[0][0] == pytest.approx(grads[1][0], abs=1e-5)
    for name in grads[0][1]:
        torch.testing.assert_close(grads[0][1][name], grads[1][1][name], rtol=1e-4, atol=1e-6)


def test_logprobs_match_full_forward(base_model):
    """The completion-only LM head returns exactly the log-softmax of the model's ordinary full forward."""
    policy = make_policy(base_model, shared_prefix=True)
    member = policy.new_member(seed=8, b_std=0.1)
    seqs = random_groups(n_prompts=2, k=2, seed=9)
    ours = policy.logprobs(member, seqs)
    with torch.no_grad():
        for s, lp in zip(seqs, ours):
            ids = torch.tensor([s.prompt + s.completion])
            logits = policy.model(input_ids=ids).logits[0, len(s.prompt) - 1:-1].float()
            ref = logits.log_softmax(-1).gather(1, torch.tensor(s.completion)[:, None])[:, 0]
            torch.testing.assert_close(lp, ref, rtol=1e-5, atol=1e-5)


def test_base_logprobs_are_the_adapter_free_model(base_model):
    policy = make_policy(base_model)
    seqs = random_groups(seed=10)
    zero_b = policy.new_member(seed=1)  # B = 0: the member is the base model
    perturbed = policy.new_member(seed=1, b_std=0.1)
    policy._activate(perturbed)  # whatever is active, the base pass ignores it
    for a, b, c in zip(policy.base_logprobs(seqs), policy.logprobs(zero_b, seqs), policy.logprobs(perturbed, seqs)):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
        assert not torch.allclose(a, c, atol=1e-3)
    assert all(p.requires_grad for p in policy.params.values())  # the adapter is trainable again afterwards


# -- distillation -------------------------------------------------------------------------------------------------

def _distill_batch(seed=11, n=5):
    rng = random.Random(seed)
    seqs, teacher = [], []
    for _ in range(n):
        q = [rng.randrange(90) for _ in range(rng.randint(4, 9))]
        seqs.append(Sequence(q, [rng.randrange(90) for _ in range(rng.randint(1, 12))]))
        teacher.append([rng.randrange(90) for _ in range(rng.randint(2, 6))] + q)  # a context before the question
    return seqs, teacher


def _reference_kl(policy, member, seqs, teacher, base_model):
    """sum over completion positions of KL(member(.|prompt) || base(.|teacher prompt)) / tokens, by full forwards;
    the teacher is the separate, never wrapped base model."""
    policy._activate(member)
    for p in policy.params.values():
        p.grad = None
    total, tokens = 0.0, sum(len(s.completion) for s in seqs)
    for s, t in zip(seqs, teacher):
        with torch.no_grad():
            q = base_model(input_ids=torch.tensor([t + s.completion])).logits[0, len(t) - 1:-1].float()
            q = q.log_softmax(-1)
        p = policy.model(input_ids=torch.tensor([s.prompt + s.completion])).logits[0, len(s.prompt) - 1:-1]
        p = p.float().log_softmax(-1)
        total = total + (p.exp() * (p - q)).sum()
    loss = total / tokens
    loss.backward()
    return loss.item(), {n: p.grad.clone() for n, p in policy.params.items()}


@pytest.mark.parametrize("shared_prefix", [True, False])
def test_distill_is_the_full_vocabulary_reverse_kl(base_model, monkeypatch, shared_prefix):
    policy = make_policy(base_model, shared_prefix=shared_prefix, token_budget=24)
    member = policy.new_member(seed=12, b_std=0.1)
    seqs, teacher = _distill_batch()
    seqs.append(Sequence(seqs[0].prompt, [3, 4, 5]))  # a second completion of one prompt: packed with the first
    teacher.append(teacher[0])
    ref_loss, ref = _reference_kl(policy, member, seqs, teacher, base_model)
    chunks = []
    real = trainer._chunk_kl
    monkeypatch.setattr(trainer, "_chunk_kl", lambda s, t, w: chunks.append(len(s)) or real(s, t, w))
    monkeypatch.setattr(trainer, "KL_CHUNK", 4)
    stats, ours = _engine_grads(policy, member, monkeypatch, lambda: policy.distill(member, seqs, teacher, 1e-3))
    assert stats["kl"] == pytest.approx(ref_loss, rel=1e-5) and ref_loss > 0
    assert stats["tokens"] == sum(len(s.completion) for s in seqs) and max(chunks) <= 4
    for name in ref:
        torch.testing.assert_close(ours[name], ref[name], rtol=1e-4, atol=1e-7)
    assert max(g.abs().max() for g in ref.values()) > 1e-5


def test_distill_teacher_pass_is_adapter_free_and_without_gradient(base_model, monkeypatch):
    """Per micro-batch the teacher pass runs first, with every LoRA layer disabled and grad off; then the member's
    pass with the adapter and grad on. A member that is the base model, given the teacher's own prompt, has KL 0."""
    policy = make_policy(base_model, token_budget=30)
    lora_layer = policy.model.base_model.model.model.layers[0].self_attn.q_proj
    calls = []
    real = policy._hidden
    monkeypatch.setattr(policy, "_hidden", lambda rows: calls.append(
        (torch.is_grad_enabled(), lora_layer.disable_adapters)) or real(rows))
    seqs, teacher = _distill_batch(seed=13)
    stats = policy.distill(policy.new_member(seed=1, b_std=0.1), seqs, teacher, 1e-3)
    assert stats["micro_batches"] > 1 and calls == [(False, True), (True, False)] * stats["micro_batches"]
    same = policy.distill(policy.new_member(seed=1), seqs, [s.prompt for s in seqs], 1e-3)  # B = 0
    assert abs(same["kl"]) < 1e-6 and same["grad_norm"] < 1e-6


def test_distill_lowers_the_kl(base_model):
    policy = make_policy(base_model)
    member = policy.new_member(seed=14)
    seqs, teacher = _distill_batch(seed=15)
    kls = [policy.distill(member, seqs, teacher, 3e-3)["kl"] for _ in range(6)]
    assert kls[-1] < kls[0] * 0.9 and member.optimizer is not None
