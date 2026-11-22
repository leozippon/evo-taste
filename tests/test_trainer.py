"""Update direction, group weights and plain versus shared-prefix equivalence on a tiny random Qwen3 model."""

import torch

from evotaste.trainer import Sequence, group_weights
from tests.conftest import make_policy, random_groups


def test_group_weights():
    assert group_weights([1, 1, 1, 1]) == [0.0] * 4
    assert group_weights([0, 0, 0, 0]) == [0.0] * 4
    w = group_weights([1, 0, 0, 0])
    assert w[0] > 0 and all(x < 0 for x in w[1:])
    assert abs(sum(w)) < 1e-9 and abs(w[0] - 3**0.5) < 1e-5  # (1 - 0.25) / sqrt(0.1875)


def _seq_logprobs(policy, member, seqs):
    return [float(lp.sum()) for lp in policy.logprobs(member, seqs)]


def test_group_update_direction(base_model):
    policy = make_policy(base_model)
    member = policy.new_member(seed=1)
    prompt = [3, 14, 15, 92, 65, 35]
    completions = [[10, 11, 12, 13], [20, 21, 22], [30, 31, 32, 33, 34], [40, 41]]
    rewards = [1.0, 0.0, 1.0, 0.0]
    seqs = [Sequence(prompt, c, w) for c, w in zip(completions, group_weights(rewards))]
    before = _seq_logprobs(policy, member, seqs)
    stats = policy.update(member, seqs, lr=1e-3)
    after = _seq_logprobs(policy, member, seqs)
    assert stats["sequences"] == 4 and stats["tokens"] == 14
    for r, b, a in zip(rewards, before, after):
        assert (a > b) if r == 1.0 else (a < b), (rewards, before, after)


def test_positive_update_lowers_nll(base_model):
    policy = make_policy(base_model)
    member = policy.new_member(seed=2)
    seqs = [Sequence(s.prompt, s.completion, 1.0) for s in random_groups(seed=3)]
    before = sum(_seq_logprobs(policy, member, seqs))
    policy.update(member, seqs, lr=1e-3)
    assert sum(_seq_logprobs(policy, member, seqs)) > before


def test_zero_weights_change_nothing(base_model):
    policy = make_policy(base_model)
    member = policy.new_member(seed=4)
    snapshot = {k: v.clone() for k, v in member.lora.items()}
    seqs = [Sequence(s.prompt, s.completion, 0.0) for s in random_groups(seed=5)]
    assert policy.update(member, seqs, lr=1e-2)["sequences"] == 0
    assert all(torch.equal(snapshot[k], member.lora[k]) for k in snapshot) and member.optimizer is None


def _loss_and_grads(policy, member, seqs):
    policy._activate(member)
    for p in policy.params.values():
        p.grad = None
    loss, _ = policy._accumulate(seqs)
    return loss, {n: p.grad.clone() for n, p in policy.params.items()}


def test_shared_prefix_equals_plain(base_model):
    """Packing a prompt's completions behind one copy of the prompt gives the plain path's loss and gradients."""
    plain = make_policy(base_model, shared_prefix=False, token_budget=40)  # small budget: several micro-batches
    shared = make_policy(base_model, shared_prefix=True, token_budget=40)
    member = plain.new_member(seed=6, b_std=0.1)  # non-zero B: the LoRA path contributes to the gradients
    seqs = random_groups(n_prompts=4, k=3, seed=7)
    lp_plain, lp_shared = plain.logprobs(member, seqs), shared.logprobs(member, seqs)
    for a, b in zip(lp_plain, lp_shared):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    loss_p, grads_p = _loss_and_grads(plain, member, seqs)
    loss_s, grads_s = _loss_and_grads(shared, member, seqs)
    assert abs(loss_p - loss_s) < 1e-5
    for name in grads_p:
        torch.testing.assert_close(grads_p[name], grads_s[name], rtol=1e-4, atol=1e-6)
    assert max(g.abs().max() for g in grads_p.values()) > 1e-4  # the comparison is not between zeros


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
