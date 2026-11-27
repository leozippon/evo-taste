"""Returns, advantages, the group update with self-imitation and exaggeration, row packing, base log-probabilities,
the reverse-KL distillation update and scaled adapters, on a tiny random Qwen3 model."""

import copy
import random

import pytest
import torch

from evotaste import config, trainer
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


def _stored(seqs):
    """Two stored answers for replay imitation, one behind a prompt of the batch and one behind another prompt."""
    return [Sequence(seqs[0].prompt, [3, 4, 5, 6], 0.0, replay=True),
            Sequence([9, 8, 7, 6, 5], [1, 2], 0.0, replay=True)]


@pytest.mark.parametrize("shared_prefix", [True, False])
def test_replay_is_its_own_normalised_term(base_model, monkeypatch, shared_prefix):
    policy = make_policy(base_model, shared_prefix=shared_prefix, token_budget=40)
    member = policy.new_member(seed=7, b_std=0.1)
    seqs, weight = _flagged(seed=3), 0.7
    stored, adv = _stored(seqs), [s for s in seqs if s.weight != 0.0]

    def loss_fn(lp):
        a = -sum(s.weight * lp(s) for s in adv) / sum(len(s.completion) for s in adv)
        return a - weight * sum(lp(s) for s in stored) / sum(len(s.completion) for s in stored)
    ref_loss, ref = _reference_grads(policy, member, loss_fn)
    stats, ours = _engine_grads(policy, member, monkeypatch,
                                lambda: policy.update(member, seqs + stored, 1e-3, replay=weight))
    assert stats["loss"] == pytest.approx(ref_loss, abs=1e-5) and stats["rp_loss"] > 0
    assert (stats["rp_sequences"], stats["rp_tokens"], stats["si_sequences"]) == (2, 6, 0)
    assert stats["sequences"] == len(adv) + 2
    for name in ref:
        torch.testing.assert_close(ours[name], ref[name], rtol=1e-4, atol=1e-6)


def test_replay_zero_is_the_update_without_stored_answers(base_model):
    policy = make_policy(base_model, token_budget=40)
    start = policy.new_member(seed=6, b_std=0.05)
    policy.update(start, random_groups(seed=1), lr=1e-3)  # with optimizer state
    seqs = _flagged(seed=2)
    a, b = copy.deepcopy(start), copy.deepcopy(start)
    stats_a = policy.update(a, seqs + _stored(seqs), lr=1e-3, replay=0.0)
    stats_b = policy.update(b, seqs, lr=1e-3)
    assert {k: v for k, v in stats_a.items() if k != "update_s"} == {k: v for k, v in stats_b.items()
                                                                     if k != "update_s"}
    assert "rp_loss" not in stats_a and all(torch.equal(a.lora[n], b.lora[n]) for n in a.lora)
    for (_, x), (_, y) in zip(sorted(a.optimizer["state"].items()), sorted(b.optimizer["state"].items())):
        assert torch.equal(x["exp_avg"], y["exp_avg"]) and torch.equal(x["exp_avg_sq"], y["exp_avg_sq"])
    with pytest.raises(ValueError, match="replay term only"):  # a stored answer is never a weighted rollout
        policy.update(a, [Sequence([1, 2], [3], 0.5, replay=True)], lr=1e-3, replay=0.5)


def test_self_imitation_raises_a_success_that_has_no_advantage(base_model):
    policy = make_policy(base_model)
    member = policy.new_member(seed=8, b_std=0.05)
    prompt = [5, 6, 7, 8]
    seqs = [Sequence(prompt, [11, 12, 13], 0.0, imitate=True), Sequence(prompt, [21, 22], 0.0, imitate=True)]
    before = _seq_logprobs(policy, member, seqs)
    stats = policy.update(member, seqs, lr=1e-3, self_imitation=0.5)  # an all-success group: advantages are zero
    assert stats["sequences"] == 2 and stats["si_tokens"] == 5 and stats["adv_loss"] == 0.0
    assert all(a > b for a, b in zip(_seq_logprobs(policy, member, seqs), before))


# -- exaggerated self-imitation -----------------------------------------------------------------------------------

EX = config.Exaggeration(gamma=1.5, weight=0.7)


def _full_logits(model, s):
    return model(input_ids=torch.tensor([s.prompt + s.completion])).logits[0, len(s.prompt) - 1:-1].float()


def _reference_exaggeration(policy, member, seqs, base_model, gamma, reverse=False):
    """sum over eligible completion positions of KL(q || member) / their tokens, q = softmax(l_m + gamma (l_m - l_b))
    with l_m, l_b detached, by ordinary full forwards; the base model is the separate, never wrapped one. With
    `reverse`, the other direction KL(member || q), for contrast. The member must be active (_reference_grads)."""
    total, tokens = 0.0, sum(len(s.completion) for s in seqs if s.imitate)
    for s in [s for s in seqs if s.imitate]:
        with torch.no_grad():
            lb = _full_logits(base_model, s)
        lm = _full_logits(policy.model, s)
        q = (lm.detach() + gamma * (lm.detach() - lb)).log_softmax(-1)
        p = lm.log_softmax(-1)
        total = total + ((p.exp() * (p - q)) if reverse else (q.exp() * (q - p))).sum()
    return total / tokens


def test_exaggeration_is_the_kl_from_the_fixed_target_to_the_member_on_eligible_rollouts(base_model, monkeypatch):
    policy = make_policy(base_model, token_budget=40)
    member = policy.new_member(seed=21, b_std=0.1)
    seqs, beta = _flagged(seed=4), 0.5  # eligible and not, with and without advantage, several micro-batches
    adv = [s for s in seqs if s.weight != 0.0]
    eligible = [s for s in seqs if s.imitate]

    def loss_fn(lp):
        a = -sum(s.weight * lp(s) for s in adv) / sum(len(s.completion) for s in adv)
        si = -sum(lp(s) for s in eligible) / sum(len(s.completion) for s in eligible)
        return a + beta * si + EX.weight * _reference_exaggeration(policy, member, seqs, base_model, EX.gamma)
    ref_loss, ref = _reference_grads(policy, member, loss_fn)
    chunks = []
    real = trainer._chunk_exaggeration
    monkeypatch.setattr(trainer, "_chunk_exaggeration", lambda m, b, w, g: chunks.append(len(m)) or real(m, b, w, g))
    monkeypatch.setattr(trainer, "KL_CHUNK", 4)
    stats, ours = _engine_grads(policy, member, monkeypatch, lambda: policy.update(member, seqs, 1e-3, beta, EX))
    assert stats["loss"] == pytest.approx(ref_loss, rel=1e-5) and stats["ex_loss"] > 0 and max(chunks) <= 4
    assert (stats["adv_sequences"], stats["ex_sequences"], stats["si_sequences"]) == (len(adv), len(eligible),
                                                                                      len(eligible))
    assert stats["ex_tokens"] == sum(len(s.completion) for s in eligible) and stats["micro_batches"] > 1
    for name in ref:
        torch.testing.assert_close(ours[name], ref[name], rtol=1e-4, atol=1e-7)
    _, reverse = _reference_grads(policy, member, lambda lp: _reference_exaggeration(
        policy, member, seqs, base_model, EX.gamma, reverse=True))
    _, forward = _reference_grads(policy, member, lambda lp: _reference_exaggeration(
        policy, member, seqs, base_model, EX.gamma))
    assert max(float((reverse[n] - forward[n]).abs().max()) for n in ref) > 1e-3 * max(
        float(forward[n].abs().max()) for n in ref)  # the direction matters, and the engine's is KL(q || member)


def test_exaggeration_base_pass_is_adapter_free_on_the_same_rows(base_model, monkeypatch):
    """Per micro-batch with an eligible rollout, the base pass runs first, without gradient and with every LoRA layer
    disabled, on exactly the rows (prompt and completion ids, positions) of the member's pass; a micro-batch without
    one has no base pass."""
    policy = make_policy(base_model, token_budget=30)
    lora_layer = policy.model.base_model.model.model.layers[0].self_attn.q_proj
    calls = []
    real = policy._hidden
    monkeypatch.setattr(policy, "_hidden", lambda rows: calls.append(
        (torch.is_grad_enabled(), lora_layer.disable_adapters, [(r.ids, r.pos, r.src) for r in rows])) or real(rows))
    seqs = random_groups(n_prompts=4, k=3, seed=5)
    for i, s in enumerate(seqs):
        s.imitate = i < 3  # the first prompt's rollouts only
    stats = policy.update(policy.new_member(seed=1, b_std=0.1), seqs, 1e-3, 0.0, EX)
    with_base = [k for k in range(len(calls) - 1) if not calls[k][0]]
    assert stats["micro_batches"] == len(calls) - len(with_base) > len(with_base) >= 1
    for k in with_base:
        assert calls[k][:2] == (False, True) and calls[k + 1][:2] == (True, False) and calls[k][2] == calls[k + 1][2]
    assert all(c[:2] == (True, False) for k, c in enumerate(calls) if k not in with_base)


def _assert_same_update(a, b):
    for name in a.lora:
        assert torch.equal(a.lora[name], b.lora[name]), name
    for (_, x), (_, y) in zip(sorted(a.optimizer["state"].items()), sorted(b.optimizer["state"].items())):
        assert torch.equal(x["exp_avg"], y["exp_avg"]) and torch.equal(x["exp_avg_sq"], y["exp_avg_sq"])


def test_exaggeration_vanishes_at_gamma_zero_and_for_a_member_equal_to_the_base(base_model):
    policy = make_policy(base_model, token_budget=40)
    start = policy.new_member(seed=22, b_std=0.05)
    policy.update(start, random_groups(seed=1), lr=1e-3)  # with optimizer state
    flagged = _flagged(seed=6)
    without = copy.deepcopy(start)
    plain = policy.update(without, flagged, 1e-3, 0.5)
    for ex in (config.Exaggeration(0.0, 0.7), config.Exaggeration(1.0, 0.0)):  # bit-identical to no term at all
        other = copy.deepcopy(start)
        stats = policy.update(other, flagged, 1e-3, 0.5, ex)
        assert stats["loss"] == plain["loss"] and stats["ex_sequences"] == 0
        _assert_same_update(other, without)
    # the function itself: at gamma 0 the target is the member, the KL is 0 and so is its gradient
    hidden = torch.randn(10, 48, requires_grad=True)
    kl = trainer.exaggeration_kl(hidden, torch.randn(10, 48), policy.head_weight.detach(), 0.0)
    kl.sum().backward()
    assert torch.equal(kl, torch.zeros(10)) and float(hidden.grad.abs().max()) < 1e-6
    # a member that is the base model (B = 0) feels no push, whatever gamma
    seqs = [Sequence(s.prompt, s.completion, 0.0, imitate=True) for s in random_groups(seed=7)]
    stats = policy.update(policy.new_member(seed=2), seqs, 1e-3, 0.0, config.Exaggeration(4.0, 1.0))
    assert stats["ex_loss"] == 0.0 and stats["grad_norm"] < 1e-6 and stats["ex_sequences"] == len(seqs)


def test_exaggeration_moves_the_member_away_from_the_base(base_model):
    policy = make_policy(base_model)
    member = policy.new_member(seed=23, b_std=0.05)
    seqs = [Sequence(s.prompt, s.completion, 0.0, imitate=True) for s in random_groups(seed=8)]

    @torch.no_grad()
    def kl_from_base():  # mean over completion positions of KL(member || base)
        policy._activate(member)
        kls = []
        for s in seqs:
            p, b = _full_logits(policy.model, s).log_softmax(-1), _full_logits(base_model, s).log_softmax(-1)
            kls.append((p.exp() * (p - b)).sum(-1))
        return float(torch.cat(kls).mean())
    kls = [kl_from_base()]
    for _ in range(4):
        policy.update(member, seqs, 3e-3, 0.0, config.Exaggeration(1.0, 1.0))
        kls.append(kl_from_base())
    assert all(b > a for a, b in zip(kls, kls[1:])) and kls[-1] > 2 * kls[0], kls


def test_scaled_adapter_changes_only_the_size_of_the_update(base_model, tmp_path):
    policy = make_policy(base_model)
    member = policy.new_member(seed=24, b_std=0.1)
    policy.save_adapter(member, tmp_path / "a")
    trainer.scale_adapter(tmp_path / "a", tmp_path / "a2", 2.0)
    assert (tmp_path / "a" / "adapter_config.json").read_bytes() == (tmp_path / "a2" / "adapter_config.json").read_bytes()
    scaled = policy.load_adapter(tmp_path / "a2")
    for name, t in member.lora.items():
        assert torch.equal(scaled[name], t * 2.0 if ".lora_B." in name else t), name
    layer = policy.model.base_model.model.model.layers[1].mlp.down_proj
    x = torch.randn(3, layer.in_features)
    with torch.no_grad():
        outs = []
        for lora in (member.lora, scaled):
            policy._activate(type(member)(lora))
            outs.append(layer(x))
        with policy.model.disable_adapter():
            base = layer(x)
    torch.testing.assert_close(outs[1] - base, 2.0 * (outs[0] - base), rtol=1e-5, atol=1e-6)
    with pytest.raises(FileExistsError):
        trainer.scale_adapter(tmp_path / "a", tmp_path / "a2", 1.5)


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
