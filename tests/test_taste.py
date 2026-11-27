"""Selective amplification in the whole member life cycle on the CPU (tiny model, stand-in sampler and scorer, the real
training stream on synthetic pools): the taste-shaped reward (its score, which rollouts it may touch, exact vanishing at
strength 0, sign symmetry, pilot E's clone layout) and practice relative to the base model."""

import copy
import math
import random
import statistics

import pytest
import torch

from evotaste import worker
from evotaste import member as member_module
from evotaste.member import load_lineage, practice_shares
from evotaste.records import read_json, read_jsonl, read_jsonl_gz
from evotaste.tasks import stable_seed
from evotaste.trainer import MemberWeights, Sequence, advantages, length_residuals, length_returns, taste_factors
from tests import engine_fakes
from tests.test_amplification import make_run, member_state

AMP = {"steps": 2, "prompts": 16, "rollouts": 4, "max_tokens": 12, "lr": 1e-3, "length": {"free": 8, "slope": 0.25},
       "imitate_within_free": True}
VARIANTS = {"own": 0.25, "neu": 0.0, "rev": -0.25}
MEMBERS = [{"id": "p", "seed": 1}] + [
    {"id": v, "seed": 2 + i, "parent": {"member": "p", "step": 2},
     "overrides": {"amplification": {"steps": 5, "taste": {"strength": s}}}} for i, (v, s) in enumerate(VARIANTS.items())]
CONFIG = {"name": "taste", "strata": {"path": "strata.tsv", "sha256": "0" * 64}, "seeding": {"method": "none"},
          "amplification": AMP, "probes": {"every": 2, "max_tokens": 12}, "culling": {"enabled": False},
          "trainer": {"check_tokens": 20, "check_rollouts": 4}, "members": MEMBERS}


def record_scoring(policy, calls):
    """Log every trainer scoring call as (kind, [(prompt, completion), ...])."""
    member, base = policy.logprobs, policy.base_logprobs

    def logprobs(weights, seqs, token_budget=None):
        calls.append(("member", [(tuple(s.prompt), tuple(s.completion)) for s in seqs]))
        return member(weights, seqs, token_budget)

    def base_logprobs(seqs, token_budget=None):
        calls.append(("base", [(tuple(s.prompt), tuple(s.completion)) for s in seqs]))
        return base(seqs, token_budget)
    policy.logprobs, policy.base_logprobs = logprobs, base_logprobs


@pytest.fixture(scope="module")
def finished(tmp_path_factory, base_model):
    mp = pytest.MonkeyPatch()
    mp.setattr(member_module, "score_many", engine_fakes.score_many)
    run = make_run(engine_fakes.write_run(tmp_path_factory.mktemp("taste"), CONFIG), base_model)
    calls, starts = [], {}
    record_scoring(run.policy, calls)  # after the base model's probe answers
    start_clone = member_module.Member.start_clone

    def recorded_start(self, ckpt):  # the state a clone starts from (its own checkpoint keeps no optimizer state)
        start_clone(self, ckpt)
        starts[self.lin.id] = (ckpt, {k: v.clone() for k, v in self.weights.lora.items()},
                               copy.deepcopy(self.weights.optimizer))
    mp.setattr(member_module.Member, "start_clone", recorded_start)
    worker.advance(run, [m["id"] for m in MEMBERS])
    run.starts = starts
    yield run, list(calls)
    mp.undo()


def history(run, lineage):
    return [r for r in read_jsonl(run.dir / "members" / lineage / "history.jsonl") if r["phase"] == "amp"]


def rollouts(run, lineage, step):
    return read_jsonl_gz(run.dir / "members" / lineage / "rollouts" / f"step_{step:06d}.jsonl.gz")


def checkpoint(run, lineage, step):
    return run.dir / "members" / lineage / "checkpoints" / f"step_{step:06d}"


def sampled_ids(record):
    return [int(x) for x in record["text"].split()]  # the stand-in sampler's text is its token ids


def groups_of(records):
    out = {}
    for r in records:
        out.setdefault(r["prompt"], []).append(r)
    return [out[j] for j in sorted(out)]


def eligible(r):
    return r["reward"] == 1.0 and r["completion_tokens"] <= AMP["length"]["free"]


def scores(run, weights, records):
    seqs = [Sequence(run.prompt(r["question"]), sampled_ids(r)) for r in records]
    return [float((m - b).sum()) / len(m) for m, b in zip(run.policy.logprobs(weights, seqs),
                                                          run.policy.base_logprobs(seqs))]


def test_the_score_is_the_sampling_weights_against_the_base_model_on_the_same_tokens(finished):
    run, calls = finished
    base_calls = [i for i, (kind, _) in enumerate(calls) if kind == "base"]
    assert len(base_calls) == 6  # own and rev at steps 2, 3, 4; nothing for the parent and the neutral clone
    for i in base_calls:  # the member's and the base model's pass on the very same prompt and completion ids
        assert calls[i - 1] == ("member", calls[i][1]) and calls[i][1]
    for lid in ("own", "rev"):
        for step in (2, 4):  # the weights before these updates are checkpoints
            records = [r for r in rollouts(run, lid, step) if eligible(r)]
            assert [r["taste_score"] for r in records] == pytest.approx(
                scores(run, MemberWeights(run.policy.load_adapter(checkpoint(run, lid, step))), records),
                rel=1e-6, abs=1e-7)
        records = [r for r in rollouts(run, lid, 4) if eligible(r)]
        after = scores(run, MemberWeights(run.policy.load_adapter(checkpoint(run, lid, 5))), records)
        seqs = [Sequence(run.prompt(r["question"]), sampled_ids(r)) for r in records]
        sampler = [(r["logprob"] - float(b.sum())) / len(b) for r, b in zip(records, run.policy.base_logprobs(seqs))]
        ours = [r["taste_score"] for r in records]
        off_sampler = statistics.fmean(abs(a - b) for a, b in zip(ours, sampler))  # the served bf16 copy
        off_after = statistics.fmean(abs(a - b) for a, b in zip(ours, after))  # the weights after the update
        assert off_sampler < 0.1 * off_after, (off_sampler, off_after)
        logged = [line["taste"]["member_vs_sampler"] for line in history(run, lid)]
        assert all(0 < x < 0.015 for x in logged), logged  # within the consistency check's limit, on every rollout


@pytest.mark.parametrize("lineage", ["own", "rev"])
def test_shaping_touches_only_correct_eligible_rewards_and_records_them_unshaped(finished, lineage):
    run, _ = finished
    strength = VARIANTS[lineage]
    lines = history(run, lineage)
    assert [line["step"] for line in lines] == [2, 3, 4] and any(line["taste"]["shaped"] for line in lines)
    for line in lines:
        records = rollouts(run, lineage, line["step"])
        groups = groups_of(records)
        assert {r["reward"] for r in records} <= {0.0, 1.0}  # the scorer's reward, as recorded
        assert line["reward"] == sum(r["reward"] for r in records) / len(records)
        factors, stats = taste_factors([[r["taste_score"] for r in g] for g in groups], strength, 2.0)
        assert [[r["taste_factor"] for r in g] for g in groups] == factors
        for key in ("shaped", "groups", "sigma", "skipped"):
            assert line["taste"][key] == stats[key]
        plain_abs = shaping_abs = 0.0
        for g, f in zip(groups, factors):
            tokens = [r["completion_tokens"] for r in g]
            if sum(eligible(r) for r in g) < 2:
                assert f == [1.0] * len(g)  # a group with fewer than two eligible rollouts is untouched
            for r in g:
                deduction = 0.25 * min(max((r["completion_tokens"] - 8) / 4, 0), 1)
                if eligible(r):
                    assert r["taste_score"] is not None and 0.5 <= r["taste_factor"] <= 1.5
                    assert r["return"] == pytest.approx(r["taste_factor"] - deduction, abs=1e-12)
                else:  # wrong, or right but beyond the free length: neither scored nor shaped
                    assert r["taste_score"] is None and r["taste_factor"] == 1.0
                    assert r["return"] == pytest.approx(r["reward"] - deduction, abs=1e-12)
            assert [r["weight"] for r in g] == advantages([r["return"] for r in g], "centered")
            plain = advantages(length_returns([r["reward"] for r in g], tokens, 12, 8, 0.25), "centered")
            plain_abs += sum(map(abs, plain))
            shaping_abs += sum(abs(r["weight"] - p) for r, p in zip(g, plain))
        assert line["taste"]["adv_plain"] == pytest.approx(plain_abs / len(records))
        assert line["taste"]["adv_shaping"] == pytest.approx(shaping_abs / len(records))
        s = [r["taste_score"] for r in records if eligible(r)]
        n = [r["completion_tokens"] for r in records if eligible(r)]
        assert line["taste"]["eligible"] == len(s)
        assert (line["taste"]["score_mean"], line["taste"]["score_sd"]) == pytest.approx(
            (statistics.fmean(s), statistics.pstdev(s)))
        assert line["taste"]["length_corr"] == pytest.approx(statistics.correlation(s, n))


def test_taste_factors():
    rng = random.Random(0)
    scores = [[rng.gauss(0, 1) if rng.random() < 0.6 else None for _ in range(4)] for _ in range(16)]
    scores[0], scores[1] = [0.3, None, None, None], [None] * 4
    scores[2] = [9.0, 0.0, 0.0, 0.0]  # an outlier: z clipped at +2 (and its group's others below 0)
    plus, stats = taste_factors(scores, 0.25, 2.0)
    minus, _ = taste_factors(scores, -0.25, 2.0)
    assert stats["shaped"] == sum(len([s for s in g if s is not None]) for g in scores
                                  if sum(s is not None for s in g) >= 2) and stats["skipped"] is None
    assert plus[0] == plus[1] == minus[0] == [1.0] * 4 and plus[2][0] == 1.5 and minus[2][0] == 0.5
    for g, p, m in zip(scores, plus, minus):
        for s, a, b in zip(g, p, m):
            assert 0.5 <= a <= 1.5 and a + b == pytest.approx(2.0, abs=1e-15)  # mirrored around 1
            if s is None:
                assert a == b == 1.0
    few = [[1.0, 2.0, None, None]] * 3 + [[None] * 4]  # 6 rollouts in groups of two or more eligible
    assert taste_factors(few, 0.25, 2.0) == ([[1.0] * 4] * 4, {
        "shaped": 0, "groups": 0, "sigma": 0.5, "skipped": "6 rollouts in groups with two or more eligible, "
                                                           "fewer than 8"})
    flat = [[0.7, 0.7, 0.7, None], [0.1, 0.1, None, None], [0.2, 0.2, 0.2, 0.2]]
    assert taste_factors(flat, 0.25, 2.0)[1]["skipped"] == "the scores do not vary within groups"


def single_member_run(tmp_path, base_model, name, taste):
    member = {"id": "m", "seed": 9, **({"overrides": {"amplification": {"taste": taste}}} if taste else {})}
    run = make_run(engine_fakes.write_run(tmp_path, {**CONFIG, "name": name, "members": [member]}), base_model)
    worker.advance(run, ["m"])
    return run


def test_strength_zero_is_the_update_without_taste_and_a_reversed_sign_mirrors_the_factors(
        tmp_path, base_model, monkeypatch):
    monkeypatch.setattr(member_module, "score_many", engine_fakes.score_many)
    runs = {name: single_member_run(tmp_path, base_model, name, taste)
            for name, taste in (("none", None), ("zero", {"strength": 0.0}), ("plus", {"strength": 0.25}),
                                ("minus", {"strength": -0.25}))}
    none, zero = member_state(runs["none"].dir), member_state(runs["zero"].dir)
    assert none.keys() == zero.keys() and any(p.suffix == ".pt" for p in none)
    for path in none:  # weights, optimizer state, rollouts and probes: bit for bit
        if path.suffix == ".pt":
            assert all(torch.equal(x, y) for x, y in zip(none[path], zero[path])), path
        else:
            assert none[path] == zero[path], path

    def lines(name):  # all but wall-clock times
        return [{k: v for k, v in line.items() if k not in ("time", "timing", "update_s")}
                for line in history(runs[name], "m")]
    assert lines("none") == lines("zero") and "taste" not in lines("zero")[0]
    final = "checkpoints/step_000002/adapter_model.safetensors"
    assert (runs["plus"].dir / "members" / "m" / final).read_bytes() != (
        runs["none"].dir / "members" / "m" / final).read_bytes()
    # step 0: an unseeded member is the base model, so every score is exactly 0 and nothing is shaped
    assert {r["taste_score"] for r in rollouts(runs["plus"], "m", 0) if eligible(r)} == {0.0}
    assert history(runs["plus"], "m")[0]["taste"]["skipped"] == "the scores do not vary within groups"
    # step 1: the same weights after the same step-0 update, hence the same rollouts, and mirrored factors
    plain, plus, minus = (rollouts(runs[n], "m", 1) for n in ("none", "plus", "minus"))
    assert [r["text"] for r in plus] == [r["text"] for r in minus] == [r["text"] for r in plain]
    assert [r["taste_score"] for r in plus] == [r["taste_score"] for r in minus]
    assert sum(r["taste_factor"] != 1.0 for r in plus) >= 8
    for a, b, c in zip(plus, minus, plain):
        assert a["taste_factor"] + b["taste_factor"] == pytest.approx(2.0, abs=1e-15)
        assert a["return"] - c["return"] == pytest.approx(c["return"] - b["return"], abs=1e-12)
        assert a["reward"] == b["reward"] == c["reward"]


def test_the_three_variants_start_from_the_parents_state_and_see_the_same_prompts(finished):
    run, _ = finished
    parent = checkpoint(run, "p", 2)
    lora = run.policy.load_adapter(parent)
    optimizer = torch.load(parent / "optimizer.pt", weights_only=True)
    assert optimizer["state"]  # the parent had made updates: a real AdamW state to inherit
    for v in VARIANTS:
        assert load_lineage(run, v).parent == {"lineage": "p", "step": 2}
        ckpt, weights, state = run.starts[v]
        assert ckpt == parent and weights.keys() == lora.keys() and all(torch.equal(weights[k], lora[k]) for k in lora)
        assert state["param_groups"] == optimizer["param_groups"] and state["state"].keys() == optimizer["state"].keys()
        assert all(torch.equal(state["state"][k][f], optimizer["state"][k][f])
                   for k in optimizer["state"] for f in ("step", "exp_avg", "exp_avg_sq"))
        own = checkpoint(run, v, 2)  # the clone's own copy, committed with its first probe
        assert (own / "adapter_model.safetensors").read_bytes() == (parent / "adapter_model.safetensors").read_bytes()
    reqs = {r.seed: r for r in run.sampler.requests}
    seeds = [m["seed"] for m in MEMBERS[1:]]
    for t in (2, 3, 4):
        prompts = [[reqs[stable_seed("rollout", seed, t, j)].prompt for j in range(16)] for seed in seeds]
        assert prompts[0] == prompts[1] == prompts[2]
        texts = [[r["text"] for r in rollouts(run, v, t)] for v in VARIANTS]
        assert len({tuple(x) for x in texts}) == 3  # sampled with each clone's own seeds
        assert len({tuple(r["uid"] for r in rollouts(run, v, t)) for v in VARIANTS}) == 1


def test_length_residuals():
    rng = random.Random(1)
    scores = [[rng.gauss(0, 1) if rng.random() < 0.7 else None for _ in range(4)] for _ in range(12)]
    scores[0] = [0.4, None, None, None]  # a group with one eligible rollout is not part of the fit
    tokens = [[rng.randint(5, 900) for _ in range(4)] for _ in range(12)]
    out, fit = length_residuals(scores, tokens)
    fitted = [(j, k) for j, g in enumerate(scores) if sum(s is not None for s in g) >= 2
              for k, s in enumerate(g) if s is not None]
    x = [math.log(tokens[j][k]) for j, k in fitted]
    r = [out[j][k] for j, k in fitted]
    assert abs(sum(r)) < 1e-9 and abs(sum(a * b for a, b in zip(r, x))) < 1e-9  # orthogonal to 1 and log length
    for j, k in fitted:
        assert out[j][k] == pytest.approx(scores[j][k] - fit["length_intercept"] - fit["length_slope"]
                                          * math.log(tokens[j][k]))
    assert out[0] == scores[0] and all(out[j][k] is None for j, g in enumerate(scores) for k, s in enumerate(g)
                                       if s is None)
    same = length_residuals([[1.0, 2.0], [3.0, None]], [[7, 7], [7, 7]])
    assert same == ([[-0.5, 0.5], [3.0, None]], {"length_intercept": 1.5, "length_slope": 0.0})
    assert length_residuals([[1.0, None]], [[5, 6]]) == ([[1.0, None]], {"length_intercept": None,
                                                                       "length_slope": None})


def test_the_length_neutral_score_shapes_on_its_residual(tmp_path, base_model, monkeypatch):
    monkeypatch.setattr(member_module, "score_many", engine_fakes.score_many)
    run = single_member_run(tmp_path, base_model, "resid", {"strength": 0.25, "residualize": True})
    shaped = 0
    for line in history(run, "m")[1:]:  # step 0: an unseeded member's scores are all 0
        groups = groups_of(rollouts(run, "m", line["step"]))
        scores = [[r["taste_score"] for r in g] for g in groups]
        residuals, fit = length_residuals(scores, [[r["completion_tokens"] for r in g] for g in groups])
        assert {k: line["taste"][k] for k in fit} == fit and fit["length_slope"] is not None
        factors, stats = taste_factors(residuals, 0.25, 2.0)
        assert [[r["taste_factor"] for r in g] for g in groups] == factors
        assert line["taste"]["shaped"] == stats["shaped"]
        assert factors != taste_factors(scores, 0.25, 2.0)[0]  # not the raw score's factors
        shaped += stats["shaped"]
    assert shaped > 0


def test_practice_relative_to_the_base_model(tmp_path, base_model, monkeypatch):
    monkeypatch.setattr(member_module, "score_many", engine_fakes.score_many)
    settings = {"eta": 4.0, "decay": 0.8, "floor": 0.25, "ceiling": 4.0, "reference": "base"}
    data = {**CONFIG, "name": "rel", "amplification": {**AMP, "steps": 3, "practice": settings},
            "members": [{"id": "r", "seed": 1}]}
    run = make_run(engine_fakes.write_run(tmp_path, data), base_model)
    rates = {uid: int(uid.rsplit("-", 1)[1]) % 9 / 8 for uid in run.streams.strata}  # census pass rates
    run.base_pass_rates = rates
    worker.advance(run, ["r"])
    p = run.cfg.settings(run.cfg.member("r"))[1].practice
    averages, drawn = {s: 0.0 for s in run.streams.source_shares}, {}  # no gain over the base model yet
    for line in history(run, "r"):
        assert line["practice"]["averages"] == averages  # bit-exact: the same float operations, in order
        shares = practice_shares(run.streams.source_shares, averages, p)
        assert line["practice"]["shares"] == shares
        tasks = [r for r in rollouts(run, "r", line["step"]) if r["k"] == 0]
        assert [r["uid"] for r in tasks] == [t.uid for t in run.streams.member_batch(line["step"], 16, shares, drawn)]
        base = {}
        for r in tasks:
            base.setdefault(r["family"], []).append(rates[r["uid"]])
        base = {f: sum(v) / len(v) for f, v in sorted(base.items())}
        assert line["practice"]["base"] == base and base.keys() == line["reward_by_family"].keys()
        for s, reward in line["reward_by_family"].items():
            averages[s] = p.decay * averages[s] + (1 - p.decay) * (reward - base[s])
        for r in tasks:
            drawn[f"{r['family']}/{r['stratum']}"] = drawn.get(f"{r['family']}/{r['stratum']}", 0) + 1
    assert len({tuple(sorted(line["practice"]["shares"].items())) for line in history(run, "r")}) == 3  # they move
    assert read_json(checkpoint(run, "r", 3) / "practice.json") == {"averages": averages, "drawn": drawn}
