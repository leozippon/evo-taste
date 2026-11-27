"""Consolidating one's own frontier in the whole member life cycle on the CPU (tiny model, stand-in sampler and scorer,
the real training stream on synthetic pools): the item record and its rebuild from a parent's rollouts, `own` and
`matched` revisits, the stream part shared by the three arms of a parent, replay imitation, and resume and clone start
with the record."""

import copy
import itertools
import random

import pytest
import torch

from evotaste import config, worker
from evotaste import member as member_module
from evotaste.member import (ITEMS_FILE, REVISIT_EPSILON, Item, Member, in_frontier, load_lineage, read_items,
                             record_attempts, revisit_plan, revisit_slots, weighted_draws)
from evotaste.records import read_jsonl, read_jsonl_gz
from evotaste.tasks import rewrap, stable_seed
from tests import engine_fakes
from tests.test_amplification import SUITES, make_run
from tests.test_taste import history, rollouts, sampled_ids

PROMPTS, FREE = 16, 8
AMP = {"steps": 2, "prompts": PROMPTS, "rollouts": 4, "max_tokens": 12, "lr": 1e-3,
       "length": {"free": FREE, "slope": 0.25}, "imitate_within_free": True}
OWN, MATCHED = {"share": 0.25, "source": "own"}, {"share": 0.25, "source": "matched"}
ARMS = {"fro": OWN, "mat": MATCHED, "pln": None}  # pilot F's arms, cloned at step 2


def config_with(parent_revisit=None, extra=()):
    """The parent p (taste-shaped, so that shaped returns differ from rewards) and its clones at step 2."""
    parent = {"id": "p", "seed": 1, "overrides": {"amplification": {
        "taste": {"strength": 0.25}, **({"revisit": parent_revisit} if parent_revisit else {})}}}
    clones = [{"id": arm, "seed": 2 + i, "parent": {"member": "p", "step": 2},
               "overrides": {"amplification": {"steps": 5, **({"revisit": rv} if rv else {})}}}
              for i, (arm, rv) in enumerate(list(ARMS.items()) + list(extra))]
    return {"name": "revisit", "strata": {"path": "strata.tsv", "sha256": "0" * 64}, "seeding": {"method": "none"},
            "amplification": AMP, "probes": {"every": 2, "max_tokens": 12}, "culling": {"enabled": False},
            "trainer": {"check_tokens": 20, "check_rollouts": 4}, "members": [parent] + clones}


RATES = {uid: int(uid.rsplit("-", 1)[1]) % 9 / 8 for uid in SUITES.strata}  # census pass rates


def prepared(tmp_path, base_model, data):
    run = make_run(engine_fakes.write_run(tmp_path, data), base_model)
    run.base_pass_rates = RATES
    return run


def slots_of(data):
    return [m["id"] for m in data["members"]]


@pytest.fixture(scope="module")
def runs(tmp_path_factory, base_model):
    """A: the parent keeps no record, so its clones rebuild theirs from its rollouts. B: the same with the parent
    keeping a record at share 0 (the shared batch), and a fourth clone with replay imitation."""
    mp = pytest.MonkeyPatch()
    mp.setattr(member_module, "score_many", engine_fakes.score_many)
    a = prepared(tmp_path_factory.mktemp("rebuilt"), base_model, config_with())
    worker.advance(a, slots_of(config_with()))
    data = config_with({"share": 0.0, "source": "own"}, [("rep", {**OWN, "imitate": 0.5})])
    b = prepared(tmp_path_factory.mktemp("kept"), base_model, data)
    updates, replays = [], {}
    real_update, real_replay = b.policy.update, Member.replay_sequences

    def update(weights, seqs, *args):
        updates.append([(s.prompt, s.completion) for s in seqs if s.replay])
        return real_update(weights, seqs, *args)

    def replay_sequences(self, revisited):
        out = real_replay(self, revisited)
        replays[(self.lin.id, self.lin.step)] = [copy.copy(s) for s in out]
        return out
    b.policy.update = update
    mp.setattr(Member, "replay_sequences", replay_sequences)
    worker.advance(b, slots_of(data))
    b.updates, b.replays = updates, replays
    yield a, b
    mp.undo()


def items_at(run, lineage, step):
    return read_items(run.dir / "members" / lineage / "checkpoints" / f"step_{step:06d}" / ITEMS_FILE)


def chain_rollouts(run, lineage, before):
    """(step, records) of a lineage's history before `before`: its parent's steps before the clone point, then its
    own."""
    lin = load_lineage(run, lineage)
    start = lin.parent["step"] if lin.parent else 0
    earlier = chain_rollouts(run, lin.parent["lineage"], start) if lin.parent else []
    return earlier + [(t, rollouts(run, lineage, t)) for t in range(start, before)]


def expected_record(steps, live_from):
    """The record by its definition, from rollout records: answers only for first successes at steps >= live_from."""
    items = {}
    for t, records in steps:
        for r in records:  # prompt-major, as recorded
            item = items.setdefault(r["uid"], Item())
            item.attempts += 1
            if r["reward"] == 1.0:
                item.successes += 1
                if item.first is None:
                    item.first = t
                    if t >= live_from and r["completion_tokens"] <= FREE:
                        item.wrapper, item.answer = r["wrapper"], sampled_ids(r)
    return items


def without_answers(items):
    return {u: Item(i.attempts, i.successes, i.first) for u, i in items.items()}


def test_the_record_counts_unshaped_successes_and_keeps_first_answers_within_the_free_length(runs):
    a, b = runs
    shaped = [r for t, records in chain_rollouts(b, "p", 2) for r in records if r["reward"] == 1.0 and (
        r["return"] != 1.0 - 0.25 * min(max((r["completion_tokens"] - FREE) / 4, 0), 1))]
    assert shaped  # the update saw shaped returns; the record counts the scorer's reward
    assert items_at(b, "p", 0) == {} and items_at(b, "p", 2) == expected_record(chain_rollouts(b, "p", 2), 0)
    kept = items_at(b, "p", 2)
    firsts = [i for i in kept.values() if i.first is not None]
    assert any(i.answer is None for i in firsts) and any(i.answer for i in firsts)  # some first successes too long
    for lineage, run in (("fro", a), ("mat", a), ("fro", b), ("rep", b)):
        live_from = 2 if run is a else 0  # in A the record up to the clone point is rebuilt: no answers
        for step in (4, 5):
            assert items_at(run, lineage, step) == expected_record(chain_rollouts(run, lineage, step), live_from)


def test_a_rebuilt_record_is_the_record_the_parent_would_have_kept(runs):
    a, b = runs
    assert all(not (a.dir / "members" / "p" / "checkpoints" / s / ITEMS_FILE).exists() for s in ("step_000000",
                                                                                             "step_000002"))
    rebuilt, kept = items_at(a, "fro", 2), items_at(b, "p", 2)
    assert rebuilt == without_answers(kept) and len(rebuilt) == 2 * PROMPTS
    for arm in ("fro", "mat", "rep"):  # inherited bit-exactly by the clones of a parent that keeps a record
        assert items_at(b, arm, 2) == kept
    assert items_at(a, "mat", 2) == rebuilt and not (a.dir / "members" / "pln" / "checkpoints" / "step_000002" /
                                                     ITEMS_FILE).exists()  # no revisit key, no record
    # without replay imitation the answers play no part: A's and B's clones train identically
    for arm in ("fro", "mat"):
        for t in (2, 3, 4):
            assert rollouts(a, arm, t) == rollouts(b, arm, t)
        final = "checkpoints/step_000005/adapter_model.safetensors"
        assert (a.dir / "members" / arm / final).read_bytes() == (b.dir / "members" / arm / final).read_bytes()


def test_share_zero_is_the_shared_batch_and_the_update_without_revisits(runs):
    a, b = runs
    for t in (0, 1):
        assert [r["uid"] for r in rollouts(b, "p", t)][::4] == [x.uid for x in b.streams.train_batch(t, PROMPTS)]
        assert [{k: v for k, v in r.items() if k != "revisit"} for r in rollouts(b, "p", t)] == rollouts(a, "p", t)
        assert not any(r["revisit"] for r in rollouts(b, "p", t))
    weights = "members/p/checkpoints/step_000002/adapter_model.safetensors"
    assert (a.dir / weights).read_bytes() == (b.dir / weights).read_bytes()
    oa, ob = (torch.load(r.dir / "members/p/checkpoints/step_000002/optimizer.pt", weights_only=True)["state"]
              for r in (a, b))
    assert all(torch.equal(oa[k][f], ob[k][f]) for k in oa for f in ("exp_avg", "exp_avg_sq"))
    clock = ("time", "timing", "update_s", "revisit")
    strip = [[{k: v for k, v in x.items() if k not in clock and k != "taste"} for x in history(r, "p")] for r in (a, b)]
    assert strip[0] == strip[1]
    assert [x["revisit"]["slots"] for x in history(b, "p")] == [0, 0]
    assert [(x["revisit"]["own"], x["revisit"]["revisits"], x["revisit"]["shortfall"]) for x in history(b, "p")] == [
        ([], [], 0)] * 2


def record_before(run, lineage, step):
    """The member's record at the start of `step`, by its definition."""
    return expected_record(chain_rollouts(run, lineage, step), 99)


def plan_of(run, lineage, step, settings):
    lin = load_lineage(run, lineage)
    return revisit_plan(run.streams.train_batch(step, PROMPTS), record_before(run, lineage, step),
                        config.Revisit(**settings), RATES, run.streams.train_task, (run.cfg.seed, lin.seed, step))


def test_own_revisits_are_distinct_frontier_items_outside_the_shared_batch(runs):
    a, _ = runs
    n = revisit_slots(0.25, PROMPTS)
    for t, line in zip((2, 3, 4), history(a, "fro")):
        record = record_before(a, "fro", t)
        frontier = {u for u, i in record.items() if in_frontier(i, 4)}
        stream = a.streams.train_batch(t, PROMPTS)
        plan = line["revisit"]
        assert plan["frontier"] == len(frontier) and plan["slots"] == n == 4 and plan["revisits"] == plan["own"]
        assert len(set(plan["own"])) == len(plan["own"]) == min(n, len(frontier - {x.uid for x in stream}))
        assert set(plan["own"]) <= frontier - {x.uid for x in stream}
        uids = [r["uid"] for r in rollouts(a, "fro", t)][::4]
        assert uids == [x.uid for x in stream[:PROMPTS - len(plan["own"])]] + plan["own"]
        assert [r["revisit"] for r in rollouts(a, "fro", t)][::4] == [False] * (PROMPTS - n) + [True] * n
        revisits = [r for r in rollouts(a, "fro", t) if r["revisit"]]
        assert all(r["wrapper"] == a.streams.train_task(r["uid"]).wrapper and r["stratum"] ==
                   a.streams.train_task(r["uid"]).stratum for r in revisits)  # the pool's own task
        assert plan["reward"] == sum(r["reward"] for r in revisits) / len(revisits)
        batch, again = plan_of(a, "fro", t, OWN)
        assert {k: v for k, v in plan.items() if "reward" not in k} == again  # a pure function of the record
        assert [x.uid for x in batch] == uids


def test_matched_revisits_are_the_nearest_shown_items_outside_the_frontier(runs):
    a, _ = runs
    for t, line in zip((2, 3, 4), history(a, "mat")):
        record = record_before(a, "mat", t)
        frontier = {u for u, i in record.items() if in_frontier(i, 4)}
        stream = {x.uid for x in a.streams.train_batch(t, PROMPTS)}
        plan = line["revisit"]
        own, _ = plan_of(a, "mat", t, OWN)  # what `own` would have drawn for this member at this step
        assert plan["own"] == [x.uid for x in own[PROMPTS - len(plan["own"]):]] and len(plan["own"]) == 4
        candidates = {u for u, i in record.items() if u not in frontier and u not in stream and i.attempts >= 4}
        taken = set()
        for x, y in zip(plan["own"], plan["revisits"]):
            options = [c for c in candidates - taken if c.split("/")[0] == x.split("/")[0]]
            if y is None:
                assert not options
                continue
            assert y in options and y not in frontier and record[y].attempts >= 4
            assert abs(RATES[y] - RATES[x]) == min(abs(RATES[c] - RATES[x]) for c in options)
            taken.add(y)
        chosen = [y for y in plan["revisits"] if y is not None]
        assert [r["uid"] for r in rollouts(a, "mat", t)][::4][PROMPTS - len(chosen):] == chosen
        assert plan["shortfall"] == 4 - len(chosen)


def test_the_three_arms_see_the_same_stream_prompts(runs):
    a, _ = runs
    reqs = {r.seed: r for r in a.sampler.requests}
    seeds = {arm: load_lineage(a, arm).seed for arm in ARMS}
    for t in (2, 3, 4):
        prompts = {arm: [reqs[stable_seed("rollout", seed, t, j)].prompt for j in range(PROMPTS)]
                   for arm, seed in seeds.items()}
        shared = [a.prompt(x.question) for x in a.streams.train_batch(t, PROMPTS)]
        assert prompts["pln"] == shared
        for arm in ("fro", "mat"):
            plan = history(a, arm)[t - 2]["revisit"]
            stream_part = PROMPTS - sum(u is not None for u in plan["revisits"])
            assert prompts[arm][:stream_part] == shared[:stream_part]
        assert prompts["fro"][-4:] != prompts["mat"][-4:]


def test_replay_imitation_uses_the_stored_first_answer_behind_its_own_prompt(runs):
    _, b = runs
    first = {}  # uid -> its first successful rollout in rep's history
    for t, records in chain_rollouts(b, "rep", 5):
        for r in records:
            if r["reward"] == 1.0:
                first.setdefault(r["uid"], r)
    replayed, beyond = 0, 0
    for line in history(b, "rep"):
        expected = []
        for uid in line["revisit"]["revisits"]:  # own revisits: frontier items, each solved before
            r = first[uid]
            assert r["step"] < line["step"]
            if r["completion_tokens"] > FREE:  # beyond the free length: never stored, never replayed
                beyond += 1
                continue
            task = b.streams.train_task(uid)
            asked = task if r["wrapper"] == task.wrapper else rewrap(task, r["wrapper"])
            expected.append((b.prompt(asked.question), sampled_ids(r)))
        got = [(s.prompt, s.completion) for s in b.replays[("rep", line["step"])]]
        assert got == expected and expected in b.updates  # the update received exactly these, flagged `replay`
        assert all(s.replay and s.weight == 0.0 and not s.imitate for s in b.replays[("rep", line["step"])])
        assert (line["rp_sequences"], line["rp_tokens"]) == (len(expected), sum(len(c) for _, c in expected))
        assert (line["rp_loss"] > 0) == bool(expected)
        replayed += len(expected)
    assert replayed > 0 and beyond > 0
    assert "rp_loss" not in history(b, "fro")[0] and "rp_sequences" not in history(b, "fro")[0]  # imitate 0


# -- the draws, unit by unit --------------------------------------------------------------------------------------

def test_the_frontier_rule():
    cases = {(4, 0): False, (4, 1): True, (4, 2): True, (4, 3): False, (8, 5): True, (8, 6): False, (3, 1): False,
             (12, 8): True, (12, 9): False}
    for (attempts, successes), inside in cases.items():
        assert in_frontier(Item(attempts, successes), 4) == inside, (attempts, successes)
    assert in_frontier(Item(3, 1), 3) and not in_frontier(Item(4, 1), 5)
    assert [revisit_slots(s, 48) for s in (0.0, 0.001, 0.33, 0.25, 1.0)] == [0, 1, 16, 12, 48]


def test_weighted_draws_follow_successive_sampling():
    uids, weights = ["a", "b", "c", "d"], [1 - r + REVISIT_EPSILON for r in (0.0, 0.5, 0.75, 1.0)]
    total = sum(weights)
    exact = {}
    for x, y in itertools.permutations(range(4), 2):  # P(x first, then y)
        exact[(uids[x], uids[y])] = weights[x] / total * weights[y] / (total - weights[x])
    counts, trials = {}, 40000
    for i in range(trials):
        pair = tuple(weighted_draws(uids, weights, 2, random.Random(i)))
        counts[pair] = counts.get(pair, 0) + 1
    assert set(counts) <= set(exact)  # never the same uid twice
    for pair, p in exact.items():
        assert abs(counts.get(pair, 0) / trials - p) < 4 * (p * (1 - p) / trials) ** 0.5 + 1e-3, pair
    assert weighted_draws(uids, weights, 9, random.Random(0)) != [] and sorted(
        weighted_draws(uids, weights, 9, random.Random(0))) == uids  # fewer than asked: all of them


def test_a_short_frontier_is_filled_from_the_shared_batch_and_held_out_items_are_refused(runs):
    a, _ = runs
    stream = a.streams.train_batch(7, PROMPTS)
    pool = [t.uid for t in a.streams.train_batch(9, PROMPTS)]
    record = {pool[0]: Item(4, 1), pool[1]: Item(4, 2), pool[2]: Item(4, 0), pool[3]: Item(4, 4)}
    batch, plan = revisit_plan(stream, record, config.Revisit(0.25, "own"), RATES, a.streams.train_task, (0, 1, 7))
    assert sorted(plan["own"]) == sorted(pool[:2]) and plan["shortfall"] == 2
    assert [t.uid for t in batch] == [t.uid for t in stream[:PROMPTS - 2]] + plan["own"]
    empty, plan = revisit_plan(stream, {}, config.Revisit(0.25, "matched"), RATES, a.streams.train_task, (0, 1, 7))
    assert empty == stream and plan["shortfall"] == 4 and plan["own"] == plan["revisits"] == []
    _, plan = revisit_plan(stream, record, config.Revisit(0.25, "matched"), RATES, a.streams.train_task, (0, 1, 7))
    assert all(m in (None, pool[2], pool[3]) for m in plan["revisits"])  # never a frontier item
    held_out = SUITES.held_out.uid
    with pytest.raises(ValueError, match="not in the training stream's pools"):
        revisit_plan(stream, {held_out: Item(4, 1)}, config.Revisit(0.25, "own"), RATES, a.streams.train_task,
                     (0, 1, 7))
    with pytest.raises(ValueError, match="not in the training stream's pools"):
        revisit_plan(stream, {**record, held_out: Item(4, 0)}, config.Revisit(0.25, "matched"), RATES,
                     a.streams.train_task, (0, 1, 7))


def test_record_attempts_keeps_the_first_success_only():
    items = {}
    record_attempts(items, 3, [("x", "box", 0.0, 5, [1]), ("x", "box", 1.0, 9, [2] * 9), ("x", "box", 1.0, 4, [3])], 8)
    record_attempts(items, 4, [("x", "final_line", 1.0, 2, [4, 5]), ("y", "box", 1.0, 3, [6, 7, 8])], 8)
    assert items == {"x": Item(4, 3, 3, None, None), "y": Item(1, 1, 4, "box", [6, 7, 8])}


# -- resume and clone start -----------------------------------------------------------------------------------------

class Killed(Exception):
    pass


def member_files(run_dir):
    out = {}
    for p in sorted((run_dir / "members").rglob("*")):
        if p.suffix == ".safetensors" or p.name == ITEMS_FILE or p.parent.name == "rollouts":
            out[p.relative_to(run_dir)] = read_jsonl_gz(p) if p.suffix == ".gz" else p.read_bytes()
        elif p.suffix == ".pt":
            state = torch.load(p, weights_only=True)["state"]
            out[p.relative_to(run_dir)] = [state[k][f] for k in sorted(state) for f in ("exp_avg", "exp_avg_sq")]
    return out


def test_a_resumed_run_is_the_uninterrupted_run_with_its_record(runs, tmp_path, base_model, monkeypatch):
    """Killed during step 4 of the clones, after their step-2 checkpoint (the inherited or rebuilt record) and the
    probe at 4, and started again: every member ends bit-identical to the uninterrupted run, records included."""
    _, b = runs
    monkeypatch.setattr(member_module, "score_many", engine_fakes.score_many)
    data = config_with({"share": 0.0, "source": "own"}, [("rep", {**OWN, "imitate": 0.5})])
    run = prepared(tmp_path, base_model, data)
    real = Member.amplify

    def amplify(self, *args):
        if self.lin.id == "mat" and self.lin.step == 4:
            raise Killed
        return real(self, *args)
    monkeypatch.setattr(Member, "amplify", amplify)
    with pytest.raises(Killed):
        worker.advance(run, slots_of(data))
    assert load_lineage(run, "mat").step == 4
    monkeypatch.setattr(Member, "amplify", real)
    again = make_run(run.dir, base_model, prepared=True)
    again.base_pass_rates = RATES
    worker.advance(again, slots_of(data))
    ours, theirs = member_files(run.dir), member_files(b.dir)
    assert ours.keys() == theirs.keys() and sum(p.name == ITEMS_FILE for p in ours) >= 12
    for path in ours:
        if path.suffix == ".pt":
            assert all(torch.equal(x, y) for x, y in zip(ours[path], theirs[path])), path
        else:
            assert ours[path] == theirs[path], path
    assert [line["step"] for line in read_jsonl(run.dir / "members" / "mat" / "history.jsonl")] == [2, 3, 4]


def test_a_lost_record_is_refused(runs, tmp_path, base_model):
    import shutil

    _, b = runs
    d = tmp_path / "copy"
    shutil.copytree(b.dir, d)
    (d / "members" / "fro" / "checkpoints" / "step_000005" / ITEMS_FILE).unlink()
    run = make_run(d, base_model, prepared=True)
    member = Member(run, load_lineage(run, "fro"))
    member.lin.status = "active"
    with pytest.raises(RuntimeError, match=f"{ITEMS_FILE} is missing"):
        member.restore()
    (d / "members" / "p" / "checkpoints" / "step_000002" / ITEMS_FILE).unlink()  # a parent that keeps a record
    clone = Member(run, load_lineage(run, "mat"))
    clone.lin.status, clone.lin.step = "new", 0
    with pytest.raises(RuntimeError, match="keeps an item record"):
        clone.start_clone(d / "members" / "p" / "checkpoints" / "step_000002")
    (d / "members" / "p" / "rollouts" / "step_000001.jsonl.gz").unlink()
    with pytest.raises(RuntimeError, match="step_000001.jsonl.gz is missing"):
        member_module.rebuild_items(run, "p", 2)
