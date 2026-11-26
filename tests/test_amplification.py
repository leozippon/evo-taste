"""Pilot D's mechanisms in the whole member life cycle on the CPU (tiny model, stand-in sampler and scorer, the real
training stream on synthetic pools): eligibility of the imitation terms, exaggeration, self-chosen practice with its
record across resume and clones, and the scaled-probe and deviation commands on the stored probes."""

import csv

import pytest
import torch

from evotaste import evaluate, worker
from evotaste import member as member_module
from evotaste.member import PRACTICE_START, Member, load_lineage, practice_shares
from evotaste.records import read_json, read_jsonl, read_jsonl_gz
from evotaste.tasks import STREAM_VERSION, Streams, stable_seed, train_cell
from evotaste.trainer import MemberWeights, Sequence
from tests import engine_fakes
from tests.conftest import make_policy
from tests.test_tasks import FakeSuites, standard

PRACTICE = {"eta": 4.0, "decay": 0.8, "floor": 0.25, "ceiling": 4.0}
MEMBERS = [
    {"id": "plain", "seed": 1},
    {"id": "zero", "seed": 2, "overrides": {"amplification": {"practice": {**PRACTICE, "eta": 0.0}}}},
    {"id": "prac", "seed": 3, "overrides": {"amplification": {"practice": PRACTICE,
                                                              "exaggeration": {"gamma": 1.0, "weight": 0.5}}}},
    {"id": "pc", "seed": 4, "parent": {"member": "prac", "step": 2}},
]
CONFIG = {"name": "mech", "strata": {"path": "strata.tsv", "sha256": "0" * 64}, "seeding": {"method": "none"},
          "amplification": {"steps": 4, "prompts": 12, "rollouts": 3, "max_tokens": 12, "lr": 1e-3,
                            "length": {"free": 6, "slope": 0.25}, "self_imitation": 0.5,
                            "imitate_within_free": True},
          "probes": {"every": 2, "max_tokens": 12}, "culling": {"enabled": False},
          "trainer": {"check_tokens": 20, "check_rollouts": 4}, "members": MEMBERS}
SLOTS = [m["id"] for m in MEMBERS]
SUITES = FakeSuites(standard)


def make_run(run_dir, base_model, prepared=False):
    run = engine_fakes.FakeRun(run_dir)
    run.streams = Streams(SUITES, run.cfg.seed, strata=SUITES.strata, version=STREAM_VERSION)  # the real stream
    run.policy = make_policy(base_model)
    run.sampler = engine_fakes.FakeSampler(run.policy, run_dir.parent)
    if not prepared:
        for context in [None, *run.contexts]:
            worker.base_probe(run, run.sampler, context)
    return run


@pytest.fixture(scope="module")
def finished(tmp_path_factory, base_model):
    mp = pytest.MonkeyPatch()
    mp.setattr(member_module, "score_many", engine_fakes.score_many)
    run = make_run(engine_fakes.write_run(tmp_path_factory.mktemp("mech"), CONFIG), base_model)
    worker.advance(run, SLOTS)
    yield run
    mp.undo()


def history(run, lineage):
    return [r for r in read_jsonl(run.dir / "members" / lineage / "history.jsonl") if r["phase"] == "amp"]


def rollouts(run, lineage, step):
    return read_jsonl_gz(run.dir / "members" / lineage / "rollouts" / f"step_{step:06d}.jsonl.gz")


def test_imitation_terms_use_successes_within_the_free_length(finished):
    for lid in ("plain", "prac", "pc"):
        for line in history(finished, lid):
            records = rollouts(finished, lid, line["step"])
            eligible = [r for r in records if r["reward"] == 1.0 and r["completion_tokens"] <= 6]
            assert 0 < len(eligible) < sum(r["reward"] == 1.0 for r in records)  # some successes are too long
            assert (line["si_sequences"], line["si_tokens"]) == (len(eligible),
                                                                 sum(r["completion_tokens"] for r in eligible))
            exaggerated = lid != "plain"
            assert line["ex_sequences"] == (len(eligible) if exaggerated else 0)
            assert line["adv_sequences"] == sum(r["weight"] != 0.0 for r in records)
            assert line["check"]["rollouts"] > 0  # the consistency check ran (and passed) on every step
    prac = history(finished, "prac")
    assert prac[0]["ex_loss"] == 0.0 and all(r["ex_loss"] > 0 for r in prac[1:])  # an unseeded member starts as base
    assert all(r["ex_loss"] == 0.0 for r in history(finished, "plain"))


def test_members_without_practice_or_with_eta_zero_see_the_shared_prompts(finished):
    reqs = {r.seed: r for r in finished.sampler.requests}
    for t in range(4):
        shared = [finished.prompt(task.question) for task in finished.streams.train_batch(t, 12)]
        for seed in (1, 2):
            assert [reqs[stable_seed("rollout", seed, t, j)].prompt for j in range(12)] == shared
        assert [r["uid"] for r in rollouts(finished, "zero", t)][::3] == [
            task.uid for task in finished.streams.train_batch(t, 12)]
    zero = history(finished, "zero")
    assert all(line["practice"]["shares"] == pytest.approx(finished.streams.source_shares) for line in zero)
    assert read_json(finished.dir / "members" / "zero" / "checkpoints" / "step_000004" / "practice.json")[
        "drawn"] == {}


def test_practice_follows_the_members_own_record(finished):
    run, settings = finished, finished.cfg.settings(finished.cfg.member("prac"))[1].practice
    lines = history(run, "prac")
    averages = {s: PRACTICE_START for s in run.streams.source_shares}
    drawn = {}
    for line in lines:
        assert line["practice"]["averages"] == averages  # bit-exact: the same float operations, in order
        shares = practice_shares(run.streams.source_shares, averages, settings)
        assert line["practice"]["shares"] == shares
        for s, share in shares.items():
            assert settings.floor * run.streams.source_shares[s] - 1e-12 <= share <= (
                settings.ceiling * run.streams.source_shares[s] + 1e-12)
        tasks = [r for r in rollouts(run, "prac", line["step"]) if r["k"] == 0]
        assert [r["uid"] for r in tasks] == [t.uid for t in run.streams.member_batch(line["step"], 12, shares, drawn)]
        for r in tasks:
            drawn[f"{r['family']}/{r['stratum']}"] = drawn.get(f"{r['family']}/{r['stratum']}", 0) + 1
        for s, reward in line["reward_by_family"].items():
            averages[s] = settings.decay * averages[s] + (1 - settings.decay) * reward
    assert len({tuple(sorted(line["practice"]["shares"].items())) for line in lines}) == len(lines)  # they move
    final = read_json(run.dir / "members" / "prac" / "checkpoints" / "step_000004" / "practice.json")
    assert final == {"averages": averages, "drawn": drawn}


def test_a_clone_inherits_the_practice_record_and_draws_its_parents_batch(finished):
    run = finished
    parent = run.dir / "members" / "prac" / "checkpoints" / "step_000002" / "practice.json"
    clone = run.dir / "members" / "pc" / "checkpoints" / "step_000002" / "practice.json"
    assert clone.read_bytes() == parent.read_bytes() and read_json(parent)["drawn"]
    a, b = rollouts(run, "prac", 2), rollouts(run, "pc", 2)
    assert [r["uid"] for r in a] == [r["uid"] for r in b] and [r["text"] for r in a] != [r["text"] for r in b]
    assert history(run, "pc")[0]["practice"] == history(run, "prac")[2]["practice"]


def member_state(run_dir):
    out = {}
    for p in sorted((run_dir / "members").rglob("*")):
        if p.suffix in (".safetensors", ".json") and p.name != "state.json" or p.name.startswith("step_0000") and (
                p.suffix == ".gz"):
            out[p.relative_to(run_dir)] = (read_jsonl_gz(p) if p.suffix == ".gz" else p.read_bytes())
        elif p.suffix == ".pt":
            state = torch.load(p, weights_only=True)["state"]
            out[p.relative_to(run_dir)] = [state[k][f] for k in sorted(state) for f in ("exp_avg", "exp_avg_sq")]
    return out


class Killed(Exception):
    pass


def test_a_resumed_run_is_the_uninterrupted_run(finished, tmp_path, base_model, monkeypatch):
    """Killed during step 3 and started again, every member resumes from its step-2 checkpoint, practice record
    included, and ends bit-identical to the uninterrupted run: weights, optimizer, record, batches and samples."""
    monkeypatch.setattr(member_module, "score_many", engine_fakes.score_many)
    run_dir = engine_fakes.write_run(tmp_path, CONFIG)
    run = make_run(run_dir, base_model)
    real = Member.amplify

    def amplify(self, *args):
        if self.lin.id == "prac" and self.lin.step == 3:
            raise Killed
        return real(self, *args)
    monkeypatch.setattr(Member, "amplify", amplify)
    with pytest.raises(Killed):
        worker.advance(run, SLOTS)
    assert load_lineage(run, "prac").step == 2
    monkeypatch.setattr(Member, "amplify", real)
    worker.advance(make_run(run_dir, base_model, prepared=True), SLOTS)
    ours, theirs = member_state(run_dir), member_state(finished.dir)
    assert ours.keys() == theirs.keys() and any(p.name == "practice.json" for p in ours)
    for path in ours:
        if path.suffix == ".pt":
            assert all(torch.equal(x, y) for x, y in zip(ours[path], theirs[path])), path
        else:
            assert ours[path] == theirs[path], path


def test_a_missing_practice_record_is_refused(finished, tmp_path, base_model):
    import shutil

    d = tmp_path / "copy"
    shutil.copytree(finished.dir, d)
    (d / "members" / "prac" / "checkpoints" / "step_000004" / "practice.json").unlink()
    run = make_run(d, base_model, prepared=True)
    member = Member(run, load_lineage(run, "prac"))
    member.lin.status = "active"
    with pytest.raises(RuntimeError, match="practice.json is missing"):
        member.restore()


@pytest.fixture(scope="module")
def scored(finished):
    """The finished run with scaled probes of prac@4 at factors 2 and 0.5."""
    mp = pytest.MonkeyPatch()
    mp.setattr(evaluate, "Run", lambda d: finished)
    mp.setattr(evaluate, "Policy", lambda *a: finished.policy)
    mp.setattr(evaluate, "start_server", lambda *a: (None, "fake"))
    mp.setattr(evaluate, "Sampler", lambda *a: finished.sampler)
    mp.setattr(member_module, "score_many", engine_fakes.score_many)
    n = len(finished.sampler.requests)
    evaluate.scaled(finished.dir, ["prac@4"], [2.0, 0.5], None)
    yield finished, finished.sampler.requests[n:]
    mp.undo()


def test_scaled_probe(scored):
    run, requests = scored
    lin = load_lineage(run, "prac")
    for factor in (2.0, 0.5):
        d = run.dir / "scaled" / f"prac@4x{factor:g}"
        summary = read_json(d / "probe_summary.json")
        assert (summary["label"], summary["lineage"], summary["step"], summary["factor"]) == (
            f"prac@4x{factor:g}", "prac", 4, factor)
        records = read_jsonl_gz(d / "probe_responses.jsonl.gz")
        assert [r["uid"] for r in records] == [t.uid for t in run.probe_tasks] and all("token_ids" in r for r in records)
        source = run.policy.load_adapter(run.dir / "members" / "prac" / "checkpoints" / "step_000004")
        scaled = run.policy.load_adapter(d / "adapter")
        assert all(torch.equal(scaled[k], v * factor if ".lora_B." in k else v) for k, v in source.items())
    # the seeds of the lineage's own probe at that step: a paired comparison with the unscaled probe
    assert sorted(r.seed for r in requests) == sorted(2 * [stable_seed("probe", lin.seed, 4, i)
                                                           for i in range(len(run.probe_tasks))])
    assert len({r.model for r in requests}) == 2 and not (run.dir / "members" / "prac" / "scaled").exists()
    for specs, factors, match in ((["prac"], [2.0], "lineage>@<step"), (["prac@9"], [2.0], "committed"),
                                  (["prac@4"], [0.0], "positive")):
        with pytest.raises(SystemExit, match=match):
            evaluate.scaled(run.dir, specs, factors, None)


def table(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_deviation(scored, monkeypatch):
    run, _ = scored
    evaluate.deviation(run.dir)
    (selection,) = [p for p in (run.dir / "deviation").iterdir()]
    rows = table(selection / "responses.csv")
    assert list(rows[0]) == list(evaluate.DEVIATION_COLUMNS)
    subjects = {r["subject"] for r in rows}
    assert {"plain@0", "plain@4", "prac@2", "pc@2", "pc@4", "prac@4x2", "prac@4x0.5"} <= subjects
    assert len(rows) == len(subjects) * len(run.probe_tasks)
    assert all(float(r["delta_sum"]) == 0.0 for r in rows if r["step"] == "0")  # an unseeded member is the base model
    for subject, adapter in (("prac@4", run.dir / "members" / "prac" / "checkpoints" / "step_000004"),
                             ("prac@4x2", run.dir / "scaled" / "prac@4x2" / "adapter")):
        path = (run.dir / "members" / "prac" / "probe_responses" / "step_000004.jsonl.gz" if "x" not in subject
                else run.dir / "scaled" / "prac@4x2" / "probe_responses.jsonl.gz")
        records = read_jsonl_gz(path)
        seqs = [Sequence(run.prompt(r["question"]), r["token_ids"]) for r in records]
        mine = run.policy.logprobs(MemberWeights(run.policy.load_adapter(adapter)), seqs)
        expected = [float((a - b).sum()) for a, b in zip(mine, run.policy.base_logprobs(seqs))]
        got = [float(r["delta_sum"]) for r in rows if r["subject"] == subject]
        assert got == pytest.approx(expected, rel=1e-5, abs=1e-4) and any(abs(x) > 1e-3 for x in got)
        assert [int(r["tokens"]) for r in rows if r["subject"] == subject] == [len(r["token_ids"]) for r in records]
    assert {r["factor"] for r in rows if r["subject"] == "prac@4x2"} == {"2.0"}

    monkeypatch.setattr(evaluate, "Policy", lambda *a: pytest.fail("nothing should be recomputed"))
    evaluate.deviation(run.dir)  # every subject from its cache
    assert table(selection / "responses.csv") == rows
    monkeypatch.setattr(evaluate, "Policy", lambda *a: run.policy)
    evaluate.deviation(run.dir, items=3, names=["prac@4x2"])
    part = next(p for p in (run.dir / "deviation").iterdir() if p != selection)
    assert {r["subject"] for r in table(part / "responses.csv")} == {"prac@4x2"} and part.name.startswith("3items-")


def test_cross_likelihood(scored):
    run, _ = scored
    with pytest.raises(SystemExit, match="exactly one"):
        evaluate.deviation(run.dir, cross=True)
    evaluate.deviation(run.dir)  # the deviation of every probe (from the cache after test_deviation)
    evaluate.deviation(run.dir, steps=[4], names=["prac", "zero"], cross=True)
    selection = next(p for p in (run.dir / "deviation").iterdir() if p.name.startswith(f"{len(run.probe_tasks)}items"))
    rows = table(selection / "cross-4.csv")
    labels = ["base", "zero@4", "prac@4", "prac@4x2", "prac@4x0.5"]
    assert {(r["scorer"], r["author"]) for r in rows} == {(a, b) for a in labels for b in labels}
    assert len(rows) == len(labels) ** 2 * len(run.probe_tasks)
    own = {(r["scorer"], r["author"], r["uid"]): float(r["logprob_sum"]) for r in rows}
    deviation = {(r["subject"], r["uid"]): float(r["delta_sum"]) for r in table(selection / "responses.csv")}
    for author in ("prac@4", "prac@4x2"):  # the diagonal minus the base row is the deviation
        for t in run.probe_tasks:
            assert own[(author, author, t.uid)] - own[("base", author, t.uid)] == pytest.approx(
                deviation[(author, t.uid)], rel=1e-5, abs=1e-5)
    records = read_jsonl_gz(run.dir / "base" / "probe_responses.jsonl.gz")
    lps = run.policy.logprobs(MemberWeights(run.policy.load_adapter(
        run.dir / "members" / "zero" / "checkpoints" / "step_000004")),
        [Sequence(run.prompt(r["question"]), r["token_ids"]) for r in records])
    assert [own[("zero@4", "base", r["uid"])] for r in records] == pytest.approx([float(x.sum()) for x in lps],
                                                                                 rel=1e-5)
