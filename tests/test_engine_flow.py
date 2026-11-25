"""The whole member life cycle on the CPU (tiny model, stand-in task layer and sampler): base references, seeding by
opcd, imitation and none in lockstep, clones that wait for and start from their parent, amplification with the length
slope and self-imitation, disabled culling, and the context score on the stored probe answers."""

import copy
import csv
import gzip
import json

import pytest

from evotaste import evaluate, worker
from evotaste import member as member_module
from evotaste.member import load_lineage
from evotaste.records import read_json, read_jsonl, read_jsonl_gz
from evotaste.tasks import stable_seed
from evotaste.trainer import Sequence
from tests import engine_fakes
from tests.conftest import make_policy

MEMBERS = [
    {"id": "a", "seed": 1, "context": "refute"},
    {"id": "b", "seed": 2, "context": "verify", "overrides": {"seeding": {"method": "imitation", "samples": 3}}},
    {"id": "c", "seed": 3, "overrides": {"seeding": {"method": "none"}, "amplification": {"steps": 2}}},
    {"id": "a1", "seed": 11, "parent": {"member": "a", "step": 1}},
    {"id": "a2", "seed": 12, "parent": {"member": "a", "step": 1}},
]
CONFIG = {"name": "flow", "strata": {"path": "strata.tsv", "sha256": "0" * 64},
          "seeding": {"steps": 3, "batch": 4, "max_tokens": 12, "lr": 3e-3},
          "amplification": {"steps": 4, "prompts": 3, "rollouts": 3, "max_tokens": 12, "lr": 1e-3,
                            "length": {"free": 6, "slope": 0.25}, "self_imitation": 0.5},
          "probes": {"every": 2, "max_tokens": 12}, "culling": {"enabled": False, "grace": 1},
          "trainer": {"check_tokens": 20, "check_rollouts": 4}, "members": MEMBERS}


def make_run(tmp_path, base_model, monkeypatch, members=MEMBERS):
    monkeypatch.setattr(member_module, "score_many", engine_fakes.score_many)
    run = engine_fakes.FakeRun(engine_fakes.write_run(tmp_path, {**CONFIG, "members": members}))
    run.policy = make_policy(base_model)
    run.sampler = engine_fakes.FakeSampler(run.policy, tmp_path)
    for context in [None, *run.contexts]:  # what prepare generates
        worker.base_probe(run, run.sampler, context)
    return run


@pytest.fixture(scope="module")
def finished(tmp_path_factory, base_model):
    mp = pytest.MonkeyPatch()
    run = make_run(tmp_path_factory.mktemp("flow"), base_model, mp)
    worker.advance(run, [m["id"] for m in MEMBERS])
    yield run
    mp.undo()


def history(run, lineage):
    return read_jsonl(run.dir / "members" / lineage / "history.jsonl")


def test_every_member_runs_its_course(finished):
    run = finished
    for lid, steps, probes in (("a", 4, [0, 2, 4]), ("b", 4, [0, 2, 4]), ("c", 2, [0, 2]),
                               ("a1", 4, [1, 2, 4]), ("a2", 4, [1, 2, 4])):
        lin = load_lineage(run, lid)
        assert (lin.status, lin.step, [p["step"] for p in lin.probes]) == ("done", steps, probes), lid
        assert all(p["culled"] is False for p in lin.probes)
    assert not list((run.dir / "members").glob("*~*"))  # culling disabled: nobody replaced
    assert sorted(p.name for p in (run.dir / "base" / "contexts").glob("*.json")) == sorted(
        f"{c}.json" for c in run.contexts)


def test_seeding_methods(finished):
    seeds = {lid: [r for r in history(finished, lid) if r["phase"] == "seed"] for lid in ("a", "b", "c")}
    assert [r["method"] for r in seeds["a"]] == ["opcd"] * 3 and all(r["kl"] > 0 and r["check"] for r in seeds["a"])
    assert seeds["a"][-1]["kl"] < seeds["a"][0]["kl"]  # the member moves toward the teacher
    assert [r["method"] for r in seeds["b"]] == ["imitation"] * 3 and all(r["correct"] > 0 for r in seeds["b"])
    assert seeds["c"] == []
    reqs = finished.sampler.requests
    opcd = [r for r in reqs if r.seed == stable_seed("opcd", 1, 0, 0)]
    assert len(opcd) == 1 and opcd[0].model.startswith("a@") and opcd[0].n == 1  # the member's own sample, no context
    imitation = [r for r in reqs if r.seed == stable_seed("imitation", 2, 0, 0)]
    assert len(imitation) == 1 and imitation[0].model == engine_fakes.MODEL and imitation[0].n == 3
    question = finished.streams.seed_batch(0, 4)[0].question
    assert opcd[0].prompt == finished.prompt(question) and imitation[0].prompt == finished.prompt(question, "verify")


def test_clones_start_from_the_parents_state_and_differ_only_in_sampling(finished):
    run = finished
    parent_ckpt = run.dir / "members" / "a" / "checkpoints" / "step_000001"
    assert (parent_ckpt / "optimizer.pt").exists()  # a clone point keeps its optimizer state
    assert sorted(p.name for p in parent_ckpt.parent.iterdir()) == [f"step_{s:06d}" for s in (0, 1, 2, 4)]
    for lid in ("a1", "a2"):  # (the optimizer state's exact copy is tested in test_resume)
        d = run.dir / "members" / lid
        assert read_json(d / "state.json")["parent"] == {"lineage": "a", "step": 1}
        ckpt = d / "checkpoints" / "step_000001"  # committed with the clone's first probe
        weights = "adapter_model.safetensors"
        assert (ckpt / weights).read_bytes() == (parent_ckpt / weights).read_bytes()
        assert [r["step"] for r in history(run, lid)] == [1, 2, 3]
    reqs = {r.seed: r for r in run.sampler.requests}
    for j in range(3):
        a1, a2 = reqs[stable_seed("rollout", 11, 1, j)], reqs[stable_seed("rollout", 12, 1, j)]
        assert a1.prompt == a2.prompt and a1.seed != a2.seed  # the same prompts, different sampling seeds
    rollouts = [read_jsonl_gz(run.dir / "members" / lid / "rollouts" / "step_000001.jsonl.gz") for lid in ("a1", "a2")]
    assert [r["uid"] for r in rollouts[0]] == [r["uid"] for r in rollouts[1]]
    assert [r["text"] for r in rollouts[0]] != [r["text"] for r in rollouts[1]]


def test_rollout_records_and_updates(finished):
    for lid in ("a", "b", "a1"):
        for line in history(finished, lid):
            if line["phase"] != "amp":
                continue
            records = read_jsonl_gz(finished.dir / "members" / lid / "rollouts" / f"step_{line['step']:06d}.jsonl.gz")
            assert {"reward", "correct_any", "complied", "channel", "truncated", "extracted", "error", "return",
                    "weight", "domain", "kind", "wrapper", "stratum"} <= set(records[0])
            for j in range(3):
                group = [r for r in records if r["prompt"] == j]
                tokens = [r["completion_tokens"] for r in group]
                expected = [r["reward"] - 0.25 * min(max((n - 6) / 6, 0), 1) for r, n in zip(group, tokens)]
                assert [r["return"] for r in group] == pytest.approx(expected)
                if len({r["return"] for r in group}) == 1:
                    assert all(r["weight"] == 0.0 for r in group)
            assert line["si_sequences"] == sum(r["reward"] == 1.0 for r in records)
            assert line["return"] == pytest.approx(sum(r["return"] for r in records) / len(records))
    assert sum(line.get("si_sequences", 0) for line in history(finished, "a")) > 0


def test_probe_records_keep_the_sampled_ids(finished):
    for path in (finished.dir / "members" / "a" / "probe_responses").glob("*.jsonl.gz"):
        for r in read_jsonl_gz(path):
            assert r["completion_tokens"] == len(r["token_ids"]) and r["text"] == " ".join(map(str, r["token_ids"]))


def test_a_clone_waits_for_a_parent_advanced_elsewhere(tmp_path, base_model, monkeypatch, caplog):
    run = make_run(tmp_path, base_model, monkeypatch)

    class Waited(Exception):
        pass

    def sleep(_):
        raise Waited
    monkeypatch.setattr(worker.time, "sleep", sleep)
    with caplog.at_level("INFO"), pytest.raises(Waited):
        worker.advance(run, ["a1"])
    assert "waiting for parents: a1 <- a@1" in caplog.text
    assert load_lineage(run, "a1").status == "new" and not (run.dir / "members" / "a1" / "probes.jsonl").exists()


def test_context_score(finished, monkeypatch):
    run = finished
    monkeypatch.setattr(evaluate, "Run", lambda d: run)
    monkeypatch.setattr(evaluate, "Policy", lambda *a: run.policy)
    evaluate.context_score(run.dir)
    with open(run.dir / "context_score" / "table.csv") as f:
        rows = list(csv.DictReader(f))
    subjects = {r["subject"] for r in rows}
    assert {"base", "base:refute", "a@0", "a@4", "a1@1", "c@2"} <= subjects and len(subjects) == 13 + 3 + 3 + 2 + 3 + 3
    assert {r["context"] for r in rows} == set(run.contexts)

    # one subject recomputed by hand: token-mean of log p_base(token | context prompt) - log p_base(token | plain)
    records = read_jsonl_gz(run.dir / "members" / "a" / "probe_responses" / "step_000002.jsonl.gz")
    for domain in ("math", "code"):
        mine = [r for r in records if r["domain"] == domain]
        ctx = run.policy.base_logprobs([Sequence(run.prompt(r["question"], "map"), r["token_ids"]) for r in mine])
        plain = run.policy.base_logprobs([Sequence(run.prompt(r["question"]), r["token_ids"]) for r in mine])
        expected = sum(float((x - y).sum()) for x, y in zip(ctx, plain)) / sum(len(x) for x in ctx)
        row = next(r for r in rows if (r["subject"], r["context"], r["domain"]) == ("a@2", "map", domain))
        assert float(row["delta_mean"]) == pytest.approx(expected, rel=1e-5)
        assert int(row["tokens"]) == sum(len(r["token_ids"]) for r in mine)

    # a rerun takes every subject from its cache; a rewritten probe file is scored again
    monkeypatch.setattr(evaluate, "Policy", lambda *a: pytest.fail("nothing should be recomputed"))
    evaluate.context_score(run.dir)
    path = run.dir / "members" / "c" / "probe_responses" / "step_000002.jsonl.gz"
    records = read_jsonl_gz(path)
    path.write_bytes(gzip.compress("".join(json.dumps(r) + "\n" for r in records[:3]).encode()))
    monkeypatch.setattr(evaluate, "Policy", lambda *a: run.policy)
    evaluate.context_score(run.dir)
    assert read_json(run.dir / "context_score" / "c@2.json")["rows"][0]["responses"] < len(records)


def test_context_score_of_a_plain_prompt_context_is_zero(finished, monkeypatch):
    run = copy.copy(finished)
    run.contexts = {"same": None}
    run.prompt = lambda question, context=None: finished.prompt(question)
    records = read_jsonl_gz(run.dir / "base" / "probe_responses.jsonl.gz")
    deltas = evaluate.context_deltas(run, run.policy, records)["same"]
    assert all(x == 0.0 for _, x, _ in deltas) and sum(n for _, _, n in deltas) == sum(
        r["completion_tokens"] for r in records)
