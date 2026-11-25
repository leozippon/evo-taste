"""The five analysis commands on a run the engine itself wrote (tiny model, stand-in task layer and sampler of
tests/engine_fakes.py): seeding by opcd and none, a step-0 twin under self-imitation, clones, probes, the context score
and a selection evaluation with four samples. A change of the engine's record formats breaks these tests rather than
an analysis. The stand-in sampler ignores the adapter, so every member has the same solve probabilities: the
evaluation is an exact null for divergence and heritability."""

import json
import random

import numpy as np
import pandas as pd
import pytest
import torch
from threadpoolctl import threadpool_limits

from evotaste import evaluate, worker
from evotaste import member as member_module
from evotaste.analysis import data, divergence, health, heritability, seeding, washout
from evotaste.records import read_jsonl
from evotaste.sampler import Sample
from tests import engine_fakes
from tests.conftest import VOCAB, make_policy

ROOT = engine_fakes.ROOT
UNITS = [("gsm8k_test", "math", "open"), ("theoremqa", "math", "open"), ("mbpp_plus", "code", "code"),
         ("logiqa", "logic", "choice"), ("fld", "logic", "verdict"), ("gcd", "math", "procedural")]
MEMBERS = [
    {"id": "st", "seed": 1, "context": "refute"},
    {"id": "ta", "seed": 2, "context": "verify"},
    {"id": "pl", "seed": 3, "context": "placebo"},
    {"id": "un", "seed": 4, "overrides": {"seeding": {"method": "none"}}},
    {"id": "st-si", "seed": 5, "parent": {"member": "st", "step": 0}, "overrides": {"amplification": {"self_imitation": 0.5}}},
    {"id": "st-c1", "seed": 6, "parent": {"member": "st", "step": 1}},
    {"id": "st-c2", "seed": 7, "parent": {"member": "st", "step": 1}},
    {"id": "ta-c1", "seed": 8, "parent": {"member": "ta", "step": 1}},
]
CONFIG = {"name": "ana", "strata": {"path": "strata.tsv", "sha256": "0" * 64},
          "seeding": {"steps": 2, "batch": 2, "max_tokens": 8, "lr": 3e-3},
          "amplification": {"steps": 2, "prompts": 2, "rollouts": 2, "max_tokens": 8, "lr": 1e-3,
                            "length": {"free": 4, "slope": 0.25}},
          "probes": {"every": 1, "max_tokens": 8}, "culling": {"enabled": False, "grace": 1},
          "trainer": {"check_tokens": 4, "check_rollouts": 2}, "members": MEMBERS}
TAG = "n40x4"
EVAL_LABELS = ["base", "st@0", "st-si@0", "st@1", "st@2", "st-c1@2", "st-c2@2", "ta@2", "ta-c1@2", "pl@2", "un@2",
               "st-si@2"]


def task(family, domain, kind, j, wrapper=None):
    uid = f"{family}/7/{j}" if kind == "procedural" else f"{family}/{j}"
    return engine_fakes.Task(uid, family, domain, kind, f"{family} question {j}", "0",
                             wrapper or ("code" if kind == "code" else "box"), None)


def score_many(jobs, timeout=None):
    """engine_fakes.score_many with answer channels: an open answer whose second token is odd is a program."""
    out = []
    for (t, text, finish), s in zip(jobs, engine_fakes.score_many(jobs)):
        ids = text.split()
        channel = ("program" if len(ids) > 1 and int(ids[1]) % 2 else "direct") if t.kind == "open" else None
        out.append(engine_fakes.Score(s.reward, s.correct_any, s.complied, channel, s.truncated, s.extracted, None))
    return out


class EvalSampler:
    """Stand-in for the evaluator's sampler: sample k of a request from seed + k, whatever the adapter."""

    def __init__(self, url, model, name):
        pass

    def load_existing(self, label, path):
        return label

    def generate(self, requests):
        out = []
        for r in requests:
            group = []
            for k in range(r.n):
                rng = random.Random(r.seed + k)
                ids = [rng.randrange(VOCAB - 1) for _ in range(rng.randint(1, r.max_tokens - 1))] + [engine_fakes.EOT]
                group.append(Sample(" ".join(map(str, ids)), ids, "stop", [0.0] * len(ids)))
            out.append(group)
        return out

    def close(self):
        pass


@pytest.fixture(autouse=True)
def few_threads():
    with threadpool_limits(4):
        yield


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory, base_model):
    torch.set_num_threads(2)
    mp = pytest.MonkeyPatch()
    mp.chdir(ROOT)  # the analysis reads the suites file by its path relative to the repository root
    mp.setattr(member_module, "score_many", score_many)
    run = engine_fakes.FakeRun(engine_fakes.write_run(tmp_path_factory.mktemp("ana"), CONFIG))
    run.probe_tasks = [task(f, d, k, j, "answer_tag" if j == 2 and k != "code" else None)
                       for f, d, k in UNITS for j in range(3)]
    run.policy = make_policy(base_model)
    run.sampler = engine_fakes.FakeSampler(run.policy, tmp_path_factory.mktemp("adapters"))
    for context in [None, *run.contexts]:
        worker.base_probe(run, run.sampler, context)
    worker.advance(run, [m["id"] for m in MEMBERS])
    mp.setattr(evaluate, "Run", lambda d: run)
    mp.setattr(evaluate, "Policy", lambda *a: run.policy)
    evaluate.context_score(run.dir)
    mp.setattr(evaluate, "Sampler", EvalSampler)
    mp.setattr(evaluate, "check_gold", lambda tasks: None)
    mp.setattr(evaluate, "suite_tasks", lambda *a: [task(f, d, k, j) for f, d, k in UNITS for j in range(10, 50)])
    evaluate.evaluate(run.dir, "selection", EVAL_LABELS, 40, 4, "0/1", TAG, "http://stand-in")
    yield run.dir
    mp.undo()


def test_health_reproduces_the_engine_s_probe_summaries(run_dir, tmp_path):
    health.main([str(run_dir), "--tag", TAG, "--out", str(tmp_path)])
    probes = pd.read_csv(tmp_path / "probes.csv")
    acc = probes[probes["metric"] == "accuracy"].set_index("subject")["value"]
    run = data.Run(run_dir)
    for lin in run.lineages.values():
        for line in read_jsonl(lin.dir / "probes.jsonl"):
            assert acc[f"{lin.id}@{line['step']}"] == pytest.approx(line["accuracy"])
    assert acc["base"] == pytest.approx(json.loads((run_dir / "base" / "probe_summary.json").read_text())["accuracy"])
    assert {"far", "near", "procedural", "heldout_complied", "program_share"} <= set(probes["metric"])
    hist = pd.read_csv(tmp_path / "history.csv")
    assert set(hist.loc[hist["phase"] == "seed", "lineage"]) == {"st", "ta", "pl"}
    ev = pd.read_csv(tmp_path / "eval.csv")
    assert set(ev["subject"]) == set(EVAL_LABELS) - {"st-si@0"}  # the twin at step 0 is st@0's adapter
    assert (tmp_path / "report.txt").read_text().startswith("health ana")


def test_washout_and_seeding(run_dir, tmp_path):
    washout.main([str(run_dir), "--out", str(tmp_path / "w")])
    idx = pd.read_csv(tmp_path / "w" / "index.csv")
    undefined = idx["S_teacher"] <= idx["S_base"]  # the instructed answers are not better predicted: no index
    assert idx.loc[undefined, "I"].isna().all() and idx.loc[~undefined, "I"].notna().all()
    assert np.allclose(idx.loc[(idx["subject"] == "base") & ~undefined, "I"], 0)
    teacher = idx[(idx["subject"] == "base:refute") & (idx["context"] == "refute") & ~undefined]
    assert len(teacher) and np.allclose(teacher["I"], 1)
    spec = pd.read_csv(tmp_path / "w" / "specific.csv")
    assert {"st", "ta", "st-si", "st-c1", "st-c2", "ta-c1"} == set(spec["lineage"])  # seeded, not the controls
    assert spec.loc[(spec["lineage"] == "st") & (spec["step"] == 0), "controls"].max() == 2  # pl@0 and un@0
    assert len(pd.read_csv(tmp_path / "w" / "noise.csv")) > 0  # from the twin st-si
    seeding.main([str(run_dir), "--out", str(tmp_path / "s")])
    members = pd.read_csv(tmp_path / "s" / "members.csv")
    assert set(members["lineage"]) == {"st", "ta", "pl", "un"}
    kl = pd.read_csv(tmp_path / "s" / "kl.csv")
    assert set(kl["lineage"]) == {"st", "ta", "pl"} and kl["kl"].notna().all()


def test_divergence_finds_nothing_in_an_exact_null(run_dir, tmp_path):
    divergence.main([str(run_dir), "--tag", TAG, "--classifier-items", "40", "--out", str(tmp_path)])
    inter = pd.read_csv(tmp_path / "interaction.csv")
    row = inter[(inter["members"] == "all") & (inter["scope"] == "all") & (inter["estimator"] == "adjusted")].iloc[0]
    assert row["m"] == 8  # the members at step 2
    assert row["var_lo"] < 0 < row["var_hi"]
    null = pd.read_csv(tmp_path / "null.csv")
    assert set(null["adapter"]) == set(EVAL_LABELS) - {"st-si@0"} | {"mean"}
    mean = null[null["adapter"] == "mean"]
    assert ((mean["lo"] < 0) & (mean["hi"] > 0)).all()
    au = pd.read_csv(tmp_path / "authorship.csv")
    assert set(au["view"]) == {"raw", "stripped", "layout", "cosmetic_share"}
    assert "st-si@0 = st@0" in (tmp_path / "report.txt").read_text()


def test_heritability_families(run_dir, tmp_path):
    heritability.main([str(run_dir), "--tag", TAG, "--out", str(tmp_path)])
    icc = pd.read_csv(tmp_path / "icc.csv")
    acc = icc[(icc["step"] == 2) & (icc["trait"] == "accuracy") & (icc["rule"] == "centered")].iloc[0]
    assert (acc["families"], acc["members"], acc["relabellings"]) == (2, 5, 10)  # {st, st-c1, st-c2}, {ta, ta-c1}
    dist = pd.read_csv(tmp_path / "distance.csv")
    d = dist[(dist["step"] == 2) & (dist["estimator"] == "adjusted") & (dist["rule"] == "centered")].iloc[0]
    assert d["diff_lo"] < 0 < d["diff_hi"] or d["p_perm"] > 0.05
    pairs = pd.read_csv(tmp_path / "pairs.csv")
    clones = pairs[(pairs["step"] == 2) & pairs["clones"] & (pairs["estimator"] == "adjusted")]
    assert {frozenset(p) for p in zip(clones["a"], clones["b"])} == {
        frozenset(p) for p in [("st@2", "st-c1@2"), ("st@2", "st-c2@2"), ("st-c1@2", "st-c2@2"), ("ta@2", "ta-c1@2")]}


def test_a_changed_record_format_is_refused(run_dir, tmp_path):
    run = data.Run(run_dir)
    lin = run.lineages["un"]
    records = run.probe_records(lin, 2)
    summary = run.probe_summaries(lin)[2]
    bad = {**summary["by_family"], "gcd": summary["by_family"]["gcd"] + 0.5}
    with pytest.raises(ValueError, match="per-family accuracy"):
        data.check_families(records, bad, "probe")
    with pytest.raises(KeyError):
        data.build_grid({"x": [{k: v for k, v in r.items() if k != "channel"} for r in records]}, 1, run.units)
