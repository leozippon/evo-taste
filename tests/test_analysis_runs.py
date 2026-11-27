"""The analysis commands on a run the engine itself wrote (tiny model, stand-in task layer and sampler of
tests/engine_fakes.py): seeding by opcd and none, a step-0 twin under self-imitation, clones, probes, the context score,
a scaled probe, the deviation and cross-likelihood tables, and a selection evaluation with four samples. A change of
the engine's record formats breaks these tests rather than an analysis. The stand-in evaluation sampler ignores the
adapter, so every member has the same solve probabilities: the evaluation is an exact null for divergence,
heritability and the length decomposition."""

import random

import numpy as np
import pandas as pd
import pytest
import torch
from threadpoolctl import threadpool_limits

from evotaste import evaluate, worker
from evotaste import member as member_module
from evotaste.analysis import (arms, data, deviation, distinct, divergence, harvest, health, heritability, length,
                               scaled, seeding, washout)
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
    {"id": "st-si-c1", "seed": 9, "parent": {"member": "st-si", "step": 1}},  # a clone of a clone
]
CONFIG = {"name": "ana", "strata": {"path": "strata.tsv", "sha256": "0" * 64},
          "seeding": {"steps": 2, "batch": 2, "max_tokens": 8, "lr": 3e-3},
          "amplification": {"steps": 2, "prompts": 2, "rollouts": 2, "max_tokens": 8, "lr": 1e-3,
                            "length": {"free": 4, "slope": 0.25}},
          "probes": {"every": 1, "max_tokens": 8}, "culling": {"enabled": False, "grace": 1},
          "trainer": {"check_tokens": 4, "check_rollouts": 2}, "members": MEMBERS}
TAG = "n40x4"
EVAL_LABELS = ["base", "st@0", "st-si@0", "st@1", "st@2", "st-c1@2", "st-c2@2", "ta@2", "ta-c1@2", "pl@2", "un@2",
               "st-si@2", "st-si-c1@2"]


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
    mp.setattr(evaluate, "start_server", lambda *a: (None, "stand-in"))
    mp.setattr(evaluate, "Sampler", lambda *a: run.sampler)
    evaluate.scaled(run.dir, ["st@2"], [2.0], None)
    evaluate.deviation(run.dir)
    evaluate.deviation(run.dir, steps=[2], cross=True)
    mp.setattr(evaluate, "Sampler", EvalSampler)
    mp.setattr(evaluate, "check_gold", lambda tasks: None)
    mp.setattr(evaluate, "suite_tasks", lambda *a: [task(f, d, k, j) for f, d, k in UNITS for j in range(10, 50)])
    evaluate.evaluate(run.dir, "selection", EVAL_LABELS, 40, 4, "0/1", TAG, "http://stand-in")
    # the shape of pilot C's s45 and s90: the parents at the clone point on every item, the clones later on the dataset
    # items and one item per procedural family, without the base model
    evaluate.evaluate(run.dir, "selection", ["st@1", "ta@1", "st-si@1"], 40, 4, "0/1", "par", "http://stand-in")
    mp.setattr(evaluate, "suite_tasks", lambda *a: [task(f, d, k, j) for f, d, k in UNITS for j in range(10, 50)
                                                    if k != "procedural" or j == 10])
    evaluate.evaluate(run.dir, "selection", ["st-c1@2", "st-c2@2", "ta-c1@2", "st-si-c1@2"], 1, 4, "0/1", "sub",
                      "http://stand-in", per_source=40)
    yield run.dir
    mp.undo()


def test_health_reads_every_probe_against_the_pooled_base_weights(run_dir, tmp_path):
    health.main([str(run_dir), "--tag", TAG, "--out", str(tmp_path)])
    probes = pd.read_csv(tmp_path / "probes.csv")
    acc = probes[probes["metric"] == "accuracy"].set_index("subject")
    run = data.Run(run_dir)
    for lin in run.lineages.values():  # every committed probe, accuracy the mean over its items
        for line in read_jsonl(lin.dir / "probes.jsonl"):
            records = run.probe_records(lin, line["step"])
            assert acc.loc[f"{lin.id}@{line['step']}", "value"] == pytest.approx(np.mean([r["reward"] for r in records]))
    ref = acc.loc[health.REFERENCE]  # the base model's prepare-time answers and un@0, an unseeded member at step 0
    assert ref["draws"] == 2 and ref["value"] == pytest.approx(acc.loc[["base", "un@0"], "value"].mean())
    assert np.isnan(acc.loc["un@0", "diff"]) and acc.loc["st@0", "diff"] == pytest.approx(
        acc.loc["st@0", "value"] - ref["value"])
    assert {"far", "near", "procedural", "heldout_complied", "program_share"} <= set(probes["metric"])
    train = pd.read_csv(tmp_path / "training_length.csv")
    assert set(train["rule"]) == {"centered", "centered+si0.5"} and (train["deduction"] >= 0).all()
    plen = pd.read_csv(tmp_path / "probe_length.csv")
    assert (plen["lost"] <= plen["truncated"] + 1e-12).all() and set(plen["step"]) == {0, 1, 2}
    hist = pd.read_csv(tmp_path / "history.csv")
    assert set(hist.loc[hist["phase"] == "seed", "lineage"]) == {"st", "ta", "pl"}
    ev = pd.read_csv(tmp_path / "eval.csv")
    assert set(ev["subject"]) == set(EVAL_LABELS) - {"st-si@0"} | {health.REFERENCE}  # st-si@0 is st@0's adapter
    assert (tmp_path / "report.txt").read_text().startswith("health ana")


def test_washout_and_seeding(run_dir, tmp_path):
    washout.main([str(run_dir), "--out", str(tmp_path / "w")])
    idx = pd.read_csv(tmp_path / "w" / "index.csv")
    undefined = idx["effect"] <= 0  # the instructed answers are not better predicted: no index
    assert idx.loc[undefined, "I"].isna().all() and idx.loc[~undefined, "I"].notna().all()
    assert np.allclose(idx.loc[(idx["subject"] == "base") & ~undefined, "I"], 0)
    teacher = idx[(idx["subject"] == "base:refute") & (idx["context"] == "refute") & ~undefined]
    assert len(teacher) and np.allclose(teacher["I"], 1)
    spec = pd.read_csv(tmp_path / "w" / "specific.csv")
    assert {"st", "ta", "st-si", "st-c1", "st-c2", "ta-c1", "st-si-c1"} == set(spec["lineage"])  # not the controls
    assert not ((spec["lineage"] == "st-c1") & (spec["step"] == 1)).any()  # a clone's first probe is its parent's
    assert spec.loc[(spec["lineage"] == "st") & (spec["step"] == 0), "controls"].max() == 2  # pl@0 and un@0
    noise = pd.read_csv(tmp_path / "w" / "noise.csv")
    assert set(noise["step"]) == {0, 1}  # st@0 with its twin st-si@0; the clone points at step 1
    rules = pd.read_csv(tmp_path / "w" / "rules.csv")  # st (centered) against its step-0 twin st-si
    assert set(rules.loc[rules["what"] == "root", "root"]) == {"st"} and set(rules["step"]) == {1, 2}
    ok = spec.dropna(subset=["specific"])
    assert np.allclose(ok["gain"], ok["specific"] * ok["effect"])  # specific = gain / effect
    assert ((ok["specific_lo"] <= ok["specific"]) & (ok["specific"] <= ok["specific_hi"])).all()
    seeding.main([str(run_dir), "--out", str(tmp_path / "s")])
    members = pd.read_csv(tmp_path / "s" / "members.csv")
    assert set(members["lineage"]) == {"st", "ta", "pl"}  # seeded, not clones; the unseeded member is the reference
    kl = pd.read_csv(tmp_path / "s" / "kl.csv")
    assert set(kl["lineage"]) == {"st", "ta", "pl"} and kl["kl"].notna().all()
    acc = pd.read_csv(tmp_path / "s" / "accuracy.csv")
    draws = acc[acc["metric"] == "accuracy"].set_index("subject")["draws"]
    assert draws[health.REFERENCE] == 2 and draws["st"] == 2 and draws["ta"] == 1  # st@0 and its twin st-si@0


def test_divergence_finds_nothing_in_an_exact_null(run_dir, tmp_path):
    divergence.main([str(run_dir), "--tag", TAG, "--classifier-items", "40", "--out", str(tmp_path)])
    inter = pd.read_csv(tmp_path / "interaction.csv")
    row = inter[(inter["members"] == "all") & (inter["scope"] == "all") & (inter["estimator"] == "adjusted")].iloc[0]
    assert row["m"] == 9  # the members at step 2
    assert row["var_lo"] < 0 < row["var_hi"]
    null = pd.read_csv(tmp_path / "null.csv")
    assert set(null["adapter"]) == set(EVAL_LABELS) - {"st-si@0"} | {"mean"}
    mean = null[null["adapter"] == "mean"]
    assert ((mean["lo"] < 0) & (mean["hi"] > 0)).all()
    au = pd.read_csv(tmp_path / "authorship.csv")
    assert set(au["view"]) == {"raw", "stripped", "layout", "cosmetic_share"}
    assert row["items"] <= 200  # the headline reads the dataset units only, not the procedural unit gcd
    fmt = pd.read_csv(tmp_path / "format.csv")
    assert set(fmt["part"]) == {"dataset", "procedural"} and np.allclose(fmt["gap"], fmt["correct_any"] - fmt["reward"])
    length = pd.read_csv(tmp_path / "length.csv")
    assert set(length["domain"]) == {"all", "math", "code", "logic"} and (length["tokens_p90"] >= length["tokens"] - 8).all()
    assert "st-si@0 = st@0" in (tmp_path / "report.txt").read_text()


def test_heritability_families(run_dir, tmp_path):
    heritability.main([str(run_dir), "--tag", TAG, "--out", str(tmp_path)])
    icc = pd.read_csv(tmp_path / "icc.csv")
    acc = icc[(icc["step"] == 2) & (icc["trait"] == "accuracy") & (icc["rule"] == "centered")].iloc[0]
    assert (acc["families"], acc["members"], acc["relabellings"]) == (2, 5, 10)  # {st, st-c1, st-c2}, {ta, ta-c1}
    fams = pd.read_csv(tmp_path / "families.csv")
    assert set(fams.loc[fams["step"] == 2, "family"]) == {"st/centered", "ta/centered", "pl/centered", "un/centered",
                                                          "st/centered+si0.5"}
    chain = fams.loc[(fams["step"] == 2) & (fams["family"] == "st/centered+si0.5"), "members"].item()
    assert set(chain.split()) == {"st-si@2", "st-si-c1@2"}  # the chain st -> st-si -> st-si-c1 has the root st
    dist = pd.read_csv(tmp_path / "distance.csv")
    d = dist[(dist["step"] == 2) & (dist["estimator"] == "adjusted") & (dist["rule"] == "centered")].iloc[0]
    assert d["diff_lo"] < 0 < d["diff_hi"] or d["p_perm"] > 0.05
    pairs = pd.read_csv(tmp_path / "pairs.csv")
    clones = pairs[(pairs["step"] == 2) & pairs["clones"] & (pairs["estimator"] == "adjusted")]
    assert {frozenset(p) for p in zip(clones["a"], clones["b"])} == {
        frozenset(p) for p in [("st@2", "st-c1@2"), ("st@2", "st-c2@2"), ("st-c1@2", "st-c2@2"), ("ta@2", "ta-c1@2"),
                               ("st-si@2", "st-si-c1@2")]}


def test_tags_of_the_same_items_merge(run_dir):
    run = data.Run(run_dir)
    one, two = data.load_eval(run, TAG), data.load_eval(run, f"{TAG},{TAG}")
    assert one.labels == two.labels and np.array_equal(one.x["reward"], two.x["reward"])
    with pytest.raises(SystemExit, match="no evaluation"):
        data.load_eval(run, f"{TAG},missing")


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


def test_deviation_reads_the_engine_table_and_its_partial_caches(run_dir, tmp_path):
    deviation.main([str(run_dir), "--boot", "50", "--out", str(tmp_path)])
    au = pd.read_csv(tmp_path / "authors.csv")
    run = data.Run(run_dir)
    probes = {f"{lin.id}@{s}" for lin in run.lineages.values() for s in run.probe_steps(lin)} | {"st@2x2"}
    assert set(au["author"]) == probes
    a = au[au["scope"] == "all"].set_index("author")
    assert a.loc["un@0", "deviation"] == 0 and a.loc["st-si@0", "deviation"] != 0  # base weights; a seeded member
    assert ((a["lo"] <= a["deviation"] + 1e-12) & (a["deviation"] <= a["hi"] + 1e-12)).all()
    tr = pd.read_csv(tmp_path / "trajectory.csv")
    assert set(tr["rule"]) == {"centered", "centered+si0.5"} and not ((tr["rule"] == "centered+si0.5") & (tr["step"] == 0)).any()
    assert set(pd.read_csv(tmp_path / "length.csv")["author"]) == probes
    sel = data.deviation_selection(run)
    full, partial = data.deviation_table(run, sel)[0], data.deviation_table(run, sel, partial=True)[0]
    key = ["subject", "uid"]
    assert full.sort_values(key).reset_index(drop=True)[["tokens", "delta_sum"]].equals(
        partial.sort_values(key).reset_index(drop=True)[["tokens", "delta_sum"]])


def test_distinct_on_the_engine_cross_table(run_dir, tmp_path):
    distinct.main([str(run_dir), "--step", "2", "--boot", "50", "--perms", "200", "--classifier-items", "6",
                   "--out", str(tmp_path)])
    per = pd.read_csv(tmp_path / "members.csv").set_index("member")
    dev = pd.read_csv(tmp_path / "margins.csv")
    run = data.Run(run_dir)
    table = data.deviation_table(run)[0]
    for m in ("st@2", "st@2x2", "ta-c1@2"):  # the diagonal minus the base row is the deviation
        rows = table[table["subject"] == m]
        assert per.loc[m, "deviation"] == pytest.approx(rows["delta_sum"].sum() / rows["tokens"].sum(), abs=1e-6)
    own = dev[(dev["scope"] == "all") & (dev["scorer"] == "base")]
    assert set(own["author"]) == set(per.index) and len(per) == 10  # nine members at step 2 and st@2x2
    ident = pd.read_csv(tmp_path / "identification.csv")
    assert set(ident["scope"]) == {"all", "math", "code", "logic"} and (ident["chance"] == 0.1).all()
    clones = pd.read_csv(tmp_path / "clones.csv")  # the self-imitation rule has one family: no unrelated pair
    assert set(clones["rule"]) == {"all", "centered"}
    assert clones.loc[clones["rule"] == "centered", "families"].item() == 4  # st (3), ta (2), pl, un
    assert {"raw", "stripped", "cosmetic_share"} <= set(pd.read_csv(tmp_path / "authorship.csv")["view"])


def test_length_decomposition_on_the_exact_null(run_dir, tmp_path):
    length.main([str(run_dir), "--tag", TAG, "--boot", "50", "--perms", "50", "--classifier-items", "40",
                 "--out", str(tmp_path)])
    inter = pd.read_csv(tmp_path / "interaction.csv")
    models = list(inter.loc[(inter["items_set"] == "all") & (inter["estimator"] == "adjusted"), "model"])
    assert models[:3] == ["none", "rule", "rule+length"]
    none = inter[(inter["items_set"] == "all") & (inter["estimator"] == "adjusted") & (inter["model"] == "none")].iloc[0]
    assert none["var_lo"] < 0 < none["var_hi"]  # every member has the same solve probabilities
    mem = pd.read_csv(tmp_path / "members.csv")
    assert set(mem["scope"]) == {"all", "math", "code", "logic"} and len(mem) == 4 * 9
    assert set(pd.read_csv(tmp_path / "association.csv")["statistic"]) >= {"r_length", "r2_gain"}
    assert {"raw", "prefix", "stripped_prefix", "length", "length_cue_share"} <= set(
        pd.read_csv(tmp_path / "authorship.csv")["view"])
    assert "reading:" in (tmp_path / "report.txt").read_text()


def test_scaled_probe_against_its_factor(run_dir, tmp_path):
    scaled.main([str(run_dir), "--boot", "50", "--out", str(tmp_path)])
    t = pd.read_csv(tmp_path / "scaled.csv")
    assert list(t["factor"]) == [0.0, 1.0, 2.0] and set(t["adapter"]) == {"st@2"}
    assert t.loc[t["factor"] == 1, "accuracy_diff"].item() == 0 and t.loc[t["factor"] == 0, "deviation"].item() == 0
    assert t["deviation"].notna().all()


def test_arms_by_update_rule(run_dir, tmp_path):
    arms.main([str(run_dir), "--boot", "50", "--classifier-items", "6", "--out", str(tmp_path)])
    hist = pd.read_csv(tmp_path / "history.csv")
    assert set(hist["arm"]) == {"centered", "centered+si0.5"} and {"reward", "si_sequences"} <= set(hist["field"])
    fields = pd.read_csv(tmp_path / "fields.csv")
    assert not fields.loc[fields["field"].str.startswith("taste"), "present"].any()  # reported as absent
    assert "absent from the history" not in (tmp_path / "report.txt").read_text()  # not a taste run: not mentioned
    probes = pd.read_csv(tmp_path / "probes.csv")
    assert set(probes["step"]) == {0, 1, 2} and set(probes["metric"]) >= {"accuracy", "far", "heldout_complied"}
    assert len(pd.read_csv(tmp_path / "deviation.csv"))
    au = pd.read_csv(tmp_path / "authorship.csv")
    assert {"raw", "stripped"} <= set(au["view"]) and "all" in set(au["arm"])


def test_heritability_on_tags_of_different_items(run_dir, tmp_path):
    """Parents at the clone point in one tag, their clones later in another with one item per procedural family and
    no base model (pilot C's s45 and s90): merged on the common items."""
    heritability.main([str(run_dir), "--tag", "par,sub", "--out", str(tmp_path)])
    text = (tmp_path / "report.txt").read_text()
    assert "tags merged on the 201 items" in text  # the 200 dataset items and the one gcd item both tags hold
    reg = pd.read_csv(tmp_path / "regression.csv")
    assert set(reg["step"]) == {2} and reg.loc[reg["trait"] == "accuracy", "families"].item() == 3
    assert {"profile_math", "profile_logic"} <= set(reg["trait"])
    sep = pd.read_csv(tmp_path / "separation.csv")
    clone = sep[(sep["pairs"] == "clone") & (sep["step"] == 2)].iloc[0]
    assert clone["apart"] == 1 and clone["n"] == 1  # st-c1@2 and st-c2@2, apart since step 1
    assert set(sep["pairs"]) <= {"clone", "unrelated", "common start"}


def test_layout_shares_of_deviation_and_margins(run_dir, tmp_path):
    deviation.main([str(run_dir), "--boot", "30", "--out", str(tmp_path / "d")])
    au = pd.read_csv(tmp_path / "d" / "authors.csv")
    a = au[(au["scope"] == "all") & (au["deviation"] != 0)]
    assert a["layout_tokens"].between(0, 1).all() and a["layout_share"].notna().all()
    distinct.main([str(run_dir), "--step", "2", "--boot", "30", "--perms", "50", "--classifier-items", "0",
                   "--out", str(tmp_path / "x")])
    per = pd.read_csv(tmp_path / "x" / "members.csv").set_index("member")
    dev = a.set_index("author")
    for m in ("st@2", "ta-c1@2"):  # the layout part of the diagonal minus the base row is the deviation's
        assert per.loc[m, "deviation_in_layout"] == pytest.approx(dev.loc[m, "layout_share"], abs=1e-3)
        assert per.loc[m, "layout_tokens"] == pytest.approx(dev.loc[m, "layout_tokens"])


def test_arms_paired_by_parent(run_dir, tmp_path):
    arms.main([str(run_dir), "--parents", "st", "--arms", "c1,c2", "--reference", "c2", "--boot", "30",
               "--classifier-items", "6", "--out", str(tmp_path)])
    pairs = pd.read_csv(tmp_path / "pairs.csv")
    assert set(pairs["step"]) == {1, 2} and set(pairs["arm"]) == {"c1"} and (pairs["parents"] == 1).all()
    assert {"deviation", "deviation_in_layout", "accuracy"} <= set(pairs["metric"])
    hp = pd.read_csv(tmp_path / "history_pairs.csv")
    assert set(hp["field"]) == {"reward", "completion_tokens", "truncated", "groups"} and (hp["cells"] == 1).all()
    with pytest.raises(SystemExit, match="not a clone"):  # st-si-c1 continues st-si, not st
        arms.main([str(run_dir), "--parents", "st", "--arms", "c1,si-c1", "--reference", "c1", "--out", str(tmp_path)])


def test_paired_arms_across_tags(run_dir, tmp_path):
    from evotaste.analysis import paired

    paired.main([str(run_dir), "--tags", "par,sub", "--reference", "c1", "--arms", "c2", "--boot", "30",
                 "--deviation", str(data.deviation_selection(data.Run(run_dir))), "--out", str(tmp_path)])
    s = pd.read_csv(tmp_path / "summary.csv").set_index("metric")
    grid = data.load_eval(data.Run(run_dir), "par,sub").real()
    acc = grid.x["reward"].mean((1, 2))
    assert s.loc["accuracy", "mean"] == pytest.approx(acc[grid.labels.index("st-c2@2")] - acc[grid.labels.index("st-c1@2")])
    assert s.loc["accuracy", "parents"] == 1 and "deviation" in s.index  # ta and st-si have the reference only
    with pytest.raises(SystemExit, match="reference clones"):  # ta-c1 would have no reference
        paired.main([str(run_dir), "--tags", "sub", "--reference", "c2", "--arms", "c1", "--out", str(tmp_path)])


def test_rule_names_name_every_computed_term():
    from evotaste.config import Amplification, Exaggeration, Practice, Revisit, Taste

    assert data.rule_name(Amplification(taste=Taste(0.0))) == "centered"  # pilot E's neutral clones: the parent's rule
    assert data.rule_name(Amplification(taste=Taste(-0.25))) == "centered+taste-0.25"
    assert data.rule_name(Amplification(revisit=Revisit(0.33, "own"))) == "centered+rvown0.33"
    assert data.rule_name(Amplification(revisit=Revisit(0.0, "own"))) == "centered"

    assert data.rule_name(Amplification()) == "centered"
    assert data.rule_name(Amplification(self_imitation=0.5)) == "centered+si0.5"
    assert data.rule_name(Amplification(exaggeration=Exaggeration(1.0, 0.01), imitate_within_free=True)) == \
        "centered+ex1x0.01~free"
    assert data.rule_name(Amplification(exaggeration=Exaggeration(0.0, 0.5), practice=Practice(0.0, 0.8, 0.25, 4.0))) \
        == "centered"  # gamma 0 and eta 0: terms the engine skips
    assert data.rule_name(Amplification(practice=Practice(4.0, 0.8, 0.25, 4.0))) == "centered+pr4"


def test_harvest_simulation_finds_no_payoff_in_an_exact_null(run_dir, tmp_path):
    harvest.main([str(run_dir), "--tag", TAG, "--splits", "40", "--out", str(tmp_path)])
    summary = pd.read_csv(tmp_path / "summary.csv").set_index(["mode", "members", "statistic"])
    for mode in ("units", "items"):
        lead = summary.loc[(mode, "all", "test_lead_top1")]
        assert lead["lo"] < 0 < lead["hi"]  # selecting on one half buys nothing on the other
        assert summary.loc[(mode, "all", "select_lead_top1"), "value"] > 0
        assert summary.loc[(mode, "all", "best_of_1"), "value"] == pytest.approx(0, abs=1e-12)
    rel = pd.read_csv(tmp_path / "reliability.csv").set_index("members")
    assert rel.loc["all", "m"] == 9 and rel.loc["all", "between_lo"] < 0 < rel.loc["all", "between_hi"]
