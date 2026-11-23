"""Final-evaluation analysis: bootstrap coverage, excess variance, winner's curse, the blind gate and niche counts."""

import gzip

import numpy as np
import pandas as pd
import pytest

from evotaste.analysis import harvest, niches
from evotaste.analysis.harvest import Suite, spread, winners_curse
from evotaste.analysis.rundata import PROCEDURAL_UID, Evaluation, load_evaluation
from evotaste.records import write_json, write_jsonl_gz


def _evaluation(reward: np.ndarray, families: list[str], adapters: list[str]) -> Evaluation:
    """In-memory evaluation of procedural families: reward is adapter x instance x sample."""
    return Evaluation("selection", "t", families.count(families[0]), reward.shape[2], tuple(adapters),
                      tuple(f"{f}/0/{i}" for i, f in enumerate(families)), np.array(families),
                      np.ones(len(families), bool), reward, np.full(reward.shape, 50.0), ())


def _draw(rng, p_members: np.ndarray, n: int, samples: int = 1) -> tuple[np.ndarray, list[str]]:
    """Rewards for members x families success probabilities, n instances per family."""
    m, f = p_members.shape
    p = np.repeat(p_members, n, axis=1)[:, :, None]
    return (rng.random((m, f * n, samples)) < p).astype(float), [f"f{j}" for j in range(f) for _ in range(n)]


def test_bootstrap_interval_covers_the_planted_macro_at_the_nominal_rate():
    rng = np.random.default_rng(0)
    p_family = np.linspace(0.1, 0.8, 8)
    hits = 0
    for rep in range(200):
        # instance difficulty varies within a family and the 4 samples of an instance are correlated
        q = rng.beta(2 * p_family / (1 - p_family), 2, size=(30, 8)).T.reshape(1, -1)  # family means p_family
        reward = (rng.random((1, q.size, 4)) < q[:, :, None]).astype(float)
        s = Suite(_evaluation(reward, [f"f{j}" for j in range(8) for _ in range(30)], ["a"]), seed=rep)
        lo, hi = harvest.ci(s.macro_boot[0])
        hits += lo <= p_family.mean() <= hi
    assert 0.90 <= hits / 200 <= 0.99


def test_excess_variance_is_zero_for_identical_members_and_recovers_a_planted_spread():
    rng = np.random.default_rng(1)
    p_family = np.linspace(0.2, 0.7, 10)
    same, fam = _draw(rng, np.tile(p_family, (20, 1)), 200)
    s = Suite(_evaluation(same, fam, [f"m{i}" for i in range(20)]), seed=0)
    r = spread(s.macro, s.noise_var(s.ev.adapters), rng)
    assert r["excess_lo"] <= 0 <= r["excess_hi"] and abs(r["excess_var"]) < r["noise_sd"] ** 2
    offsets = np.linspace(-0.1, 0.1, 20)
    planted, fam = _draw(rng, p_family[None, :] + offsets[:, None], 200)
    s = Suite(_evaluation(planted, fam, [f"m{i}" for i in range(20)]), seed=0)
    r = spread(s.macro, s.noise_var(s.ev.adapters), rng)
    assert abs(r["excess_var"] / offsets.var(ddof=1) - 1) < 0.25 and r["excess_lo"] > 0


def _two_suites(rng, offsets_sel, offsets_blind, n=200):
    names = [f"m{i}" for i in range(len(offsets_sel))]
    p_family = np.linspace(0.2, 0.7, 5)
    sel, fam = _draw(rng, p_family[None, :] + offsets_sel[:, None], n)
    blind, _ = _draw(rng, p_family[None, :] + offsets_blind[:, None], n)
    s, b = Suite(_evaluation(sel, fam, names), seed=0), Suite(_evaluation(blind, fam, names), seed=1)
    ranked = sorted(names, key=lambda a: -s.macro[s.pos[a]])
    roles = {a: {"role": "member", "arm": "plain", "bias": "plain"} for a in names}
    return winners_curse(s, b, None, roles, ranked[0], ranked[:3], rng)


def test_selection_blind_correlation_and_retained_fraction_recover_planted_values_and_vanish_under_noise():
    rng = np.random.default_rng(2)
    skill = np.linspace(-0.15, 0.15, 20)
    w = _two_suites(rng, skill, skill)  # transfer is perfect: retained fraction 1
    assert w["spearman"]["r"] > 0.9 and w["spearman"]["p_perm"] < 0.001 and w["pearson"]["r"] > 0.9
    assert abs(w["retained_outlier"]["value"] - 1) < 0.25
    assert w["retained_outlier"]["lo"] <= 1 <= w["retained_outlier"]["hi"]
    w = _two_suites(rng, skill, -skill)  # transfer is reversed: correlation and retained fraction -1
    assert w["spearman"]["r"] < -0.9 and abs(w["retained_top3"]["value"] + 1) < 0.25
    w = _two_suites(rng, np.zeros(20), np.zeros(20))  # identical members: differences are evaluation noise
    assert abs(w["spearman"]["r"]) < 0.45 and w["spearman"]["p_perm"] > 0.05
    assert w["retained_outlier"]["lo"] <= 0 <= w["retained_outlier"]["hi"]


# -- synthetic run directories --------------------------------------------------------------------------------------

CONFIG = """name: {name}
model: tiny
seed: 0
scaffolds: {scaffolds}
members:
- {{id: m1, arm: plain, seed: 1}}
- {{id: m2, arm: plain, seed: 2}}
- {{id: m3, arm: scaffold, scaffold: verify, seed: 3}}
- {{id: bl, arm: plain, seed: 4, role: baseline}}
"""


def _run(tmp_path, name="syn"):
    run = tmp_path / "runs" / name
    run.mkdir(parents=True)
    (run / "scaffolds.yaml").write_text("preamble: P\ncontrol:\n  cot: {text: c}\nscaffolds:\n  verify: {text: v}\n")
    (run / "config.yaml").write_text(CONFIG.format(name=name, scaffolds=run / "scaffolds.yaml"))
    return run


def _write_eval(run, suite, tag, correct: dict[str, list[int]], uids: list[str], samples: int):
    """Records where adapter a gets the first correct[a][i] of its samples on instance i right."""
    records = [{"adapter": a, "adapter_sha256": None, "suite": suite, "sample": k, "uid": u, "family": u.split("/")[0],
                "question": f"q {u}", "answer": "1", "extracted": "1" if k < c else "0", "reward": float(k < c),
                "completion_tokens": 40 + 100 * k, "parsed": True, "truncated": False}
               for a, cs in correct.items() for u, c in zip(uids, cs) for k in range(samples)]
    out = run / "eval" / suite / tag
    write_json(out / "params.json", {"suite": suite, "per_family": 3, "samples": samples})
    write_jsonl_gz(out / "records-x-0of1.jsonl.gz", records)
    macro = {}
    for a, cs in correct.items():
        fams = {}
        for u, c in zip(uids, cs):
            if PROCEDURAL_UID.match(u):
                fams.setdefault(u.split("/")[0], []).append(c / samples)
        macro[a] = {"macro": float(np.mean([np.mean(v) for v in fams.values()]))}
    write_json(out / "summary.json", {"adapters": macro})


UIDS = ["fa/1/0", "fa/1/1", "fa/1/2", "fb/1/0", "fb/1/1", "fb/1/2", "fc/1/0", "gsm8k_train/7"]
CORRECT = {  # correct samples out of 4 per instance
    "base": [4, 1, 0, 0, 0, 2, 0, 4],
    "m1": [4, 2, 0, 3, 0, 0, 2, 4],
    "m2": [4, 0, 2, 0, 1, 2, 0, 0],
    "m3": [4, 0, 0, 3, 0, 0, 0, 1],
    "bl": [4, 0, 1, 0, 0, 4, 0, 4],
}


def test_niche_counts_are_exact_on_a_hand_built_example(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # no harvest manifest here: the outlier is provisional
    run = _run(tmp_path)
    _write_eval(run, "selection", "t", CORRECT, UIDS, samples=4)
    niches.main([str(run), "--suite", "selection", "--tag", "t", "--out", str(tmp_path / "n")])
    assert "m1 (PROVISIONAL) vs base, bl" in capsys.readouterr().out
    j = pd.read_csv(tmp_path / "n" / "jaccard.csv").set_index(["a", "b"])["jaccard"]
    # solved (>= 2 of 4): m1 {0,1,3,6,7}, m2 {0,2,5}, m3 {0,3}; base {0,5,7}; bl {0,5,7}
    assert j[("m1", "m2")] == pytest.approx(1 / 7) and j[("m1", "m3")] == pytest.approx(2 / 5)
    assert j[("m2", "m3")] == pytest.approx(1 / 4)
    a = pd.read_csv(tmp_path / "n" / "adapters.csv").set_index("adapter")
    assert a.loc[["m1", "m2", "m3"], "unique"].tolist() == [3, 2, 0]
    assert a.loc[["m1", "m2", "m3", "bl"], ["preserved", "gained", "lost"]].to_numpy().tolist() == \
        [[2, 3, 1], [2, 1, 1], [1, 1, 2], [3, 0, 0]]
    assert a.loc["m2", "ever"] == 4 and a.loc["bl", "role"] == "baseline"
    cov = pd.read_csv(tmp_path / "n" / "coverage.csv").set_index("k")["mean_coverage"]
    # members solving each instance: 3, 1, 1, 2, 0, 1, 1, 1
    assert cov.to_dict() == pytest.approx({1: 10 / 3, 2: 1 + 5 * 2 / 3 + 1, 3: 7})
    u = pd.read_csv(tmp_path / "n" / "unlocks.csv")
    assert u[["adapter", "family", "n"]].values.tolist() == [["m1", "fc", 4]] and u["accuracy"][0] == 0.5
    d = pd.read_csv(tmp_path / "n" / "designated.csv").set_index("family")
    # comparators ever solve {0,1,2,5,7}: m1 alone solves 3 (fb) and 6 (fc); comparators solve 5 (fb), m1 never does
    assert d.loc["all"].tolist() == [2, 1] and d.loc["fb"].tolist() == [1, 1] and d.loc["fc"].tolist() == [1, 0]


def test_blind_vault_is_read_only_with_the_flag_and_a_manifest(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    run = _run(tmp_path)
    _write_eval(run, "selection", "t", CORRECT, UIDS, samples=4)
    _write_eval(run, "blind1", "b", CORRECT, UIDS, samples=4)
    junk = run / "eval" / "blind2" / "b"
    junk.mkdir(parents=True)
    (junk / "params.json").write_text("not json")  # any read of blind2 would fail
    (junk / "records-x-0of1.jsonl.gz").write_bytes(gzip.compress(b"not json\n"))

    with pytest.raises(PermissionError, match="--blind"):
        load_evaluation(run, "blind1", "b")
    with pytest.raises(PermissionError, match="no harvest manifest"):
        load_evaluation(run, "blind1", "b", blind=True)
    for tool, args in ((harvest, ["--tag", "t", "--blind", "blind1", "--blind-tag", "b"]),
                       (niches, ["--blind", "blind1", "--tag", "b", "--adapter", "m1"])):
        with pytest.raises(PermissionError, match="no harvest manifest"):
            tool.main([str(run), *args, "--out", str(tmp_path / "o")])
    harvest.main([str(run), "--tag", "t", "--out", str(tmp_path / "h")])  # never touches the vaults
    assert "PROVISIONAL" in capsys.readouterr().out

    write_json(tmp_path / "docs" / "harvest" / "syn.json", {"outlier": "m1", "top3": ["m1", "m2", "m3"], "tag": "t"})
    harvest.main([str(run), "--tag", "t", "--blind", "blind1", "--blind-tag", "b", "--out", str(tmp_path / "h")])
    out = capsys.readouterr().out
    assert "outlier m1 (harvested" in out and "winner's curse" in out
    comparisons = pd.read_csv(tmp_path / "h" / "comparisons.csv")
    assert set(comparisons["suite"]) == {"selection", "blind1"}
    assert set(zip(comparisons["comparator"], comparisons["kind"])) == {
        ("base", "base"), ("bl", "baseline"), ("m2", "median member"), ("mean", "population mean")}
