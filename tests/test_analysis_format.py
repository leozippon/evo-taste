"""Format-lenient re-scoring: each transform, the no-false-credit guard on real instances, the time bound, and the
variance decomposition identity."""

import time

import numpy as np
import pandas as pd
import pytest
import sympy

from evotaste.analysis import format as fmt
from evotaste.analysis.rundata import Evaluation
from evotaste.tasks import procedural_task


@pytest.mark.parametrize("name, answer, gold, expected", [
    ("strip", "  $\\boxed{42}$. ", "42", "42"),
    ("strip", "**Paris**", "paris", "Paris"),
    ("strip", "'a', 'b'", "a, b", "'a', 'b'"),  # an inner quote keeps the outer pair
    ("latex", "\\frac{3}{4}x \\cdot y", "", "(3)/(4)x * y"),
    ("latex", "\\sqrt{x^{2}} − 1", "", "sqrt(x^(2)) - 1"),
    ("exponent", "-140x⁶ + x^2 - y⁻¹", "", "-140x**6 + x**2 - y**-1"),
    ("implicit", "3x + 2(x+1)(x-1) + 2e-3", "", "3*x + 2*(x+1)*(x-1) + 2e-3"),
    ("implicit", "x sin(x)", "", "x*sin(x)"),
    ("numeric", "237.0", "", "237"),
    ("numeric", "+1,234.50", "", "1234.5"),
    ("numeric", ".5", "", "0.5"),
    ("numeric", "2x", "", "2x"),
    ("case_space", "  Fill   A ", "", "fill a"),
    ("list", "held,identify\nfor", "held, identify, for", "held, identify, for"),
    ("list", "[fill A; pour A->B]", '["fill A", "pour A->B"]', '["fill A", "pour A->B"]'),
    ("list", "1 2 3", "['1', '2', '3']", "['1', '2', '3']"),
])
def test_each_transform(name, answer, gold, expected):
    assert fmt.TRANSFORMS[name](answer, gold) == expected


def test_equality_criteria():
    assert fmt.numeric_equal("2.50e2", "250") and not fmt.numeric_equal("6699/77", "87")
    assert not fmt.numeric_equal("237.4", "237")
    g = "exp(x)*sin(x) - exp(x)*cos(x) + C"
    assert fmt.symbolic_applicable("e**x*(sin(x) - cos(x)) + 7", g) and fmt.symbolic_equal("e**x*(sin(x) - cos(x)) + 7", g)
    assert not fmt.symbolic_equal("x*(3*x + 2)", "3*x**2 + 2*x")  # an unexpanded product is not the expansion
    assert not fmt.symbolic_applicable("__import__('os')", "x + 1")


def _evaluation(rows):
    """One adapter, one sample: rows of (task, extracted answer)."""
    tasks, answers = zip(*rows)
    reward = np.array([[[float(a == t.answer)] for t, a in rows]])
    return Evaluation("selection", "t", 1, 1, ("a",), tuple(t.uid for t in tasks), np.array([t.family for t in tasks]),
                      np.ones(len(rows), bool), reward, np.full(reward.shape, 50.0), (),
                      np.array([[[a] for a in answers]], dtype=object), np.zeros(reward.shape, bool),
                      np.array([t.answer for t in tasks], dtype=object))


def test_lenient_rescoring_rescues_format_and_never_credits_content(monkeypatch):
    geo, geo2 = procedural_task("simple_geometry", 7, 0), procedural_task("simple_geometry", 7, 1)
    words, words2 = procedural_task("word_sorting", 7, 0), procedural_task("word_sorting", 7, 1)
    poly, poly2 = procedural_task("polynomial_multiplication", 7, 0), procedural_task("polynomial_multiplication", 7, 1)
    arith = procedural_task("basic_arithmetic", 7, 0)
    assert words.answer != words.answer.lower()  # the gold has capitals: lowering case changes the answer
    caret = poly2.answer.replace("**", "^").replace("*", "")  # -152y^3 - 2339y^2 + 10863y: notation only
    factored = str(sympy.factor(sympy.parse_expr(poly.answer)))
    rows = [(geo, f"{geo.answer}.0"),  # format: rescued (numeric)
            (geo2, f"{int(geo2.answer) + 1}.0"),  # wrong value: not rescued
            (words, words.answer.lower()),  # wrong case in a case-sensitive sort: never credited
            (words2, words2.answer.replace(", ", "\n")),  # separators only: rescued (list)
            (poly, factored),  # equal value but not expanded: not credited
            (poly2, caret),  # notation only: rescued (combined)
            (arith, f"({arith.answer} - 1) + 1"),  # an unevaluated expression: not credited
            (geo, geo.answer)]  # a correct answer, for the guard
    ev = _evaluation(rows)
    lenient, rescued, info = fmt.rescore(ev, processes=2)
    by = dict(zip(rescued["extracted"], rescued["transform"]))
    assert by == {f"{geo.answer}.0": "numeric", words2.answer.replace(", ", "\n"): "list", caret: "combined"}
    assert lenient.reward[0, :, 0].tolist() == [1, 0, 0, 1, 0, 1, 0, 1]
    assert "case_space" not in info["safe"]["word_sorting"] and "numeric" in info["safe"]["simple_geometry"]
    g = info["guard"].set_index(["family", "transform"])
    assert g.loc[("word_sorting", "case_space"), "violations"] > 0
    # no enabled transform, alone or chained, turns a gold or correct answer into a rejected one
    assert not g.loc[[(f, t) for f, ts in info["safe"].items() for t in ts if (f, t) in g.index], "violations"].any()


def test_a_hanging_job_is_killed_within_the_bound(monkeypatch):
    monkeypatch.setattr(fmt, "JOB_TIMEOUT_S", 1)
    monkeypatch.setitem(fmt._WORK, "sleep", lambda job: time.sleep(job))
    t0 = time.monotonic()
    out = fmt.run_jobs("sleep", [30, 0], processes=2)
    assert out == [(None, "timeout"), (None, None)] and time.monotonic() - t0 < 10


def test_decomposition_identity():
    rng = np.random.default_rng(0)
    strict = rng.random(12) * 0.4
    lenient = strict + rng.random(12) * 0.1
    draws = lambda v: v[:, None] + rng.normal(0, 0.01, (12, 200))  # noqa: E731
    d = fmt.decomposition(strict, lenient, draws(strict), draws(lenient), rng)
    assert d["var_strict"] == pytest.approx(d["var_lenient"] + d["var_loss"] - 2 * d["cov_lenient_loss"], rel=1e-12)
    assert d["var_loss"] == pytest.approx(np.var(lenient - strict, ddof=1))


def test_blind_mode_needs_the_manifest_and_compares_both_suites(tmp_path, monkeypatch, capsys):
    from evotaste.records import read_jsonl_gz, write_json, write_jsonl_gz
    from evotaste.tasks import Task
    from tests.test_analysis_eval import CORRECT, UIDS, _run, _write_eval

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fmt, "make_task", lambda uid, family, gold: Task(uid, "gsm8k_train", "", gold))  # number
    run = _run(tmp_path)
    for suite, tag in (("selection", "t"), ("blind1", "b")):
        _write_eval(run, suite, tag, CORRECT, UIDS, samples=4)
        path = run / "eval" / suite / tag / "records-x-0of1.jsonl.gz"
        records = read_jsonl_gz(path)
        for r in records:
            if r["adapter"] == "m2" and r["reward"] == 0:
                r["extracted"] = "1.0"  # right value, wrong format
        write_jsonl_gz(path, records)
    args = [str(run), "--blind", "blind1", "--tag", "b", "--selection-tag", "t", "--processes", "2",
            "--out", str(tmp_path / "f")]
    with pytest.raises(PermissionError, match="no harvest manifest"):
        fmt.main(args)
    write_json(tmp_path / "docs" / "harvest" / "syn.json", {"outlier": "m1", "top3": ["m1", "m2", "m3"], "tag": "t"})
    fmt.main(args)
    assert "selection vs blind1 across 3 members" in capsys.readouterr().out
    rescues = pd.read_csv(tmp_path / "f" / "rescues.csv")
    assert rescues.groupby("transform")["records"].sum().to_dict() == {"numeric": 4 * 8 - 9}  # m2's wrong samples
    comparisons = pd.read_csv(tmp_path / "f" / "outlier_comparisons.csv")
    assert set(zip(comparisons["scoring"], comparisons["suite"])) == {
        ("strict", "selection"), ("strict", "blind1"), ("lenient", "selection"), ("lenient", "blind1")}
