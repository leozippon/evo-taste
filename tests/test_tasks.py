"""Scoring guards, external scorers, stream determinism and probe disjointness, suite disjointness and isolation."""

from types import SimpleNamespace

import pytest
import yaml

from evotaste import external, tasks
from evotaste.tasks import Streams, Suites, instance, score

SUITES = "configs/suites.yaml"


def test_scoring_guards():
    t = instance("test", "prime_factorization", 0)
    gold = f"<answer>{t.answer}</answer>"
    assert score(t, gold, "stop").reward == 1.0
    assert score(t, "reasoning " + gold + " more text", "stop").reward == 1.0  # last <answer> span counts
    no_answer = score(t, t.answer, "stop")
    assert (no_answer.reward, no_answer.parsed) == (0.0, False)
    truncated = score(t, gold, "length")
    assert (truncated.reward, truncated.truncated, truncated.parsed) == (0.0, True, True)
    raised = score(t, "<answer>not a number</answer>", "stop")  # this scorer raises on non-numeric answers
    assert raised.reward == 0.0 and raised.error is not None


def test_format_credit_is_not_reward():
    t = instance("test", "countdown", 0)  # gives 0.01 to a missing or wrong answer
    s = score(t, "<answer>1 + 1</answer>", "stop")
    assert 0.0 < s.score < 1.0 and s.reward == 0.0


def test_scorer_exception_is_caught():
    t = tasks.Task("x/0/0", "x", "q", "a", lambda a: 1 / 0)
    s = score(t, "<answer>a</answer>", "stop")
    assert s.reward == 0.0 and "ZeroDivisionError" in s.error


def test_external_scorers():
    assert external.score_number("18 dollars", "18") == 1.0
    assert external.score_number("1,234", "1234") == 1.0
    assert external.score_number("18 or 19", "18") == 0.0
    assert external.score_letter("(C) because", "C") == 1.0 and external.score_letter("C", "D") == 0.0
    assert external.score_letter("Cats", "C") == 0.0
    assert external.score_math("\\dfrac{\\pi}{2}", "\\frac{\\pi}{2}") == 1.0
    assert external.score_math("\\$18.90", "\\$18.90") == 1.0
    assert external.score_math("zzz", "3\\sqrt{13}") == 0.0


@pytest.fixture(scope="module")
def suites():
    return Suites(SUITES)


@pytest.fixture(scope="module")
def probes_cfg():
    return SimpleNamespace(train_per_family=2, selection_per_family=1, foundation_per_source=3)


def test_streams_deterministic(suites, probes_cfg):
    a, b = Streams(suites, 7, probes_cfg), Streams(suites, 7, probes_cfg)
    assert [t.uid for t in a.train_batch(3, 40)] == [t.uid for t in b.train_batch(3, 40)]
    assert [t.question for t in a.seed_pool(50)] == [t.question for t in b.seed_pool(50)]
    other = Streams(suites, 8, probes_cfg)
    assert [t.uid for t in a.train_batch(3, 40)] != [t.uid for t in other.train_batch(3, 40)]
    assert [t.uid for t in a.probe_tasks["train"]] == [t.uid for t in other.probe_tasks["train"]]  # run-independent
    batch = a.train_batch(0, 2 * len(suites.train))
    assert sorted(t.family for t in batch) == sorted(2 * list(suites.train))  # every family once per block
    assert {t.family for t in a.train_batch(0, 64)} <= set(suites.train)


def test_streams_avoid_probe_questions(suites, probes_cfg):
    s = Streams(suites, 7, probes_cfg)
    probe_questions = {t.question for t in s.probe_tasks["train"]}
    items = s.train_batch(0, 200) + s.seed_pool(200)
    assert not probe_questions & {t.question for t in items}
    first = s.train_batch(0, 1)[0]
    s._probe_questions.add(first.question)  # force a collision: the item must be redrawn deterministically
    redrawn = s.train_batch(0, 1)[0]
    assert redrawn.family == first.family and redrawn.question != first.question
    assert redrawn.uid == s.train_batch(0, 1)[0].uid


def test_suite_disjointness():
    data = yaml.safe_load(open(SUITES))
    suites = data["suites"]
    seen = {}
    for name, fams in suites.items():
        for f in fams:
            assert f not in seen, f"{f} in {seen.get(f)} and {name}"
            seen[f] = name
    assert not set(seen) & set(data["excluded"])
    for f, info in data["families"].items():
        assert {seen[g] for g in info["unit"]} == {seen[f]}, f"near-duplicate unit {info['unit']} split"
        assert info["reward_check"]["gold_reward_min"] == 1.0
        assert info["reward_check"]["empty_reward_max"] == 0.0 and info["reward_check"]["junk_reward_max"] == 0.0
    for suite, cat in data["held_out_categories"].items():
        if suite != "pooled":
            fams = [f for f, i in data["families"].items() if i["category"] == cat]
            assert fams and all(seen[f] == suite for f in fams)
    for f in tasks.Suites(SUITES).train:  # a held-out category never reaches training
        assert data["families"][f]["category"] not in {data["held_out_categories"][s] for s in ("selection", "blind1", "blind2")}


def test_blind_vaults_need_the_guard():
    s = Suites(SUITES)
    assert not hasattr(s, "blind1") and not hasattr(s, "blind2")
    for vault in ("blind1", "blind2"):
        with pytest.raises(PermissionError):
            tasks.blind_suite(SUITES, vault, None)
        with pytest.raises(PermissionError):
            external.items("math500", vault)
        with pytest.raises(PermissionError):
            tasks.blind_suite(SUITES, vault, tasks.BlindReceipt("blind1" if vault == "blind2" else "blind2", ""))


def test_identical_external_questions_share_a_part():
    """Data only: a normalised question text maps to one part across all ids and all external sources."""
    parts, counts = {}, {}
    for src in external.PARTS:
        for _, r in external._rows(src):
            q = external.question_text(r)
            part = external.part_of(src, q)
            parts.setdefault(external.normalized(q), set()).add(part)
            counts[(src, part)] = counts.get((src, part), 0) + 1
    assert all(len(p) == 1 for p in parts.values())
    mmlu = [counts[("mmlu_pro", p)] for p in ("foundation", "blind1", "blind2")]
    assert min(mmlu) > 0.3 * sum(mmlu)


def test_worker_code_never_names_blind_vaults():
    """Only evaluate.py (and the guarded accessors) may refer to blind vaults."""
    import pathlib

    for module in ("worker", "member", "trainer", "sampler"):
        text = pathlib.Path(f"evotaste/{module}.py").read_text()
        assert "blind" not in text and "vault" not in text, module
