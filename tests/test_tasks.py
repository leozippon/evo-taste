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
    t = tasks.Task("gsm8k_train/x", "gsm8k_train", "q", "not a number")  # float(gold) raises inside the scorer
    s = score(t, "<answer>5</answer>", "stop")
    assert s.reward == 0.0 and "ValueError" in s.error


def test_scoring_is_bounded_in_wall_time(monkeypatch):
    """A scorer stuck in C code (sympy on a huge exponent) is killed at the timeout; the pool keeps working."""
    import time

    monkeypatch.setattr(tasks, "SCORE_TIMEOUT_S", 3)
    bombs = [instance("eval", f, 0) for f in ("simple_integration", "intermediate_integration",
                                              "polynomial_multiplication")]
    t0 = time.monotonic()
    scores = tasks.score_many([(t, "<answer>1e999999999</answer>", "stop") for t in bombs]
                              + [(t, f"<answer>{t.answer}</answer>", "stop") for t in bombs])
    assert time.monotonic() - t0 < 15
    assert [(s.reward, s.error) for s in scores[:3]] == [(0.0, "timeout")] * 3
    assert [s.reward for s in scores[3:]] == [1.0] * 3


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
        assert {seen[g] for g in info["unit"] if g in seen} == {seen[f]}, f"near-duplicate unit {info['unit']} split"
        assert info["reward_check"]["gold_reward_min"] == 1.0
        assert info["adversarial_check"]["constant"][0] == 0 and info["adversarial_check"]["prefix"][0] == 0
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


def test_generator_failure_is_redrawn_deterministically(monkeypatch, caplog, suites, probes_cfg):
    real = tasks.procedural_task
    first = instance("stream", "chain_sum", 3)
    family, seed, index = first.uid.split("/")
    planted = (family, int(seed), int(index))

    def flaky(f, s, i):
        if (f, s, i) == planted:
            raise RuntimeError("Could not find suitable path cell after 1000 attempts")
        return real(f, s, i)

    monkeypatch.setattr(tasks, "procedural_task", flaky)
    monkeypatch.setattr(tasks, "_GENERATOR_FAILURES", set())
    with caplog.at_level("WARNING"):
        a, b = instance("stream", "chain_sum", 3), instance("stream", "chain_sum", 3)
    assert a.uid == b.uid == f"{family}/{seed}/{int(index) + 1}"  # the next draw, the same every time
    assert sum("generator of chain_sum failed" in r.message for r in caplog.records) == 1  # logged once
    assert instance("stream", "chain_sum", 4).uid == real(family, int(seed), int(index) + tasks.REDRAWS).uid
    assert [t.uid for t in tasks.fixed_tasks(["chain_sum"], "stream", 5)][3] == a.uid

    s = Streams(suites, 7, probes_cfg)  # the training stream redraws the same way
    j = next(j for j in range(200) if tasks.family_at(suites.train, "train/7", j)[0] == suites.train[0])
    fam, n = tasks.family_at(suites.train, "train/7", j)
    target = next(tasks.draws("train/7", fam, n)).uid.split("/")
    planted = (target[0], int(target[1]), int(target[2]))
    assert s.train_batch(0, j + 1)[j].uid == f"{target[0]}/{target[1]}/{int(target[2]) + 1}"

    planted = None
    monkeypatch.setattr(tasks, "procedural_task", lambda f, s, i: (_ for _ in ()).throw(ValueError("always")))
    with pytest.raises(RuntimeError, match="all 16 draws raised"):
        instance("stream", "chain_sum", 3)


def test_unusable_gold_takes_the_next_draw_in_fixed_sets(monkeypatch, caplog):
    real = tasks.procedural_task
    clean = tasks.fixed_tasks(["chain_sum"], "evalx", 4)
    family, seed, index = clean[2].uid.split("/")

    def planted(f, s, i):
        t = real(f, s, i)
        if (f, s, i) == (family, int(seed), int(index)):  # a gold answer its own scorer rejects
            return tasks.Task(t.uid, t.family, t.question, t.answer + "999", t.entry)
        return t

    monkeypatch.setattr(tasks, "procedural_task", planted)
    monkeypatch.setattr(tasks, "_GOLD_FAILURES", set())
    with caplog.at_level("WARNING"):
        a, b = tasks.fixed_tasks(["chain_sum"], "evalx", 4), tasks.fixed_tasks(["chain_sum"], "evalx", 4)
    assert [t.uid for t in a] == [t.uid for t in b]  # a pure function of the position
    assert [t.uid for i, t in enumerate(a) if i != 2] == [t.uid for i, t in enumerate(clean) if i != 2]
    assert a[2].uid == f"{family}/{seed}/{int(index) + 1}"
    assert sum("does not earn reward 1" in r.message for r in caplog.records) == 1
    tasks.check_gold(a)


def test_usable_items_skip_bad_gold():
    items = [tasks.Task(f"gsm8k_train/{i}", "gsm8k_train", "q", "x" if i == 1 else str(i)) for i in range(6)]
    assert [t.uid for t in tasks.usable_items(items, 3)] == ["gsm8k_train/0", "gsm8k_train/2", "gsm8k_train/3"]
    assert [t.uid for t in tasks.usable_items(items, 2, from_end=True)] == ["gsm8k_train/4", "gsm8k_train/5"]
    assert [t.uid for t in tasks.usable_items(items, 5, from_end=True)] == [f"gsm8k_train/{i}" for i in (0, 2, 3, 4, 5)]
    assert len(tasks.usable_items(items, 100)) == 5  # more than the part: every usable item


def test_dataset_build_is_not_part_of_the_scorer_time_limit():
    """acre builds 8,192 items when a scorer process first meets it (about 2 s): that may not count as a timeout,
    while a scorer that hangs on the answer still does."""
    import time

    pool = tasks.ScorerPool(1)
    try:
        acre = instance("buildtest", "acre", 0)
        assert pool.run([(acre, acre.answer)], timeout=0.5) == [(1.0, None)]
        bomb = instance("buildtest", "simple_integration", 0)
        t0 = time.monotonic()
        assert pool.run([(bomb, "1e999999999")], timeout=2)[0] == (0.0, "timeout")
        assert time.monotonic() - t0 < 60
        assert pool.run([(acre, acre.answer)], timeout=0.5) == [(1.0, None)]  # a replaced process rebuilds
    finally:
        pool.close()


def test_gold_check_timeout_is_an_error_not_a_redraw(monkeypatch):
    def slow(jobs, timeout=None):
        assert timeout == tasks.GOLD_TIMEOUT_S
        return [tasks.Score(0.0, 0.0, True, False, "x", "timeout") for _ in jobs]

    monkeypatch.setattr(tasks, "score_many", slow)
    with pytest.raises(RuntimeError, match="timed out"):
        tasks.fixed_tasks(["chain_sum"], "evaly", 2)
    with pytest.raises(RuntimeError, match="timed out"):
        tasks.check_gold([instance("evaly", "chain_sum", 0)])
