"""Tasks, instructions, scoring, the training stream and the seeding prompts (synthetic pools), probe sets."""

import collections
import pathlib
from fractions import Fraction

import pytest

from evotaste import sources, tasks
from evotaste.sources import Item
from evotaste.tasks import Streams, make_task, rewrap, score_many
from evotaste.verify import HELDOUT_WRAPPERS, TRAIN_WRAPPERS

SUITES = "configs/suites-r2.yaml"


def open_item(source="gsm8k_train", k=0, gold="18"):
    return Item(source, str(k), "math", "open", "How many eggs are left?", gold, "How many eggs are left?",
                "How many eggs are left?")


def test_instructions():
    t = make_task(open_item())
    assert (t.wrapper, t.channels) == ("box", ("direct", "program")) and "\\boxed{}" in t.question
    assert tasks.PROGRAM_OFFER in t.question and t.question.startswith(t.problem)
    v = rewrap(t, "final_line")
    assert (v.wrapper, v.channels) == ("final_line", ("direct",))
    assert "Final answer:" in v.question and tasks.PROGRAM_OFFER not in v.question and "\\boxed" not in v.question
    forced = rewrap(t, channels=("program",))
    assert forced.question.endswith(tasks.PROGRAM_ONLY) and "\\boxed" not in forced.question
    choice = make_task(sources._choice("aqua", "1", "math", "Which?", ["2", "3"], 1))
    assert choice.channels == () and "letter of the correct option" in choice.question
    verdict = make_task(sources._verdict("folio", "1", "logic", "P. C?", ("True", "False", "Uncertain"), "True"))
    assert "exactly one of: True, False, Uncertain" in verdict.question
    code = make_task(Item("kodcode", "x", "code", "code", "Add two numbers.",
                          {"harness": "pytest", "tests": "", "reference": "", "entry_point": "add",
                           "signature": "def add(a, b):"}, "", ""))
    assert code.wrapper == "code" and "`def add(a, b)`" in code.question


def test_prompt_ids_append_the_context_to_the_system_prompt():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    plain = tok.decode(tasks.prompt_ids(tok, "Q?"))
    with_context = tok.decode(tasks.prompt_ids(tok, "Q?", context="How you work: carefully."))
    assert tasks.SYSTEM_PROMPT in plain and "How you work" not in plain
    assert f"{tasks.SYSTEM_PROMPT}\n\nHow you work: carefully." in with_context
    assert "<think>\n\n</think>" in plain  # thinking disabled


def test_scores_follow_the_wrapper_the_channel_and_truncation():
    t = make_task(open_item())
    boxed, final = "So 18 eggs.\n\\boxed{18}", "So 18 eggs.\nFinal answer: 18"
    program = "Compute it.\n```python\nprint(20 - 2)\n```"
    s = score_many([(t, boxed, "stop"), (t, boxed, "length"), (t, final, "stop"), (t, program, "stop"),
                    (rewrap(t, "final_line"), final, "stop"), (t, "\\boxed{17}", "stop")])
    assert (s[0].reward, s[0].complied, s[0].channel, s[0].truncated) == (1.0, True, "direct", False)
    assert (s[1].reward, s[1].truncated, s[1].correct_any) == (0.0, True, True)
    assert (s[2].reward, s[2].complied, s[2].correct_any) == (0.0, False, True)  # right value, wrong wrapper
    assert (s[3].reward, s[3].channel) == (1.0, "program")
    assert s[4].reward == 1.0
    assert (s[5].reward, s[5].correct_any) == (0.0, False)


def test_procedural_scoring_is_strict_and_lenient():
    t = next(tasks.draws("test", "prime_factorization", 0, "math"))
    gold = t.gold["answer"]
    s = score_many([(t, f"\\boxed{{{gold}}}", "stop"), (t, f"<answer>{gold}</answer>", "stop"),
                    (t, f"\\boxed{{{gold}}}", "length"), (t, "\\boxed{2}", "stop")])
    assert (s[0].reward, s[0].correct_any) == (1.0, True)
    assert (s[1].reward, s[1].complied, s[1].correct_any) == (0.0, False, True)  # Round 1's tags: lenient only
    assert (s[2].reward, s[2].truncated) == (0.0, True)
    assert (s[3].reward, s[3].correct_any) == (0.0, False)

    def exact_item(source, gold):
        return make_task(Item(source, "0", "logic", "procedural", "...", gold, "", ""))

    dyck = exact_item("bbh_dyck_languages", {"exact": "{ ( [ ] ) }", "accept": ["] ) }"], "compare": "brackets"})
    answers = ["\\boxed{])}}", "\\boxed{{ ( [ ] ) }}", "\\boxed{\\{([])\\}}", "\\boxed{) ] }}", "\\boxed{{([)]}}"]
    assert [x.reward for x in score_many([(dyck, a, "stop") for a in answers])] == [0.0, 1.0, 1.0, 0.0, 0.0]
    assert score_many([(dyck, "\\boxed{] ) }", "stop")])[0].reward == 0.0  # its closing brace ends the box
    words = exact_item("bbh_word_sorting", {"exact": "apple banana cherry", "accept": [], "compare": "words"})
    answers = ["\\boxed{apple, banana, cherry}", "\\boxed{\\text{Apple banana cherry.}}", "\\boxed{banana apple cherry}",
               "\\boxed{apple banana}"]
    assert [x.reward for x in score_many([(words, a, "stop") for a in answers])] == [1.0, 1.0, 0.0, 0.0]
    assert tasks.gold_ok_many([t, dyck, words]) == [True, True, True]


def test_undecided_reference_checks_are_retried_and_never_verdicts(monkeypatch):
    import evotaste.verify as verify

    calls = []

    def flaky(items, timeout=None):
        calls.append(len(items))
        return [reason for _ in items]

    monkeypatch.setattr(verify, "gold_check", flaky)
    t = make_task(open_item())
    reason = "wrapper box: timeout"  # undecided: repeated, then confirmed by a second round
    assert tasks.gold_reasons([t]) == [reason] and calls == [1] * 2 * tasks.GOLD_ATTEMPTS
    reason = "wrapper box: verifier process died (0)"
    with pytest.raises(RuntimeError, match="process died"):
        tasks.gold_reasons([t])
    reason = "wrapper box: not verified"  # a rejection is confirmed once
    calls.clear()
    assert tasks.gold_reasons([t]) == [reason] and calls == [1, 1]

    def transient(items, timeout=None):  # rejected once, then verified: not rejected
        calls.append(len(items))
        return ["wrapper box: GoldError: reference does not parse" if len(calls) == 1 else None for _ in items]

    calls.clear()
    monkeypatch.setattr(verify, "gold_check", transient)
    assert tasks.gold_reasons([t]) == [None] and calls == [1, 1]


def test_stratum_rule():
    assert [tasks.stratum_of(p, 8) for p in range(9)] == ["frontier"] + ["sweet"] * 6 + ["mastered"] * 2


# ---------------------------------------------------------------------------------------------------------------
# The training stream on synthetic pools


class FakeSuites:
    """Training sources of the real table, each with synthetic train items; `sizes` per (source, stratum)."""

    def __init__(self, sizes):
        self.train = sources.TRAINING_SOURCES
        self.selection, self.procedural = (), {}
        self._items, self.strata = {}, {}
        for s in self.train:
            domain = sources.SOURCES[s].domain
            kind = {"math": "open", "code": "code", "logic": "choice"}[domain]
            items = []
            for stratum, n in sizes(s).items():
                for k in range(n):
                    gold = ({"options": ["a", "b"], "answer": "A"} if kind == "choice" else
                            {"harness": "pytest", "tests": "", "reference": "", "entry_point": "f",
                             "signature": "def f():"} if kind == "code" else "1")
                    it = Item(s, f"{stratum}-{k}", domain, kind, f"{s} {stratum} problem {k}", gold, "", "")
                    items.append(make_task(it))
                    self.strata[it.uid] = stratum
            self._items[s] = items
        self.held_out = make_task(Item("arc", "test-0", "logic", "choice", "held out",
                                       {"options": ["a", "b"], "answer": "A"}, "", ""))

    def items(self, source, part):
        assert part == "train", "the stream reads only the train part"
        return list(self._items[source])


def standard(source):
    return {"sweet": 200, "mastered": 80, "frontier": 60}


@pytest.fixture(scope="module")
def fake():
    return FakeSuites(standard)


def test_stream_is_deterministic_in_seed_and_step(fake):
    a, b = Streams(fake, 7, strata=fake.strata), Streams(fake, 7, strata=fake.strata)
    assert [t.question for t in a.train_batch(5, 64)] == [t.question for t in b.train_batch(5, 64)]
    assert [t.uid for t in b.train_batch(5, 64)] == [t.uid for t in Streams(fake, 7, strata=fake.strata)
                                                      .train_batch(5, 64)]  # without computing steps 0-4 first
    assert [t.uid for t in a.train_batch(5, 64)] != [t.uid for t in Streams(fake, 8, strata=fake.strata)
                                                      .train_batch(5, 64)]
    with pytest.raises(ValueError, match="strata"):
        Streams(fake, 7).train_batch(0, 64)


def test_stream_obeys_the_mix(fake):
    s, steps, n = Streams(fake, 3, strata=fake.strata), 30, 64
    batches = [s.train_batch(t, n) for t in range(steps)]
    assert all(len(b) == n for b in batches)
    cells = collections.Counter((t.family, t.stratum) for b in batches for t in b)
    for src in fake.train:
        domain = sources.SOURCES[src].domain
        n_src = sum(sources.SOURCES[x].domain == domain for x in fake.train)
        for stratum, share in tasks.STRATUM_MIX.items():
            target = tasks.DOMAIN_MIX[domain] / n_src * share * n * steps
            assert abs(cells[(src, stratum)] - target) <= 2, (src, stratum, cells[(src, stratum)], float(target))
    for b in batches:  # every batch follows the domain mix, not only the run as a whole
        domains = collections.Counter(t.domain for t in b)
        assert all(abs(domains[d] - share * n) < 1 for d, share in tasks.DOMAIN_MIX.items()), domains
    noncode = [t for b in batches for t in b if t.domain != "code"]
    variants = [t for t in noncode if t.wrapper != "box"]
    assert abs(len(variants) - Fraction(3, 20) * len(noncode)) <= 1
    assert {t.wrapper for t in variants} <= set(TRAIN_WRAPPERS) and len({t.wrapper for t in variants}) > 1
    assert all(t.wrapper == "code" for b in batches for t in b if t.domain == "code")
    assert all(tasks.PROGRAM_OFFER not in t.question for t in variants)


def test_stream_draws_without_replacement_until_a_pool_is_exhausted(fake):
    s = Streams(fake, 11, strata=fake.strata)
    frontier = [t.uid for step in range(80) for t in s.train_batch(step, 64)
                if t.family == "gsm8k_train" and t.stratum == "frontier"]
    assert len(frontier) > 60 and len(set(frontier[:60])) == 60 and len(set(frontier)) == 60


def test_short_pools_are_filled_from_sweet_and_held_out_items_refused():
    short = FakeSuites(lambda s: {"sweet": 200, "mastered": 10 if s == "arc" else 80, "frontier": 60})
    s = Streams(short, 5, strata=short.strata)
    drawn = [t for step in range(20) for t in s.train_batch(step, 64)]
    assert not [t for t in drawn if t.family == "arc" and t.stratum == "mastered"]
    arc = collections.Counter(t.stratum for t in drawn if t.family == "arc")
    target = tasks.DOMAIN_MIX["logic"] / 4 * 64 * 20
    assert abs(sum(arc.values()) - target) <= 2 and arc["sweet"] > arc["frontier"]
    assert {t.uid for t in drawn} <= set(short.strata)
    bad = dict(short.strata)
    bad[short.held_out.uid] = "sweet"
    with pytest.raises(ValueError, match="outside the training part"):
        Streams(short, 5, strata=bad).train_batch(0, 64)
    empty = FakeSuites(lambda s: {"sweet": 10, "mastered": 80, "frontier": 60})
    with pytest.raises(ValueError, match="sweet-spot"):
        Streams(empty, 5, strata=empty.strata).train_batch(0, 64)


def test_seed_batches(fake):
    s = Streams(fake, 9)  # seeding needs no strata
    a = s.seed_batch(3, 64)
    assert [t.uid for t in a] == [t.uid for t in Streams(fake, 9).seed_batch(3, 64)]
    assert all(t.wrapper == tasks.default_wrapper(t.kind) for t in a)
    assert abs(collections.Counter(t.domain for t in a)["code"] - 16) <= 1
    math_only = s.seed_batch(0, 64, ("math",))
    assert len(math_only) == 64 and {t.domain for t in math_only} == {"math"}
    assert len({t.uid for i in range(10) for t in s.seed_batch(i, 64, ("math",))}) == 640  # no repeats yet


# ---------------------------------------------------------------------------------------------------------------
# Probe and evaluation sets on the real suites file


def test_probe_set_and_evaluation_set_are_disjoint_and_run_independent():
    suites = tasks.Suites(SUITES)
    probe = tasks.probe_set(suites)
    assert [t.uid for t in probe] == [t.uid for t in tasks.probe_set(suites)]
    plain = [t for t in probe if t.wrapper in ("box", "code")]
    wrapped = [t for t in probe if t.wrapper not in ("box", "code")]
    assert 250 <= len(plain) <= 350 and len(wrapped) == tasks.PROBE_WRAPPED
    assert {t.wrapper for t in wrapped} == set(HELDOUT_WRAPPERS) and all(t.domain != "code" for t in wrapped)
    assert len({t.uid for t in probe}) == len(probe)
    assert {t.family for t in plain} >= set(suites.selection) | set(suites.procedural)
    evaluation = tasks.fixed_tasks(suites, "selection", 30, per_family=2)
    assert not {t.uid for t in evaluation} & {t.uid for t in probe}
    selection = {t.uid for s in suites.selection for t in suites.items(s, "selection")}
    assert {t.uid for t in probe + evaluation if t.kind != "procedural"} <= selection


def test_engine_code_never_names_the_vault():
    """Only evaluate.py (and the guarded accessors) may refer to blind vaults."""
    for module in ("worker", "member", "trainer", "sampler"):
        text = pathlib.Path(f"evotaste/{module}.py").read_text()
        assert "blind" not in text and "vault" not in text, module
