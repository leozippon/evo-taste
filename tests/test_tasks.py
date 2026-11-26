"""Tasks, instructions, scoring, the training stream and the seeding prompts (synthetic pools), probe sets."""

import collections
import hashlib
import json
import pathlib
import random
from fractions import Fraction

import pytest

from evotaste import sources, tasks
from evotaste.sources import Item
from evotaste.tasks import Streams, make_task, rewrap, score_many
from evotaste.verify import HELDOUT_WRAPPERS, TRAIN_WRAPPERS

SUITES = "configs/suites-r2.yaml"
NOW = tasks.STREAM_VERSION  # the stream of a run prepared now


def open_item(source="gsm8k_train", k=0, gold="18"):
    return Item(source, str(k), "math", "open", "How many eggs are left?", gold, "How many eggs are left?",
                "How many eggs are left?")


def test_instructions():
    from evotaste.verify import WRAPPERS

    t = make_task(open_item())
    assert t.wrapper == "box" and t.question == f"{t.problem}\n\n{WRAPPERS['box'].instruction}"
    assert "Python" not in t.question  # no program route is offered
    v = rewrap(t, "final_line")
    assert v.wrapper == "final_line" and "Final answer:" in v.question and "\\boxed" not in v.question
    choice = make_task(sources._choice("aqua", "1", "math", "Which?", ["2", "3"], 1))
    assert "letter of the correct option" in choice.question
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


def test_scores_follow_the_wrapper_and_truncation():
    t = make_task(open_item())
    boxed, final = "So 18 eggs.\n\\boxed{18}", "So 18 eggs.\nFinal answer: 18"
    program = "Compute it.\n```python\nprint(20 - 2)\n```"
    s = score_many([(t, boxed, "stop"), (t, boxed, "length"), (t, final, "stop"), (t, program, "stop"),
                    (rewrap(t, "final_line"), final, "stop"), (t, "\\boxed{17}", "stop"),
                    (t, boxed + "\n```python\nprint(17)\n```", "stop")])
    assert (s[0].reward, s[0].complied, s[0].channel, s[0].truncated) == (1.0, True, "direct", False)
    assert (s[1].reward, s[1].truncated, s[1].correct_any) == (0.0, True, True)
    assert (s[2].reward, s[2].complied, s[2].correct_any) == (0.0, False, True)  # right value, wrong wrapper
    assert (s[3].reward, s[3].complied) == (0.0, False)  # a program is no answer: the route is not offered
    assert s[4].reward == 1.0
    assert (s[5].reward, s[5].correct_any) == (0.0, False)
    assert (s[6].reward, s[6].channel) == (1.0, "direct")  # a program after the box is ignored


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
    reason = "wrapper box: undecided: verifier deadline passed"  # undecided: repeated, then confirmed by a second round
    assert tasks.gold_reasons([t]) == [reason] and calls == [1] * 2 * tasks.GOLD_ATTEMPTS
    reason = "wrapper box: undecided: verifier process died (0)"
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
    a, b = Streams(fake, 7, strata=fake.strata, version=NOW), Streams(fake, 7, strata=fake.strata, version=NOW)
    assert [t.question for t in a.train_batch(5, 64)] == [t.question for t in b.train_batch(5, 64)]
    assert [t.uid for t in b.train_batch(5, 64)] == [t.uid for t in Streams(fake, 7, strata=fake.strata, version=NOW)
                                                      .train_batch(5, 64)]  # without computing steps 0-4 first
    assert [t.uid for t in a.train_batch(5, 64)] != [t.uid for t in Streams(fake, 8, strata=fake.strata, version=NOW)
                                                      .train_batch(5, 64)]
    with pytest.raises(ValueError, match="strata"):
        Streams(fake, 7, version=NOW).train_batch(0, 64)


def test_stream_obeys_the_mix(fake):
    s, steps, n = Streams(fake, 3, strata=fake.strata, version=NOW), 30, 64
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
    assert all("\\boxed" not in t.question for t in variants)


def test_stream_draws_without_replacement_until_a_pool_is_exhausted(fake):
    s = Streams(fake, 11, strata=fake.strata, version=NOW)
    frontier = [t.uid for step in range(80) for t in s.train_batch(step, 64)
                if t.family == "gsm8k_train" and t.stratum == "frontier"]
    assert len(frontier) > 60 and len(set(frontier[:60])) == 60 and len(set(frontier)) == 60


def batches_digest(s, n, steps):
    """SHA-256 of a stream's first `steps` batches of n prompts: every prompt's uid, wrapper and question."""
    return hashlib.sha256(json.dumps([[(t.uid, t.wrapper, t.question) for t in s.train_batch(step, n)]
                                      for step in range(steps)]).encode()).hexdigest()


# Computed with the stream code before stream versions existed (commit 85c3417) on FakeSuites(standard), steps 0-59:
# run seed 4 at 48 prompts (a count of -1 at steps 13, 32, 48 and 51) and run seed 0 at 32 prompts (many).
BEFORE_VERSIONS = {(4, 48): "c6d410f5c40fd80c2c7d57723aef817ea8fb1b79dd80d34efc2987969c86f256",
                   (0, 32): "e96163f8a0b2682058540507cfc8fc11003954c789c31003ddc1af4002d2e3a2"}


def repeats_before_exhaustion(batches, mix) -> bool:
    """Whether some pool gives an item twice in the batches before the one in which it is exhausted."""
    size = {tuple(name.split("/")): len(pool) for name, pool in zip(mix.names, mix.pools)}
    seen, exhausted = collections.defaultdict(list), set()
    for b in batches:
        drawn = collections.defaultdict(list)
        for t in b:
            drawn[(t.family, t.stratum)].append(t.uid)
        for cell, uids in drawn.items():
            if len(seen[cell]) + len(uids) > size[cell]:
                exhausted.add(cell)
            if cell not in exhausted:
                seen[cell] += uids
    return any(len(set(u)) < len(u) for u in seen.values())


@pytest.mark.parametrize("seed, n", sorted(BEFORE_VERSIONS))
def test_stream_version_1_is_the_stream_before_versions_and_version_2_never_overfills(fake, seed, n):
    v1, v2 = (Streams(fake, seed, strata=fake.strata, version=v) for v in (1, 2))
    assert batches_digest(v1, n, 60) == BEFORE_VERSIONS[(seed, n)]  # runs prepared before keep their batches
    old, new = [v1.train_batch(t, n) for t in range(60)], [v2.train_batch(t, n) for t in range(60)]
    overfull = [t for t, b in enumerate(old) if len(b) > n]
    assert overfull and all(len(b) == n for b in new)
    assert [[t.uid for t in b] for b in new[:overfull[0]]] == [[t.uid for t in b] for b in old[:overfull[0]]]
    mix = v2._train_mix()
    assert repeats_before_exhaustion(old, mix) and not repeats_before_exhaustion(new, mix)  # after a count of -1
    counts = mix.counts(59, n)
    assert min(min(h) for h in counts) >= 0
    for c, w in enumerate(mix.weights):  # every pool still receives its share of the stream
        assert all(abs(sum(h[c] for h in counts[:t + 1]) - w * n * (t + 1)) <= 2 for t in range(60))


def test_a_nonnegative_split_never_gives_a_negative_count():
    shares, carry = [Fraction(1, 20), Fraction(1, 20), Fraction(9, 10)], [Fraction(-9, 10), Fraction(-9, 10),
                                                                          Fraction(9, 5)]
    assert tasks._apportion(shares, 2, list(carry), random.Random(0), False) == [-1, -1, 4]  # quotas -.8, -.8, 3.6
    assert tasks._apportion(shares, 2, carry, random.Random(0), True) == [0, 0, 2]  # rounded down where it can be
    assert carry == [Fraction(-4, 5), Fraction(-4, 5), Fraction(8, 5)]
    rng = random.Random(1)
    for _ in range(50):  # random shares and batch sizes: counts >= 0 summing to n, bounded carries
        raw = [rng.random() ** 3 for _ in range(rng.randint(2, 9))]
        shares = [Fraction(x).limit_denominator(10**6) for x in raw]
        shares = [s / sum(shares) for s in shares]
        carry = [Fraction(0)] * len(shares)
        for _ in range(40):
            n = rng.randint(1, 30)
            counts = tasks._apportion(shares, n, carry, rng, True)
            assert min(counts) >= 0 and sum(counts) == n and all(-1 < c < 2 for c in carry) and sum(carry) == 0


def test_short_pools_are_filled_from_sweet_and_held_out_items_refused():
    short = FakeSuites(lambda s: {"sweet": 200, "mastered": 10 if s == "arc" else 80, "frontier": 60})
    s = Streams(short, 5, strata=short.strata, version=NOW)
    drawn = [t for step in range(20) for t in s.train_batch(step, 64)]
    assert not [t for t in drawn if t.family == "arc" and t.stratum == "mastered"]
    arc = collections.Counter(t.stratum for t in drawn if t.family == "arc")
    target = tasks.DOMAIN_MIX["logic"] / 4 * 64 * 20
    assert abs(sum(arc.values()) - target) <= 2 and arc["sweet"] > arc["frontier"]
    assert {t.uid for t in drawn} <= set(short.strata)
    bad = dict(short.strata)
    bad[short.held_out.uid] = "sweet"
    with pytest.raises(ValueError, match="outside the training part"):
        Streams(short, 5, strata=bad, version=NOW).train_batch(0, 64)
    empty = FakeSuites(lambda s: {"sweet": 10, "mastered": 80, "frontier": 60})
    with pytest.raises(ValueError, match="sweet-spot"):
        Streams(empty, 5, strata=empty.strata, version=NOW).train_batch(0, 64)


def member_batches(s, shares, steps, n=48):
    """A practising member's batches over `steps` steps, its draws carried from step to step, as the engine does."""
    drawn, batches = {}, []
    for t in range(steps):
        batches.append(s.member_batch(t, n, shares, drawn))
        for x in batches[-1]:
            drawn[tasks.train_cell(x)] = drawn.get(tasks.train_cell(x), 0) + 1
    return batches


def skewed(s):
    """kodcode at 0.6, gsm8k_train at 0.2, the other seven sources sharing 0.2."""
    rest = [x for x in s.source_shares if x not in ("kodcode", "gsm8k_train")]
    return {"kodcode": 0.6, "gsm8k_train": 0.2, **{x: 0.2 / len(rest) for x in rest}}


def test_member_batch_follows_its_shares(fake):
    s, steps, n = Streams(fake, 3, strata=fake.strata, version=NOW), 40, 48
    assert s.source_shares == pytest.approx({x: float(tasks.DOMAIN_MIX[sources.SOURCES[x].domain]) / sum(
        sources.SOURCES[y].domain == sources.SOURCES[x].domain for y in fake.train) for x in fake.train})
    shares = skewed(s)
    batches = member_batches(s, shares, steps, n)
    for b in batches:
        assert len(b) == n and {t.uid for t in b} <= set(fake.strata)  # train items only: never a held-out one
        per_source = collections.Counter(t.family for t in b)
        for x, share in shares.items():  # every step, every source: the floor or the ceiling of its expectation
            assert int(n * share) <= per_source[x] <= int(n * share) + 1, (x, per_source[x], n * share)
        per_cell = collections.Counter((t.family, t.stratum) for t in b)
        for x in fake.train:
            for st, w in tasks.STRATUM_MIX.items():
                q = n * shares[x] * float(w)
                assert int(q) <= per_cell[(x, st)] <= int(q) + 1
        noncode = [t for t in b if t.domain != "code"]
        variants = [t for t in noncode if t.wrapper != "box"]
        assert int(len(noncode) * 0.15) <= len(variants) <= int(len(noncode) * 0.15) + 1
        assert {t.wrapper for t in variants} <= set(TRAIN_WRAPPERS) and all(t.wrapper == "code" for t in b
                                                                             if t.domain == "code")
    cells = collections.Counter((t.family, t.stratum) for b in batches for t in b)
    for x in fake.train:  # on average each pool gets its expected count (strata in the stream's proportions)
        for st, w in tasks.STRATUM_MIX.items():
            assert abs(cells[(x, st)] / steps - n * shares[x] * float(w)) < 0.25, (x, st)
    variants = sum(t.wrapper != "box" for b in batches for t in b if t.domain != "code")
    assert abs(variants / sum(t.domain != "code" for b in batches for t in b) - 0.15) < 0.02


def test_member_batch_is_a_pure_function_of_step_shares_and_draws(fake):
    s, shares = Streams(fake, 3, strata=fake.strata, version=NOW), skewed(Streams(fake, 3, strata=fake.strata, version=NOW))
    drawn = {"kodcode/sweet": 17, "arc/frontier": 3}
    first = s.member_batch(7, 48, shares, drawn)
    again = Streams(fake, 3, strata=fake.strata, version=NOW).member_batch(7, 48, dict(shares), dict(drawn))  # another member
    assert [(t.uid, t.question) for t in first] == [(t.uid, t.question) for t in again]
    assert [t.uid for t in s.member_batch(8, 48, shares, drawn)] != [t.uid for t in first]
    assert [t.uid for t in s.member_batch(7, 48, shares, {})] != [t.uid for t in first]  # continues its own walk
    assert [t.uid for t in Streams(fake, 4, strata=fake.strata, version=NOW).member_batch(7, 48, shares, drawn)] != [
        t.uid for t in first]
    for bad, match in (({**shares, "kodcode": 0.5}, "summing to 1"), ({"kodcode": 1.0}, "every training source")):
        with pytest.raises(ValueError, match=match):
            s.member_batch(0, 48, bad, {})
    with pytest.raises(ValueError, match="unknown training cells"):
        s.member_batch(0, 48, shares, {"kodcode/easy": 1})


def test_member_batch_draws_without_replacement_until_a_pool_is_exhausted(fake):
    s = Streams(fake, 11, strata=fake.strata, version=NOW)
    frontier = [t.uid for b in member_batches(s, skewed(s), 60) for t in b
                if t.family == "gsm8k_train" and t.stratum == "frontier"]  # about 1.44 per step from a pool of 60
    assert len(frontier) > 60 and len(set(frontier[:60])) == 60 and len(set(frontier)) == 60


def test_seed_batches(fake):
    s = Streams(fake, 9, version=NOW)  # seeding needs no strata
    a = s.seed_batch(3, 64)
    assert [t.uid for t in a] == [t.uid for t in Streams(fake, 9, version=NOW).seed_batch(3, 64)]
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
    from evotaste import census  # an evaluation within the census size uses censused items only

    cfg = census.load_config("configs/census.yaml")
    censused = {t.uid for t in census.selection_items(cfg, suites, None)}
    assert {t.uid for t in evaluation} <= censused
    assert {t.uid for t in probe if t.kind != "procedural"} <= censused


def test_engine_code_never_names_the_vault():
    """Only evaluate.py (and the guarded accessors) may refer to blind vaults."""
    for module in ("worker", "member", "trainer", "sampler"):
        text = pathlib.Path(f"evotaste/{module}.py").read_text()
        assert "blind" not in text and "vault" not in text, module
