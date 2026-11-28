"""The method census's deterministic parts: the item draw (stratified, seeded, train part only), pair construction,
strict judge parsing, the construct profile of programs and the typicality statistics."""

import ast
import collections
import json
import pathlib

import numpy as np
import pytest

from evotaste import method
from evotaste.analysis import method as am
from evotaste.tasks import Suites, Task

# -- item draw -----------------------------------------------------------------------------------------------------


def _train(sources: dict[str, tuple[str, str]], per_source: int) -> dict[str, Task]:
    """uid -> task for `per_source` items of each source (source -> (domain, kind))."""
    out = {}
    for s, (domain, kind) in sources.items():
        for i in range(per_source):
            uid = f"{s}/{i}"
            out[uid] = Task(uid, s, domain, kind, "q", None, "code" if kind == "code" else "box")
    return out


SOURCES = {"m1": ("math", "open"), "m2": ("math", "choice"), "c1": ("code", "code"), "l1": ("logic", "verdict")}
TARGETS = {"math": {"mastered": 4, "sweet": 9, "frontier": 6}, "code": {"mastered": 2, "sweet": 3, "frontier": 2},
           "logic": {"mastered": 1, "sweet": 2, "frontier": 1}}


def _rows(train: dict[str, Task]) -> list[tuple[str, int, int]]:
    """m1 has twice as many items as m2 in every stratum (passes cycle 0..8, m2 only on even indices)."""
    return [(uid, int(uid.split("/")[1]) % 9, 8) for uid in train if not (uid.startswith("m2/")
                                                                           and int(uid.split("/")[1]) % 2)]


def test_item_draw_is_stratified_seeded_and_proportional():
    train = _train(SOURCES, 180)
    rows = _rows(train)
    drawn = method.draw_items(rows, train, TARGETS, 1)
    cells = collections.Counter((t.domain, method.stratum_of(p, n)) for t, p, n in drawn)
    assert cells == {(d, s): n for d, t in TARGETS.items() for s, n in t.items()}
    assert len({t.uid for t, _, _ in drawn}) == len(drawn)
    sweet = collections.Counter(t.family for t, p, n in drawn if t.domain == "math"
                                and method.stratum_of(p, n) == "sweet")
    available = collections.Counter(uid.split("/")[0] for uid, p, n in rows
                                    if uid.startswith("m") and method.stratum_of(p, n) == "sweet")
    assert abs(sweet["m1"] - 9 * available["m1"] / (available["m1"] + available["m2"])) < 1
    again = method.draw_items(rows, train, TARGETS, 1)
    assert [t.uid for t, _, _ in again] == [t.uid for t, _, _ in drawn]
    other = method.draw_items(rows, train, TARGETS, 2)
    assert {t.uid for t, _, _ in other} != {t.uid for t, _, _ in drawn}
    assert method.proportional(5, {"a": 4, "b": 2, "c": 2}) == {"a": 3, "b": 1, "c": 1}
    assert method.proportional(2, {"c": 1, "b": 1, "a": 1}) == {"a": 1, "b": 1, "c": 0}


def test_item_draw_refuses_items_outside_the_train_part_and_short_cells():
    train = _train(SOURCES, 180)
    rows = _rows(train)
    with pytest.raises(ValueError, match="outside the train part"):
        method.draw_items(rows + [("mmlu_pro/blind/3", 4, 8)], train, TARGETS, 1)
    with pytest.raises(ValueError, match="logic/frontier"):
        method.draw_items(rows, train, TARGETS | {"logic": {"mastered": 1, "sweet": 2, "frontier": 500}}, 1)
    wrapped = dict(train)
    wrapped["m1/0"] = Task("m1/0", "m1", "math", "open", "q", None, "final_line")
    with pytest.raises(ValueError, match="default wrapper"):
        method.draw_items(rows, wrapped, TARGETS, 1)


def test_real_draw_uses_train_items_only():
    """On the real census table and suites: every drawn item is a train-part task, and no selection item can be
    turned into one (the vault is never loaded at all, see the next test)."""
    cfg = method.load_config("configs/method.yaml")
    if not pathlib.Path(cfg["census"]["strata"]).exists():
        pytest.skip("the census strata table is not on this machine")
    suites = Suites(cfg["suites"])
    train = method.train_index(suites)
    rows = method.strata_rows(cfg["census"]["strata"], cfg["census"]["sha256"])
    drawn = method.draw_items(rows, train, cfg["items"]["targets"], cfg["items"]["seed"])
    assert len(drawn) == sum(sum(t.values()) for t in cfg["items"]["targets"].values())
    assert all(t.uid in train for t, _, _ in drawn)
    selection = [t.uid for s in suites.selection for t in suites.items(s, "selection")[:3]]
    with pytest.raises(RuntimeError, match="not in the train part"):
        method.train_tasks(cfg, [drawn[0][0].uid, selection[0]])


def test_method_never_names_the_vault():
    for path in ("evotaste/method.py", "evotaste/analysis/method.py"):
        text = pathlib.Path(path).read_text()
        tree = ast.parse(text)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        assert not names & {"BlindReceipt", "blind_guard", "blind_suite", "check_receipt", "VAULT", "fixed_tasks"}
        assert "blind2" not in text and '"selection"' not in text


# -- pairs ---------------------------------------------------------------------------------------------------------

ITEMS = [("u1", "math"), ("u2", "code"), ("u3", "logic"), ("u4", "math")]
CORRECT = {"s": {"u1": [0, 1, 2, 3, 4], "u2": [3], "u3": [1, 5, 6]},
           "l": {"u1": list(range(8)), "u2": [0, 2], "u4": [7]}}
ALTERNATIVES = {"s": {"u1": (0, 1), "u2": (3, None), "u3": (1, 0)}, "l": {"u1": (0, None), "u2": (0, 1), "u4": (7, 0)}}


def test_pairs_follow_the_rules():
    pairs = method.build_pairs(["s", "l"], ITEMS, CORRECT, ALTERNATIVES, 5, 2, 4)
    kinds = collections.Counter(p["kind"] for p in pairs)
    assert kinds == {"spontaneous": 6, "requested": 4, "cross": 2, "repeat": 4}
    assert len({p["id"] for p in pairs}) == len(pairs)
    by_id = {p["id"]: p for p in pairs}
    used = collections.defaultdict(set)
    for p in pairs:
        a, b = p["a"], p["b"]
        assert (a["model"], a["stage"], a["sample"]) != (b["model"], b["stage"], b["sample"])
        if p["kind"] == "spontaneous":
            assert a["model"] == b["model"] and a["stage"] == b["stage"] == "attempts"
            assert {a["sample"], b["sample"]} <= set(CORRECT[a["model"]][p["uid"]])
            key = (a["model"], p["uid"])
            assert not used[key] & {a["sample"], b["sample"]}  # the pairs of an item are disjoint
            used[key] |= {a["sample"], b["sample"]}
        elif p["kind"] == "requested":
            ref, alt = ALTERNATIVES[a["model"]][p["uid"]]
            assert {(a["stage"], a["sample"]), (b["stage"], b["sample"])} == {("attempts", ref), ("alternatives", alt)}
        elif p["kind"] == "cross":
            assert {a["model"], b["model"]} == {"s", "l"} and p["uid"] in ("u1", "u2")
            for side in (a, b):
                assert side["sample"] == CORRECT[side["model"]][p["uid"]][0]
        else:
            orig = by_id[p["of"]]
            assert orig["kind"] == "spontaneous" and (p["a"], p["b"]) == (orig["b"], orig["a"])
    assert method.build_pairs(["s", "l"], ITEMS, CORRECT, ALTERNATIVES, 5, 2, 4) == pairs


def test_pair_sides_are_shuffled():
    items = [(f"u{i}", "math") for i in range(40)]
    correct = {"s": {u: [0] for u, _ in items}, "l": {u: [1] for u, _ in items}}
    pairs = method.build_pairs(["s", "l"], items, correct, {"s": {}, "l": {}}, 5, 2, 0)
    first = collections.Counter(p["a"]["model"] for p in pairs)
    assert len(pairs) == 40 and min(first.values()) >= 10


# -- judge replies -------------------------------------------------------------------------------------------------

GOOD = '{"approach_a": "solve the linear system", "approach_b": "test each option", "relation": "different"}'


@pytest.mark.parametrize("reply", [
    GOOD.replace('"relation": "different"', '"relation": "similar"'),
    '{"approach_b": "x", "approach_a": "y", "relation": "same"}',
    '{"approach_a": "x", "approach_b": "y", "relation": "same", "note": "z"}',
    '{"approach_a": "x", "approach_b": "y"}',
    '{"approach_a": "  ", "approach_b": "y", "relation": "same"}',
    '{"approach_a": 3, "approach_b": "y", "relation": "same"}',
    "```json\n" + GOOD + "\n```",
    "The verdict: " + GOOD,
    GOOD[:-10],
    "[" + GOOD + "]",
    "",
    "</think>" + GOOD,
])
def test_malformed_replies_do_not_parse(reply):
    assert method.parse_verdict(reply, thinking=False) == (None, False)


def test_well_formed_replies_parse():
    assert method.parse_verdict(GOOD, thinking=False) == (json.loads(GOOD), False)
    assert method.parse_verdict("  " + GOOD + "\n", thinking=False)[0] == json.loads(GOOD)
    assert method.parse_verdict("Both set up equations.\n</think>" + GOOD, thinking=True) == (json.loads(GOOD), True)
    assert method.parse_verdict("\n\n</think>\n\n" + GOOD, thinking=True) == (json.loads(GOOD), False)
    assert method.parse_verdict("Still thinking about it", thinking=True) == (None, True)
    assert method.parse_verdict("a</think>b</think>" + GOOD, thinking=True) == (None, True)
    assert method.parse_verdict(GOOD, thinking=True) == (None, True)  # a thinking reply must close its reasoning


# -- construct profile ---------------------------------------------------------------------------------------------

RECURSIVE = "def f(n):\n    if n <= 1:\n        return 1\n    return n * f(n - 1)\n"
ITERATIVE = "def f(n):\n    out = 1\n    for k in range(2, n + 1):\n        out *= k\n    return out\n"
COUNTER = "from collections import Counter\ndef top(xs, k):\n    return [x for x, _ in Counter(xs).most_common(k)]\n"
MANUAL = ("def top(xs, k):\n    counts = {}\n    for x in xs:\n        counts[x] = counts.get(x, 0) + 1\n"
          "    return sorted(counts, key=counts.get, reverse=True)[:k]\n")
HEAP = "import heapq\ndef top(xs, k):\n    return heapq.nlargest(k, xs)\n"
NESTED = ("def pairs(xs, t):\n    for i in range(len(xs)):\n        for j in range(i + 1, len(xs)):\n"
          "            if xs[i] + xs[j] == t:\n                return (i, j)\n    return None\n")
WORDS = "def words(s):\n    return ' '.join(w.capitalize() for w in s.split())\n"


def test_construct_profiles():
    p = am.construct_profile
    assert p(RECURSIVE) == {"recursion", "early_return"}
    assert p(ITERATIVE) == {"for_loop", "loop_depth_1"}
    assert p(COUNTER) == {"hashing", "comprehension"}
    assert p(MANUAL) == {"hashing", "for_loop", "loop_depth_1", "sorting"}
    assert p(HEAP) == {"heap"}
    assert p(NESTED) == {"for_loop", "index_scan", "loop_depth_2", "early_return"}
    assert p(WORDS) == {"string_methods", "comprehension"}
    assert p("def f(:\n") is None and p(None) is None
    assert am.jaccard(p(RECURSIVE), p(RECURSIVE)) == 0.0
    assert am.jaccard(p(RECURSIVE), p(ITERATIVE)) == 1.0
    assert am.jaccard(frozenset(), frozenset()) == 0.0
    assert am.jaccard(p(COUNTER), p(MANUAL)) == pytest.approx(1 - 1 / 5)


# -- typicality ----------------------------------------------------------------------------------------------------


def test_typicality_statistics_on_a_synthetic_table():
    correct = np.array([[1, 1, 0, 0, 1, 1, 0, 0], [1] * 8, [0] * 8])
    e = np.array([0.1, -0.1, 0.1, -0.1, 0.2, -0.2, 0.2, -0.2])  # balanced over correctness and over length
    log_len = np.where(correct[0] == 1, 4.0, 6.0)  # correct answers shorter
    a = np.array([[-0.3], [-0.5], [-0.9]])
    x = a - 0.5 * log_len + e  # all of the correct answers' typicality comes from their length
    t = am.typicality(x, correct)
    assert list(t["mixed"]) == [True, False, False]
    assert t["d"][0] == pytest.approx(1.0 / x[0].std(ddof=1)) and np.isnan(t["d"][1:]).all()
    assert (t["n_t"] == [3, 3, 2]).all()
    assert t["c_t"][0].tolist() == [0, 2, 2] and t["c_t"][1].tolist() == [3, 3, 2]
    adjusted, slopes = am.residualize(x, np.tile(log_len, (3, 1)), np.array(["math"] * 3))
    assert slopes["math"] == pytest.approx(-0.5)
    assert am.typicality(adjusted, correct)["d"][0] == pytest.approx(0.0, abs=1e-12)
    assert am.terciles(np.arange(8.0)[None]).tolist() == [[0, 0, 0, 1, 1, 1, 2, 2]]
    assert am.kappa(["same", "same", "different"], ["same", "same", "different"]) == 1.0
    assert am.kappa(["same", "different"], ["different", "same"]) == pytest.approx(-1.0)
