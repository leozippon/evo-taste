"""Parts of the Round 2 dataset items: Round 1's frozen assignment, one part per item, de-duplication precedence, the
unit draw, and the vault's guard. Failure messages report counts only, never blind items."""

import collections
import gzip
import json
import shutil
import subprocess
import types

import numpy as np
import pytest

from evotaste import external, sources
from evotaste.sources import Item
from evotaste.tasks import BlindReceipt, Suites, blind_suite, fixed_tasks

SUITES = "configs/suites-r2.yaml"


def item(source, k, text, part_hint=None, kind="open", held_out=False):
    s = sources.SOURCES[source]
    return Item(source, str(k), s.domain, kind, text, "1", text, text, held_out)


LONG = ("A train leaves the station at nine in the morning travelling at sixty miles per hour while a second train "
        "leaves two hours later from the same station on a parallel track travelling at {} miles per hour. At what "
        "time does the second train catch up with the first one, and how far from the station are they then?")
GARDEN = ("In a rectangular garden that is {} metres long and twelve metres wide, a gardener plants roses along the "
          "whole border, one rose every half metre, and tulips in a square bed of side four metres in the middle. "
          "How many roses does she plant, and what area of the garden is left for the lawn?")
TANK = ("A water tank in the shape of a cylinder with radius {} metres is filled by two pipes, the first delivering "
        "forty litres per minute and the second sixty litres per minute, while a leak at the bottom loses ten litres "
        "per minute. How long does it take to fill the empty tank to a depth of three metres?")


def test_dedup_precedence():
    """blind2 over selection over train; within a part the smaller source keeps its copy; near copies within one
    source and part are distinct items; omni_rest loses its copies to DeepScaleR whatever the parts."""
    items = [
        item("math_train", 0, LONG.format(80)),  # 0 train: exact copy of 1, which is in blind2
        item("math_test", 1, LONG.format(80), held_out=True),  # 1 blind2
        item("theoremqa", 2, "Rewritten. " + GARDEN.format(25)),  # 2 selection: near copy of 3 (blind2)
        item("svamp", 3, GARDEN.format(25)),  # 3 blind2
        item("mmlu_pro", 4, "unrelated question about the boiling point of water at altitude, with five options"),
        item("theoremqa", 5, "unrelated question about the boiling point of water at altitude, with five options"),
        item("folio", 6, "Premises: all cats are mammals and some mammals swim in the river every summer. "
                         "Conclusion: some cats swim in the river every summer."),
        item("folio", 7, "Premises: all cats are mammals and some mammals swim in the river every summer. "
                         "Conclusion: no cat swims in the river every summer."),
        item("omni_rest", 8, TANK.format(2)),  # selection, near copy of a training DeepScaleR item
        item("deepscaler", 9, TANK.format(2) + " Give the time in hours."),
    ]
    items += [item("mmlu_pro", 100 + k, f"filler question number {k} about history and geography") for k in range(5)]
    parts = ["train", "blind2", "selection", "blind2", "selection", "selection", "blind2", "blind2", "selection",
             "train"] + ["train"] * 5
    groups, near = external.copies(items)
    removed, counts = external.deduplicate(items, parts, groups, near)
    assert removed[0] == "copy in blind2" and removed[1] is None
    assert removed[2] == "copy in blind2" and removed[3] is None
    assert removed[4] == "copy in the same part" and removed[5] is None  # theoremqa (2 items) < mmlu_pro (6)
    assert removed[6] is None and removed[7] is None  # same source and part: distinct conclusions
    assert removed[8] == "in DeepScaleR" and removed[9] is None
    assert all(r is None for r in removed[10:])
    assert sum(counts.values()) == 4


def test_near_copy_recall_matches_the_banding():
    """MinHash LSH on random sets of known Jaccard similarity finds a pair as often as 1 - (1 - J^3)^32 predicts
    (the audit measured 0.66 at 0.5 with the former linear hash family)."""
    rng = np.random.default_rng(0)
    for jac, least in ((0.5, 0.95), (0.6, 0.99), (0.8, 1.0)):
        m = 60
        c = round(2 * jac * m / (1 + jac))
        found = 0
        for _ in range(400):
            u = rng.choice(2**62, 2 * m - c, replace=False).astype(np.uint64)
            found += bool(external.near_pairs([np.unique(u[:m]), np.unique(u[m - c:])], ["x", "y"], threshold=0.0))
        assert found / 400 >= least, (jac, found / 400)


def test_reformatted_copies_are_found():
    a = r"Compute $\left( x^{3} \right) + \dfrac{1}{2}$ when $x = 2$, and give the answer as a fraction."
    b = r"Compute \( (x^3) + \frac12 \) when \(x=2\), and give the answer as a fraction"
    c = r"Compute $\left( x^{3} \right) + \dfrac{1}{2}$ when $x = 3$, and give the answer as a fraction."
    items = [item("math_test", 0, a, held_out=True), item("deepscaler", 1, b), item("deepscaler", 2, c),
             item("bbh_dyck_languages", 3, "Close it. Input: ( ["), item("bbh_dyck_languages", 4, "Close it. Input: [ (")]
    groups, pairs = external.copies(items)
    assert (0, 1) in pairs and (0, 2) in pairs  # other delimiters; near-identical with another number
    assert (3, 4) not in pairs and not groups  # within one unseen unit, brackets-only differences are distinct


def test_copy_rule_separates_copies_from_siblings():
    long = ("A student measures the temperature of a cup of water every minute while it cools on a table in a room "
            "kept at {} degrees. Which graph best shows how the temperature of the water changes over time?")
    reworded = ("Find the number of ordered triples (a, b, c) of integers with 1 <= a, b, c <= 100 and "
                "a^2 b + b^2 c + c^2 a = a b^2 + b c^2 + c a^2.", "Find the number of ordered triples of integers "
                "$(a, b, c)$ with $1 \\leq a, b, c \\leq 100$ and $a^{2} b+b^{2} c+c^{2} a=a b^{2}+b c^{2}+c a^{2}$.")
    cases = [(long.format(20), long.format(25), True),  # near-identical: a copy whatever the numbers
             ("What is the largest prime factor of 78?", "What is the largest prime factor of 999?", False),  # sibling
             (*reworded, True)]
    for x, y, want in cases:
        a, b = item("math_test", 0, x, held_out=True), item("deepscaler", 1, y)
        assert external.is_copy(a, b, external.shingles(x), external.shingles(y)) is want, (x, y)


def test_exposed_reads_hashes_only(tmp_path):
    import hashlib

    path = tmp_path / "exposure.yaml"
    path.write_text(f"sha256:\n- {hashlib.sha256(b'svamp/x').hexdigest()}\n")
    assert external.exposed(["svamp/x", "svamp/y"], path) == {"svamp/x"}
    assert "svamp/x" not in path.read_text()


def test_copied_items_of_sources_without_a_test_split_stay_in_training():
    text = next(LONG.format(n) for n in range(1000)
                if external._hash2(external.normalized(LONG.format(n))) % external.HELD_OUT_SHARE == 0)
    alone = item("deepscaler", 0, text)
    assert external.provisional_part(alone, {}) == "selection"
    assert external.provisional_part(alone, {}, copied=True) == "train"
    items = [alone, item("math_train", 1, text)]
    groups, near = external.copies(items)
    assert external.copied_elsewhere(items, groups, near) == {0, 1}


def test_unit_draw_is_seeded_balanced_and_whole():
    domains = {s.unit: s.domain for s in sources.SOURCES.values() if s.role == "unseen"}
    draw = external.draw_units(domains)
    assert draw == external.draw_units(domains) and set(draw) == set(domains)
    for d in set(domains.values()):
        sides = collections.Counter(draw[u] for u in domains if domains[u] == d)
        assert abs(sides["selection"] - sides["blind2"]) <= 1
    assert external.draw_units(domains, seed=1) != draw


@pytest.fixture(scope="module")
def built():
    f = external.SuitesFile(SUITES)
    return external.build(f.units, f.tokenizer)


def test_round1_parts_are_frozen(built):
    """Round 1's own code (tag round-1) and the Round 2 assignment agree on every item of GSM8K, MATH-500 and
    MMLU-Pro: foundation -> train, blind1 -> selection, blind2 -> blind2."""
    code = subprocess.run(["git", "show", "round-1:evotaste/external.py"], capture_output=True, text=True,
                          check=True).stdout
    r1 = types.ModuleType("round1_external")
    exec(compile(code, "round1_external", "exec"), r1.__dict__)
    items, parts, _, _ = built
    part_of = {it.uid: p for it, p in zip(items, parts)}
    by_text = {external.normalized(it.problem): p for it, p in zip(items, parts) if it.source == "math_test"}
    mismatches = collections.Counter()
    seen = collections.Counter()
    for source in r1.PARTS:
        for item_id, row in r1._rows(source):
            want = external.ROUND1_TO_ROUND2[r1.part_of(source, r1.question_text(row))]
            got = by_text.get(external.normalized(row["problem"])) if source == "math500" else part_of.get(
                f"{source}/{item_id}")
            seen[source] += 1
            mismatches[source] += got != want
    assert seen == {"gsm8k_train": 7473, "gsm8k_test": 1319, "math500": 500, "mmlu_pro": 12027}
    assert sum(mismatches.values()) == 0, f"items whose part differs from Round 1's: {dict(mismatches)}"


def test_every_item_in_exactly_one_part(built):
    items, parts, reasons, _ = built
    uids = [it.uid for it in items]
    assert len(set(uids)) == len(uids)
    assert set(parts) == set(external.PARTS)
    kept = [(it, p) for it, p, r in zip(items, parts, reasons) if r is None]
    by_text = collections.defaultdict(set)
    for it, p in kept:
        by_text[external.normalized(it.problem)].add(p)
    crossing = sum(len(ps) > 1 for ps in by_text.values())
    assert crossing == 0, f"{crossing} problem texts appear in more than one part"


def test_the_parts_cache_holds_no_vault_item_and_is_not_used_when_stale(built, tmp_path, monkeypatch):
    monkeypatch.setattr(external, "CACHE_DIR", tmp_path / "cache")
    suites = tmp_path / "suites.yaml"
    shutil.copyfile(SUITES, suites)
    external._open_parts.cache_clear()
    monkeypatch.setattr(external, "build", lambda *a: built)  # the module fixture's build, not a second one
    parts = external._open_parts(str(suites))  # built, checked against the file's hashes, then cached
    files = list((tmp_path / "cache").iterdir())
    assert [f.name for f in files] == [f"{external.cache_key(suites)}.json.gz"]
    data = json.loads(gzip.decompress(files[0].read_bytes()))
    items, provisional, _, _ = built
    vault = {it.uid for it, p in zip(items, provisional) if p == external.VAULT}
    cached = {d["source"] + "/" + d["id"] for e in data["parts"] for d in e["items"]}
    assert vault and not cached & vault and {e["part"] for e in data["parts"]} == {"train", "selection"}
    with pytest.raises(RuntimeError, match="vault"):
        external._write_cache("x", {("svamp", external.VAULT): ()})

    monkeypatch.setattr(external, "build", lambda *a: pytest.fail("rebuilt"))
    external._open_parts.cache_clear()
    assert external._open_parts(str(suites)) == parts  # a matching cache is read, nothing is built
    suites.write_text(suites.read_text() + "# another suites file\n")
    external._open_parts.cache_clear()
    with pytest.raises(pytest.fail.Exception, match="rebuilt"):  # a stale cache is not read
        external._open_parts(str(suites))
    key = external.cache_key(suites)
    monkeypatch.setattr(external, "BUILD_CODE", external.BUILD_CODE + ("census.py",))
    assert external.cache_key(suites) != key  # the key covers the building code
    external._open_parts.cache_clear()


def test_open_parts_match_the_suites_file_and_hold_no_vault_items():
    parts = external._open_parts(SUITES)  # checks every open part against the file's hash
    assert parts and all(part != external.VAULT for _, part in parts)
    f = external.SuitesFile(SUITES)
    assert {k for k in f.hashes if k[1] != external.VAULT} == set(parts)
    train = {it.uid for (s, p), v in parts.items() if p == "train" for it in v}
    selection = {it.uid for (s, p), v in parts.items() if p == "selection" for it in v}
    assert train and selection and not train & selection


def test_the_vault_needs_the_receipt():
    f = external.SuitesFile(SUITES)
    vault_sources = sorted({s for s, p in f.hashes if p == external.VAULT})
    assert vault_sources
    for source in vault_sources:
        for receipt in (None, BlindReceipt("blind1", "")):
            with pytest.raises(PermissionError):
                external.items(source, external.VAULT, receipt, suites=SUITES)
    with pytest.raises(PermissionError):
        blind_suite(SUITES, external.VAULT, None)
    with pytest.raises(PermissionError):
        fixed_tasks(Suites(SUITES), external.VAULT, 5)
    suites = Suites(SUITES)
    assert set(suites.selection) == {s for s, p in f.hashes if p == "selection"}
    assert set(suites.train) == {s for s, p in f.hashes if p == "train"}
