"""Round 2 dataset items behind the task interface: parts, de-duplication, filters and guarded access.

Every item of every source belongs to exactly one part: `train`, `selection` or the vault `blind2`.

1. Provisional part. Round 1's assignment is frozen for its three datasets (GSM8K train -> train; GSM8K test and
   MATH-500 blind1 -> selection, blind2 -> blind2; MMLU-Pro foundation -> train, blind1 -> selection, blind2 ->
   blind2), computed by Round 1's own function. Other held-out items of training sources (official test splits) are
   halved between selection and blind2 by a seeded hash of their text; training sources without a test split give 5%
   to each. An unseen unit goes whole to the part its seeded draw gave it (configs/suites-r2.yaml).
2. De-duplication over all sources: exact copies (the same problem up to case and whitespace), and copies under
   `is_copy` (the same problem in letters and digits; the same numbers with character 8-gram Jaccard >= 0.5 or edit
   similarity >= 0.9 of the letters-and-digits stems; or edit similarity >= 0.97 whatever the numbers). The latter
   count only between different sources or different parts; within one
   source and part, items sharing a context (ProofWriter theories, FOLIO premises) are distinct items. Of two copies
   the one in the higher part survives (blind2 over selection over train), within a part the one of the smaller
   source. Omni-MATH items with a copy in DeepScaleR are removed whatever their parts.
3. Filters: no gold, free-text gold (an open gold with \\text or \\mbox, a bare word, or longer than 40 characters),
   a prompt over 3,500 tokens, and a gold its own verifier rejects (`gold_ok`). Gold verdicts of train and selection items come
   from the suites file, where the generator recorded them; blind items are checked when the vault is opened.

The suites file pins a SHA-256 of every part's uid list; loading recomputes the parts (or reads the train and
selection parts from a disk cache written for the same suites file and code) and refuses a mismatch. Only the
requested part is ever turned into tasks, and blind items only with the guard's receipt. Blind items and their ids
are never written anywhere.
"""

from __future__ import annotations

import collections
import dataclasses
import difflib
import functools
import gzip
import hashlib
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import yaml

from evotaste import sources
from evotaste.tasks import BlindReceipt, Task, check_receipt, make_task, stable_seed

PARTS = ("train", "selection", "blind2")
VAULT = "blind2"
PART_RANK = {"blind2": 0, "selection": 1, "train": 2}  # lower survives a duplicate
SUITES = Path("configs/suites-r2.yaml")
# Built open parts, one file per (suites file, building code), outside git; safe to delete. The key covers the code
# below the package root that decides which items an open part holds and how they are worded, not the dataset cache.
CACHE_DIR = Path("results/cache/parts")
BUILD_CODE = ("sources.py", "external.py", "tasks.py", "verify/wrappers.py")
_PACKAGE = Path(__file__).parent
SPLIT_SEED_R2 = 20261125
HELD_OUT_SHARE = 20  # training sources without a test split: 1/20 to selection, 1/20 to blind2
MAX_GOLD_CHARS = 40
MAX_PROMPT_TOKENS = 3500
SHINGLE_CHARS = 8
NEAR_JACCARD = 0.5  # with the same numbers
NEAR_EDIT = 0.9  # with the same numbers
SAME_EDIT = 0.97  # whatever the numbers
CANDIDATE_JACCARD = 0.4  # pairs below are not compared further
# 32 bands of 3 rows find a pair at Jaccard 0.4 with probability 1 - (1 - 0.4^3)^32 = 0.88, at 0.5 with 0.986, at 0.6
# with 0.999 (tests/test_external.py measures it).
MINHASH_PERMS, MINHASH_BANDS = 96, 32
STEM_MIN_CHARS = 80  # a shorter choice stem is generic: its whole problem, options included, is compared


# ---------------------------------------------------------------------------------------------------------------
# Round 1's part assignment, frozen (identical to evotaste/external.py at the tag round-1).

SPLIT_SEED = 20261122
ROUND1_PARTS = {"gsm8k_train": ("foundation",), "gsm8k_test": ("blind1", "blind2"), "math500": ("blind1", "blind2"),
                "mmlu_pro": ("foundation", "blind1", "blind2")}
ROUND1_TO_ROUND2 = {"foundation": "train", "blind1": "selection", "blind2": "blind2"}


def _hash(key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SPLIT_SEED}/{key}".encode()).digest()[:8], "big")


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def round1_part(source: str, question: str) -> str:
    """Round 1's part of an item of gsm8k_train, gsm8k_test, math500 or mmlu_pro (a function of its text only)."""
    h = _hash(normalized(question))
    vault = ("blind1", "blind2")[h % 2]
    if ROUND1_PARTS[source] == ("foundation",):
        return "foundation"
    if "foundation" in ROUND1_PARTS[source] and (h // 2) % 3 == 0:
        return "foundation"
    return vault


# ---------------------------------------------------------------------------------------------------------------
# Provisional parts


def _hash2(key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SPLIT_SEED_R2}/{key}".encode()).digest()[:8], "big")


@functools.lru_cache(maxsize=1)
def _math500() -> frozenset[str]:
    return frozenset(normalized(p) for p in sources.math500_problems())


def provisional_part(item: sources.Item, units: dict[str, str], copied: bool = False) -> str:
    """The part an item gets before de-duplication; `copied`: the item has a copy in another source, which keeps an
    item of a source without a test split in training (its held-out shares are drawn from its own problems only, so
    that DeepScaleR's copies of MATH-train problems do not pull those out of training)."""
    source = sources.SOURCES[item.source]
    if source.role == "unseen":
        return units[source.unit]
    if item.source in ("gsm8k_train", "gsm8k_test", "mmlu_pro"):
        return ROUND1_TO_ROUND2[round1_part(item.source, item.group)]
    if item.source == "math_test" and normalized(item.group) in _math500():
        return ROUND1_TO_ROUND2[round1_part("math500", item.group)]
    h = _hash2(normalized(item.group))
    if item.held_out:
        return ("selection", "blind2")[h % 2]
    if not source.test_split and not copied:
        return {0: "selection", 1: "blind2"}.get(h % HELD_OUT_SHARE, "train")
    return "train"


def draw_units(unit_domains: dict[str, str], seed: int = SPLIT_SEED_R2) -> dict[str, str]:
    """Unseen unit -> selection or blind2: within each domain a seeded shuffle, then alternating parts from a seeded
    start, so that the two halves of a domain differ by at most one unit."""
    out = {}
    for domain in sorted(set(unit_domains.values())):
        units = sorted(u for u, d in unit_domains.items() if d == domain)
        rng = random.Random(stable_seed(seed, "units", domain))
        rng.shuffle(units)
        first = rng.randrange(2)
        for k, u in enumerate(units):
            out[u] = ("selection", "blind2")[(k + first) % 2]
    return out


# ---------------------------------------------------------------------------------------------------------------
# De-duplication

_LATEX_NOISE = re.compile(r"\\left|\\right|\\displaystyle|\\[dt](?=frac)")
_NOT_ALNUM = re.compile(r"[^a-z0-9]")


def letters_digits(text: str) -> str:
    """Lower-case letters and digits only, after removing LaTeX sizing (`\\left`, `\\dfrac` -> `frac`): the same
    problem in other delimiters, spacing, punctuation or option formatting gives the same string."""
    return _NOT_ALNUM.sub("", _LATEX_NOISE.sub("", text.lower()))


def shingles(text: str) -> np.ndarray:
    """Sorted unique character 8-grams of the letters-and-digits string, each packed exactly into a 64-bit integer
    (a shorter string is one gram)."""
    s = letters_digits(text).encode()
    if len(s) < SHINGLE_CHARS:
        return np.array([int.from_bytes(s.ljust(SHINGLE_CHARS, b"\0"), "little")], dtype=np.uint64) if s else \
            np.zeros(0, dtype=np.uint64)
    windows = np.lib.stride_tricks.sliding_window_view(np.frombuffer(s, dtype=np.uint8), SHINGLE_CHARS)
    return np.unique(np.ascontiguousarray(windows).view(np.uint64).ravel())


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if not len(a) or not len(b):
        return 0.0
    inter = len(np.intersect1d(a, b, assume_unique=True))
    return inter / (len(a) + len(b) - inter)


def _mix(z: np.ndarray) -> np.ndarray:
    """splitmix64's finaliser on uint64 arrays (wrapping arithmetic): a well-mixing hash, so that the MinHash
    functions are independent permutations in effect."""
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def near_pairs(stems: list[np.ndarray], keys: list, threshold: float = NEAR_JACCARD) -> set[tuple[int, int]]:
    """Index pairs (i < j) whose shingle sets have Jaccard similarity >= threshold, among pairs with different keys
    (MinHash LSH for candidates, exact Jaccard to confirm)."""
    seeds = np.random.default_rng(SPLIT_SEED_R2).integers(0, 2**63, MINHASH_PERMS, dtype=np.uint64)
    rows = MINHASH_PERMS // MINHASH_BANDS
    buckets: dict[tuple, list[int]] = collections.defaultdict(list)
    for i, s in enumerate(stems):
        if not len(s):
            continue
        with np.errstate(over="ignore"):
            sig = _mix(s[:, None] ^ seeds[None, :]).min(axis=0)
        for band in range(MINHASH_BANDS):
            buckets[(band, sig[band * rows:(band + 1) * rows].tobytes())].append(i)
    pairs: set[tuple[int, int]] = set()
    checked: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        by_key = collections.defaultdict(list)
        for i in members:
            by_key[keys[i]].append(i)
        if len(by_key) < 2:
            continue
        groups = list(by_key.values())
        for x in range(len(groups)):
            for y in range(x + 1, len(groups)):
                for i in groups[x]:
                    for j in groups[y]:
                        p = (i, j) if i < j else (j, i)
                        if p in checked:
                            continue
                        checked.add(p)
                        if jaccard(stems[i], stems[j]) >= threshold:
                            pairs.add(p)
    return pairs


def compared_text(item: sources.Item) -> str:
    """What near copies are judged on: the stem (problem without options or fixed wording), except for a choice item
    whose stem is too short to identify a question ("Which of the following is true?"), judged with its options."""
    if item.kind == "choice" and len(letters_digits(item.stem)) < STEM_MIN_CHARS:
        return item.problem
    return item.stem


_DIGITS = re.compile(r"\d+")


def numbers(item: sources.Item) -> list[str]:
    """The digit runs of the compared text's letters-and-digits string (`\\frac12` and `\\frac{1}{2}` agree)."""
    return _DIGITS.findall(letters_digits(compared_text(item)))


def is_copy(a: sources.Item, b: sources.Item, sa: np.ndarray, sb: np.ndarray) -> bool:
    """Copies: the same problem in letters and digits; or compared texts with the same numbers and either Jaccard >=
    NEAR_JACCARD between their character 8-grams or edit similarity >= NEAR_EDIT between their letters-and-digits
    strings; or texts that are near-identical (edit similarity >= SAME_EDIT) whatever their numbers. The number
    condition keeps template siblings apart (`largest prime factor of 78` / `of 999` reach Jaccard 0.81), while
    reformatted and reworded copies keep their numbers."""
    if letters_digits(a.problem) == letters_digits(b.problem):
        return True
    jac = jaccard(sa, sb)
    if jac < CANDIDATE_JACCARD:
        return False
    same = numbers(a) == numbers(b)
    if same and jac >= NEAR_JACCARD:
        return True
    m = difflib.SequenceMatcher(None, letters_digits(compared_text(a)), letters_digits(compared_text(b)), autojunk=False)
    floor = NEAR_EDIT if same else SAME_EDIT
    return m.quick_ratio() >= floor and m.ratio() >= floor


def copies(items: list[sources.Item]) -> tuple[list[list[int]], set[tuple[int, int]]]:
    """(groups of exact copies: the same problem up to case and whitespace; copy pairs under `is_copy`). Copy pairs
    are looked for between sources and within a training source (whose items fall into different parts), not within
    an unseen source, which is one unit in one part: there symbol-only differences (BBH's bracket sequences and
    Boolean expressions) and shared templates make different items look alike, and only exact copies count."""
    keys = [it.source if sources.SOURCES[it.source].role == "unseen" else it.uid for it in items]
    exact, alnum = collections.defaultdict(list), collections.defaultdict(list)
    for k, it in enumerate(items):
        exact[normalized(it.problem)].append(k)
        alnum[letters_digits(it.problem)].append(k)
    sh = [shingles(compared_text(it)) for it in items]
    pairs = {(i, j) for i, j in near_pairs(sh, keys, CANDIDATE_JACCARD) if is_copy(items[i], items[j], sh[i], sh[j])}
    for ks in alnum.values():
        pairs.update((i, j) for x, i in enumerate(ks) for j in ks[x + 1:] if keys[i] != keys[j])
    return [ks for ks in exact.values() if len(ks) > 1], pairs


def copied_elsewhere(items: list[sources.Item], groups: list[list[int]], near: set[tuple[int, int]]) -> set[int]:
    """Items with a copy in another source."""
    out = {k for g in groups if len({items[k].source for k in g}) > 1 for k in g}
    out.update(k for i, j in near if items[i].source != items[j].source for k in (i, j))
    return out


def deduplicate(items: list[sources.Item], parts: list[str], groups: list[list[int]],
                near: set[tuple[int, int]]) -> tuple[list[str | None], collections.Counter]:
    """For every item the reason it is removed (None: kept), and removal counts by (source, part, reason).

    An item is removed when it is a copy of a better-ranked item: rank is (part, size of the source, table order,
    position), parts ranked blind2, selection, train, so that within a part the smaller source keeps its copy (about
    600 TheoremQA problems recur in MMLU-Pro, 7,300 MATH-train problems in DeepScaleR). Every exact copy counts; a near
    copy counts between different sources or different parts (within one source and part, items sharing a context,
    such as questions on one ProofWriter theory, are distinct items). Omni-MATH items with a copy in DeepScaleR are
    removed whatever their parts: omni_rest is Omni-MATH minus DeepScaleR."""
    order = {s: k for k, s in enumerate(sources.SOURCES)}
    size = collections.Counter(it.source for it in items)
    rank = [(PART_RANK[p], size[it.source], order[it.source], k) for k, (it, p) in enumerate(zip(items, parts))]
    pairs = [(i, j) for g in groups for x, i in enumerate(g) for j in g[x + 1:]]
    pairs += [(i, j) for i, j in near if items[i].source != items[j].source or parts[i] != parts[j]]
    removed: list[str | None] = [None] * len(items)
    for i, j in pairs:
        si, sj = items[i].source, items[j].source
        if {si, sj} == {"omni_rest", "deepscaler"}:
            loser, why = (i if si == "omni_rest" else j), "in DeepScaleR"
        else:
            loser = i if rank[i] > rank[j] else j
            winner = j if loser == i else i
            why = "copy in the same part" if parts[i] == parts[j] else f"copy in {parts[winner]}"
        if removed[loser] is None:
            removed[loser] = why
    counts = collections.Counter((it.source, p, why) for it, p, why in zip(items, parts, removed) if why)
    return removed, counts


# ---------------------------------------------------------------------------------------------------------------
# Filters


_ENVIRONMENT = re.compile(r"\\(?:begin|end)\{[A-Za-z*]+\}")
_WORD = re.compile(r"(?<![\\A-Za-z])[A-Za-z]{3,}")
MATH_WORDS = {"sin", "cos", "tan", "cot", "sec", "csc", "log", "exp", "sqrt", "arcsin", "arccos", "arctan", "sinh",
              "cosh", "tanh", "max", "min", "gcd", "lcm", "mod", "det", "deg", "inf", "lim"}


def free_text(item: sources.Item) -> bool:
    """An open reference that is text rather than one mathematical value: \\text or \\mbox, more than MAX_GOLD_CHARS
    characters, equations listed or chained (`r = 4, s = 5`, `a=b=c=d`; a tuple equation is rewritten as its tuple by
    the loader), or a bare word of three or more letters (`9 and -7`, `all`, `Saturday`), which math-verify would read
    as a product of symbols, so that any anagram would match it."""
    if item.kind != "open":
        return False
    gold = item.gold["answer"] if isinstance(item.gold, dict) else item.gold  # a typed reference
    if "\\text" in gold or "\\mbox" in gold or len(gold) > MAX_GOLD_CHARS:
        return True
    if "=" in gold and ("," in gold or gold.count("=") > 1):  # equations listed or chained: not one comparable value
        return True
    return any(w.lower() not in MATH_WORDS for w in _WORD.findall(_ENVIRONMENT.sub(" ", gold)))


@functools.lru_cache(maxsize=2)
def _tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def prompt_tokens(tasks: list[Task], tokenizer_name: str) -> list[int]:
    """Prompt length of each task under the system prompt and chat template: the question's tokens plus the template's
    (tokenised separately, so at most a token or two off at the seam)."""
    from evotaste.tasks import prompt_ids

    tok = _tokenizer(tokenizer_name)
    overhead = len(prompt_ids(tok, ""))
    lengths = tok([t.question for t in tasks], add_special_tokens=False).input_ids if tasks else []
    return [len(ids) + overhead for ids in lengths]


# ---------------------------------------------------------------------------------------------------------------
# Building the parts


def uid_hash(uids) -> str:
    return hashlib.sha256("\n".join(sorted(uids)).encode()).hexdigest()


def candidates() -> list[sources.Item]:
    """Every source's items with a gold reference, in table order."""
    return [it for s in sources.SOURCES.values() for it in s.load()]


def build(units: dict[str, str], tokenizer_name: str):
    """(items, parts, reasons, dedup counts): every candidate item, its part, and why it is excluded (None if it is
    kept, gold_ok aside); reasons are "no gold", a de-duplication reason, "free-text gold" or "prompt too long"."""
    return build_with_copies(units, tokenizer_name)[:4]


def build_with_copies(units: dict[str, str], tokenizer_name: str):
    """`build`'s result and the copy structure it used, (exact groups, copy pairs) of `copies`."""
    items = candidates()
    groups, near = copies(items)
    copied = copied_elsewhere(items, groups, near)
    parts = [provisional_part(it, units, k in copied) for k, it in enumerate(items)]
    removed, counts = deduplicate(items, parts, groups, near)
    reasons: list[str | None] = [("no gold" if it.gold is None else why) for it, why in zip(items, removed)]
    for k, it in enumerate(items):
        if reasons[k] is None and free_text(it):
            reasons[k] = "free-text gold"
    alive = [k for k, r in enumerate(reasons) if r is None]
    lengths = prompt_tokens([make_task(items[k]) for k in alive], tokenizer_name)
    for k, n in zip(alive, lengths):
        if n > MAX_PROMPT_TOKENS:
            reasons[k] = "prompt too long"
    return items, parts, reasons, counts, (groups, near)


class SuitesFile:
    """What loading needs from configs/suites-r2.yaml: the unit draw, the tokenizer, the gold verdicts of train and
    selection items, and the uid hash of every part."""

    def __init__(self, path: str | Path):
        data = yaml.safe_load(Path(path).read_text())
        self.path = Path(path)
        self.units: dict[str, str] = {u: v["part"] for u, v in data["units"].items()}
        self.tokenizer: str = data["generated"]["rules"]["tokenizer"]
        from evotaste.verify import RULES_VERSION

        rules = data["generated"]["rules"].get("verifier_rules")
        if rules != RULES_VERSION:  # its stored verdicts and the vault's hash belong to other rules
            raise RuntimeError(f"{path} was generated under verifier rules {rules}, the verifier now has "
                               f"{RULES_VERSION}; regenerate it with scripts/make_suites_r2.py")
        self.gold_failed: set[str] = {f"{s}/{i}" for s, ids in (data.get("gold_failed") or {}).items() for i in ids}
        self.hashes: dict[tuple[str, str], str] = {(s, p): v["sha256"] for s, ps in data["parts"].items()
                                                   for p, v in ps.items()}


def cache_key(path: str | Path) -> str:
    """SHA-256 of the suites file and of the code that builds, filters and words the open parts."""
    h = hashlib.sha256(Path(path).read_bytes())
    for name in BUILD_CODE:
        h.update(b"\0" + name.encode() + b"\0" + (_PACKAGE / name).read_bytes())
    return h.hexdigest()


def _read_cache(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json.gz"
    if not path.exists():
        return None
    data = json.loads(gzip.decompress(path.read_bytes()))
    return {(e["source"], e["part"]): tuple(sources.Item(**d) for d in e["items"]) for e in data["parts"]}


def _write_cache(key: str, parts: dict[tuple[str, str], tuple[sources.Item, ...]]) -> None:
    """The open parts only: a vault item reaching this function is a bug, and nothing is written."""
    if any(part == VAULT for _, part in parts):
        raise RuntimeError("refusing to cache the vault")
    entries = [{"source": s, "part": p, "items": [dataclasses.asdict(it) for it in v]} for (s, p), v in parts.items()]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_DIR / f".{key}.{os.getpid()}.tmp"
    tmp.write_bytes(gzip.compress(json.dumps({"key": key, "parts": entries}).encode(), compresslevel=1))
    os.replace(tmp, CACHE_DIR / f"{key}.json.gz")  # concurrent writers write the same content


@functools.lru_cache(maxsize=2)
def _open_parts(path: str) -> dict[tuple[str, str], tuple[sources.Item, ...]]:
    """Train and selection items by (source, part), after every filter, each list in a fixed seeded order and checked
    against the suites file's hashes. Read from the disk cache when it was written for this suites file and this code;
    otherwise built and then cached. Blind items are not kept, and never cached."""
    f = SuitesFile(path)
    key = cache_key(path)
    parts = _read_cache(key)
    if parts is None:
        items, provisional, reasons, _ = build(f.units, f.tokenizer)
        out = collections.defaultdict(list)
        for it, part, why in zip(items, provisional, reasons):
            if why is None and part != VAULT and it.uid not in f.gold_failed:
                out[(it.source, part)].append(it)
        parts = {k: tuple(sorted(v, key=lambda it: _hash2(it.uid))) for k, v in out.items()}
        _write_cache(key, parts)
    keys = {k for k in f.hashes if k[1] != VAULT} | set(parts)
    for source, part in sorted(keys):
        if uid_hash(it.uid for it in parts.get((source, part), ())) != f.hashes.get((source, part)):
            raise RuntimeError(f"{source}/{part}: the items differ from those {path} was generated for (code, data "
                               "or verifier changed); regenerate it with scripts/make_suites_r2.py")
    return parts


@functools.lru_cache(maxsize=None)
def _open_tasks(path: str, source: str, part: str) -> tuple[Task, ...]:
    return tuple(make_task(it) for it in _open_parts(path).get((source, part), ()))


@functools.lru_cache(maxsize=1)
def _vault_candidates(path: str) -> dict[str, list[sources.Item]]:
    """The vault's items by source before the gold check; reached only after the receipt check in `items`."""
    f = SuitesFile(path)
    items, parts, reasons, _ = build(f.units, f.tokenizer)
    out = collections.defaultdict(list)
    for it, part, why in zip(items, parts, reasons):
        if why is None and part == VAULT:
            out[it.source].append(it)
    return dict(out)


def _blind_part(path: str, source: str) -> tuple[sources.Item, ...]:
    """Blind items of one source, gold-checked now (their verdicts are never stored) and checked against the hash."""
    from evotaste.tasks import gold_ok_many

    f = SuitesFile(path)
    vault = _vault_candidates(path).get(source, [])
    ok = gold_ok_many([make_task(it) for it in vault])
    vault = [it for it, good in zip(vault, ok) if good]
    if uid_hash(it.uid for it in vault) != f.hashes.get((source, VAULT), uid_hash([])):
        raise RuntimeError(f"{source}/{VAULT}: the vault's items differ from those {path} was generated for")
    return tuple(sorted(vault, key=lambda it: _hash2(it.uid)))


EXPOSURE = Path("configs/vault-exposure.yaml")


def exposed(uids, path: str | Path = EXPOSURE) -> set[str]:
    """The given uids that configs/vault-exposure.yaml lists as exposed: a stored model output exists on the item or
    on a copy of it (scripts/vault_exposure.py). The blind evaluation reports its results with and without them."""
    listed = set(yaml.safe_load(Path(path).read_text())["sha256"])
    return {u for u in uids if hashlib.sha256(u.encode()).hexdigest() in listed}


def items(source: str, part: str, receipt: BlindReceipt | None = None, suites: str | Path = SUITES) -> list[Task]:
    """Tasks of one part of a source (default wrapper, no stratum), in a fixed seeded order. The vault needs the
    blind guard's receipt."""
    if part not in PARTS:
        raise ValueError(f"unknown part {part!r} (parts: {PARTS})")
    if source not in sources.SOURCES:
        raise ValueError(f"unknown source {source!r}")
    if part == VAULT:
        check_receipt(receipt, VAULT)
        return [make_task(it) for it in _blind_part(str(suites), source)]
    return list(_open_tasks(str(suites), source, part))
