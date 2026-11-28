"""Analysis of the method census (docs/dev/SCALE_SPEC.md, WP-M "Analysis"; docs/dev/SCALE_NOTES.md).

  python -m evotaste.analysis.method results/method/<name> [--out DIR] [--labels FILE ...] [--boot 1000]

Reads a finished method census, every stage of it (missing chunks, or manifests whose recorded upstream hashes are not
those on disk, stop the command), and writes report.txt, figures.json (every figure of the report) and pairs.csv (one
row per judged pair) to DIR, by default <dir>/analysis. With --labels, reader label files (JSON lines `reader_id`,
`relation`, optionally `reader`) are scored against the judge through reader-key.jsonl.

Intervals are 95% percentile intervals of an item bootstrap: items are resampled within the cells of the draw
(domain x census stratum), the same draws for every figure, so that differences between models are paired. Shares of
pairs are ratios of item sums, so an item's pairs move together. The census stratum is the 1.7B census's (frontier 0,
sweet 1-6, mastered 7-8 of 8); a model's own stratum is the same rule on its attempts here.
"""

from __future__ import annotations

import argparse
import ast
import collections
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from evotaste import method
from evotaste.analysis.stats import ci, item_draws
from evotaste.tasks import stratum_of

DOMAINS = method.DOMAINS
STRATA = method.STRATA
RELATIONS = method.RELATIONS
ORDINAL = {r: k for k, r in enumerate(RELATIONS)}  # same < variant < different
SEED = 20261128


# ---------------------------------------------------------------------------------------------------------------
# Construct profile of a program (mechanical cross-check of the judge on code)

STRING_METHODS = frozenset({"split", "rsplit", "join", "strip", "lstrip", "rstrip", "replace", "find", "rfind",
                            "startswith", "endswith", "lower", "upper", "isdigit", "isalpha", "isalnum", "isspace",
                            "isupper", "islower", "partition", "rpartition", "splitlines", "translate", "maketrans",
                            "zfill", "ljust", "rjust", "center", "title", "capitalize", "swapcase", "casefold",
                            "format"})
MODULES = {"heapq": "heap", "bisect": "bisect", "itertools": "itertools", "re": "regex", "math": "math"}
HASHING = frozenset({"dict", "set", "frozenset", "Counter", "defaultdict", "OrderedDict"})
LOOPS = (ast.For, ast.AsyncFor, ast.While)
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _loop_depth(node, depth: int = 0) -> int:
    deepest = depth
    for child in ast.iter_child_nodes(node):
        deepest = max(deepest, _loop_depth(child, depth + isinstance(child, LOOPS)))
    return deepest


def _called(node) -> str | None:
    f = node.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def construct_profile(program: str | None) -> frozenset[str] | None:
    """The coarse constructs a program uses, as a set of names: recursion, for_loop, while_loop, comprehension,
    sorting, hashing (dict and set displays, comprehensions and constructors, Counter, defaultdict), heap, deque,
    bisect, itertools, regex, math, string_methods, index_scan (for over range(len(...)), or a while loop that
    subscripts), early_return (a return that is not a function's last statement) and loop_depth_<1|2|3> (deepest
    nesting of loops, 3 meaning 3 or more). None when there is no parsable program."""
    try:
        tree = ast.parse(program) if program else None
    except (SyntaxError, ValueError):
        return None
    if tree is None:
        return None
    found = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            found |= {MODULES[a.name.split(".")[0]] for a in n.names if a.name.split(".")[0] in MODULES}
        elif isinstance(n, ast.ImportFrom) and n.module:
            if n.module.split(".")[0] in MODULES:
                found.add(MODULES[n.module.split(".")[0]])
            if any(a.name == "deque" for a in n.names):
                found.add("deque")
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            found.add("for_loop")
            it = n.iter
            if (isinstance(it, ast.Call) and _called(it) == "range"
                    and any(isinstance(a, ast.Call) and _called(a) == "len" for x in it.args for a in ast.walk(x))):
                found.add("index_scan")
        elif isinstance(n, ast.While):
            found.add("while_loop")
            if any(isinstance(x, ast.Subscript) for x in ast.walk(n)):
                found.add("index_scan")
        elif isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            found.add("comprehension")
        elif isinstance(n, ast.Call):
            name = _called(n)
            if name in ("sorted", "sort"):
                found.add("sorting")
            if name in HASHING:
                found.add("hashing")
            if name == "deque":
                found.add("deque")
            if isinstance(n.func, ast.Attribute) and name in STRING_METHODS:
                found.add("string_methods")
        if isinstance(n, (ast.Dict, ast.Set, ast.DictComp, ast.SetComp)):
            found.add("hashing")
    for f in (n for n in ast.walk(tree) if isinstance(n, FUNCTIONS)):
        inner = list(ast.walk(f))
        if any(isinstance(c, ast.Call) and _called(c) == f.name for c in inner):
            found.add("recursion")
        if any(isinstance(r, ast.Return) and r is not f.body[-1] for r in inner):
            found.add("early_return")
    depth = _loop_depth(tree)
    if depth:
        found.add(f"loop_depth_{min(depth, 3)}")
    return frozenset(found)


def jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard distance of two construct profiles (0 for two empty ones)."""
    union = a | b
    return 1 - len(a & b) / len(union) if union else 0.0


# ---------------------------------------------------------------------------------------------------------------
# Typicality: within-item statistics of the mean token log-probability


def terciles(x: np.ndarray) -> np.ndarray:
    """Within-item tercile of each sample by x (items x samples): 0 the least typical (lowest), 2 the most."""
    k = x.shape[1]
    return 3 * np.argsort(np.argsort(x, axis=1, kind="stable"), axis=1, kind="stable") // k


def typicality(x: np.ndarray, correct: np.ndarray) -> dict[str, np.ndarray]:
    """Per item (rows of items x samples arrays): `mixed`, whether it has both correct and wrong samples; `d`, the
    standardised difference (mean x of correct - mean x of wrong) / SD of x over the item's samples, NaN unless mixed
    with SD > 0; per tercile t (0 least typical) the samples `n_t` and correct samples `c_t`."""
    correct = correct.astype(bool)
    k = correct.sum(1)
    mixed = (k > 0) & (k < correct.shape[1])
    sd = x.std(1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = (np.where(correct, x, 0).sum(1) / k) - (np.where(~correct, x, 0).sum(1) / (correct.shape[1] - k))
        d = np.where(mixed & (sd > 0), diff / sd, np.nan)
    t = terciles(x)
    return {"mixed": mixed, "d": d, "n_t": np.stack([(t == j).sum(1) for j in range(3)], 1),
            "c_t": np.stack([((t == j) & correct).sum(1) for j in range(3)], 1)}


def residualize(x: np.ndarray, log_len: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, dict]:
    """x minus the within-item linear effect of log length: one slope per group, fitted on the deviations of x and of
    log length from their item means over all items of the group (items x samples arrays). Returns the residual
    and the slopes."""
    xc, lc = x - x.mean(1, keepdims=True), log_len - log_len.mean(1, keepdims=True)
    out, slopes = x.copy(), {}
    for g in dict.fromkeys(groups):
        rows = groups == g
        denom = (lc[rows] ** 2).sum()
        slopes[g] = float((xc[rows] * lc[rows]).sum() / denom) if denom > 0 else 0.0
        out[rows] = x[rows] - slopes[g] * log_len[rows]
    return out, slopes


# ---------------------------------------------------------------------------------------------------------------
# Data


@dataclass
class Census:
    base: Path
    items: pd.DataFrame  # uid, domain, source, passes, samples, stratum (census)
    subjects: list[str]
    attempts: dict[str, dict[str, np.ndarray]]  # label -> field -> items x samples
    alternatives: dict[str, np.ndarray]  # label -> reward, items x samples (NaN rows: no alternatives asked)
    sides: dict[tuple, dict]  # (label, stage, uid, sample) -> logprob_mean, program (code)
    judged: pd.DataFrame  # one row per pair with its verdict
    thinking: object  # the judge's thinking setting


FIELDS = ("reward", "tokens", "logprob", "complied", "truncated")


def _grid(records: list[dict], index: dict[str, int], samples: int, what: str) -> tuple[dict, np.ndarray]:
    grid = {f: np.full((len(index), samples), np.nan) for f in FIELDS}
    seen = np.zeros((len(index), samples), bool)
    for r in records:
        i, k = index[r["uid"]], r["sample"]
        if seen[i, k]:
            raise ValueError(f"{what}: {r['uid']} sample {k} twice")
        seen[i, k] = True
        for f, v in zip(FIELDS, (r["reward"], r["completion_tokens"], r["logprob_mean"], r["complied"],
                                 r["truncated"])):
            grid[f][i, k] = float(v)
    return grid, seen


def load(base: str | Path) -> Census:
    base = Path(base)
    items = pd.DataFrame(method.read_items(base))
    items["stratum"] = [stratum_of(p, n) for p, n in zip(items["passes"], items["samples"])]
    index = {u: i for i, u in enumerate(items["uid"])}
    pm = method.stored_manifest(base, "pairs")
    subjects = pm["config"]["subjects"]
    stages = [("items", None)] + [(s, lbl) for lbl in subjects for s in ("attempts", "alternatives")] + [
        ("pairs", None), ("judge", None)]
    manifests = {method._key(s, lbl): method.stored_manifest(base, s, lbl) for s, lbl in stages}
    for key, m in manifests.items():
        for up, sha in m["upstream"].items():
            if manifests[up]["sha256"] != sha:
                raise SystemExit(f"{base}: stage {key} was made from another {up} than the one on disk")
    attempts, alternatives, sides = {}, {}, {}
    for lbl in subjects:
        k = manifests[f"attempts/{lbl}"]["config"]["attempts"]["samples"]
        recs = method.read_stage(base, "attempts", lbl)
        attempts[lbl], seen = _grid(recs, index, k, f"attempts of {lbl}")
        if not seen.all():
            raise SystemExit(f"attempts of {lbl}: {int((~seen).sum())} samples missing")
        k_alt = manifests[f"alternatives/{lbl}"]["config"]["alternatives"]["samples"]
        alts = method.read_stage(base, "alternatives", lbl)
        grid, seen = _grid(alts, index, k_alt, f"alternatives of {lbl}")
        if (seen.any(1) & ~seen.all(1)).any():
            raise SystemExit(f"alternatives of {lbl}: items with some samples missing")
        alternatives[lbl] = grid["reward"]
        for stage, rs in (("attempts", recs), ("alternatives", alts)):
            for r in rs:
                sides[(lbl, stage, r["uid"], r["sample"])] = {
                    "logprob": r["logprob_mean"], "program": r["extracted"] if r["kind"] == "code" else None}
    judged = method.read_stage(base, "judge")
    pairs = method.read_jsonl_gz(base / "pairs.jsonl.gz")
    if [r["id"] for r in judged] != [p["id"] for p in pairs]:
        raise SystemExit(f"{base}: the judged pairs are not those of pairs.jsonl.gz")
    rows = [{"id": r["id"], "kind": r["kind"], "uid": r["uid"], "domain": r["domain"], "of": r["of"],
             "model_a": r["a"]["model"], "stage_a": r["a"]["stage"], "sample_a": r["a"]["sample"],
             "model_b": r["b"]["model"], "stage_b": r["b"]["stage"], "sample_b": r["b"]["sample"],
             "parsed": r["parsed"], "relation": r["relation"], "thought": r["thought"],
             "truncated_a": r["truncated_a"], "truncated_b": r["truncated_b"], "prompt_tokens": r["prompt_tokens"],
             "reply_tokens": r["completion_tokens"], "attempts": len(r["replies"])} for r in judged]
    judged = pd.DataFrame(rows)
    judged["item"] = [index[u] for u in judged["uid"]]
    return Census(base, items, subjects, attempts, alternatives, sides, judged,
                  manifests["judge"]["config"]["judge"]["thinking"])


# ---------------------------------------------------------------------------------------------------------------
# Bootstrap helpers


class Boot:
    """Row 0: the estimate (every item once); rows 1..b: item bootstrap draws within the given groups."""

    def __init__(self, groups, b: int, seed: int = SEED):
        self.w = np.vstack([np.ones(len(groups)), item_draws(groups, b, np.random.default_rng(seed))])

    def ratio(self, num, den) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.w @ np.asarray(num, float)) / (self.w @ np.asarray(den, float))


def fig(draws: np.ndarray, n: int | None = None) -> dict:
    out = {"value": None if not np.isfinite(draws[0]) else round(float(draws[0]), 4),
           "ci": [None if not np.isfinite(x) else round(x, 4) for x in ci(draws[1:])]}
    return out if n is None else out | {"n": int(n)}


def _domains(items: pd.DataFrame):
    yield "all", np.ones(len(items), bool)
    for d in DOMAINS:
        yield d, (items["domain"] == d).to_numpy()


def kappa(a, b) -> float:
    """Cohen's kappa of two label sequences."""
    a, b = np.asarray(a), np.asarray(b)
    if not len(a):
        return math.nan
    cats = sorted(set(a) | set(b))
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in cats))
    return (po - pe) / (1 - pe) if pe < 1 else math.nan


def _item_counts(df: pd.DataFrame, n: int, mask=None) -> np.ndarray:
    """Per item, the number of rows of df (or of rows where mask holds)."""
    item = df["item"].to_numpy()
    return np.bincount(item, weights=None if mask is None else np.asarray(mask, float), minlength=n)


# ---------------------------------------------------------------------------------------------------------------
# Figures


def accuracy(c: Census, boot: Boot) -> dict:
    out = {}
    for lbl in c.subjects:
        a = c.attempts[lbl]
        k = a["reward"].shape[1]
        acc = a["reward"].mean(1)
        res = {"accuracy": {}, "complied": {}, "truncated": {}, "tokens": {}}
        for dom, dmask in _domains(c.items):
            for st in ("all", *STRATA):
                mask = dmask & ((c.items["stratum"] == st).to_numpy() if st != "all" else True)
                res["accuracy"][f"{dom}/{st}"] = fig(boot.ratio(acc * mask, mask), mask.sum())
            for f in ("complied", "truncated", "tokens"):
                res[f][dom] = fig(boot.ratio(a[f].mean(1) * dmask, dmask), dmask.sum())
            res["tokens"][dom] |= {f"p{q}": int(np.percentile(a["tokens"][dmask], q)) for q in (50, 90, 99)}
        own = np.array([stratum_of(int(p), k) for p in a["reward"].sum(1)])
        res["own_strata"] = {f"own {o} / census {s}": int(((own == o) & (c.items["stratum"] == s)).sum())
                             for o in STRATA for s in STRATA}
        out[lbl] = res
    for l1, l2 in itertools.combinations(c.subjects, 2):
        k1, k2 = c.attempts[l1]["reward"].shape[1], c.attempts[l2]["reward"].shape[1]
        o1 = np.array([stratum_of(int(p), k1) for p in c.attempts[l1]["reward"].sum(1)])
        o2 = np.array([stratum_of(int(p), k2) for p in c.attempts[l2]["reward"].sum(1)])
        out[f"{l1} x {l2}"] = {f"{l1} {s1} / {l2} {s2}": int(((o1 == s1) & (o2 == s2)).sum())
                               for s1 in STRATA for s2 in STRATA}
    return out


def own_stratum(c: Census, lbl: str) -> np.ndarray:
    r = c.attempts[lbl]["reward"]
    return np.array([stratum_of(int(p), r.shape[1]) for p in r.sum(1)])


def _verdict_counts(df: pd.DataFrame, n: int) -> dict[str, np.ndarray]:
    ok = df[df["parsed"]]
    return {"pairs": _item_counts(ok, n), "different": _item_counts(ok, n, ok["relation"] == "different"),
            "variant_or_different": _item_counts(ok, n, ok["relation"] != "same")}


def variation(c: Census, boot: Boot) -> tuple[dict, dict]:
    n = len(c.items)
    sp = c.judged[c.judged["kind"] == "spontaneous"]
    counts = {lbl: _verdict_counts(sp[sp["model_a"] == lbl], n) for lbl in c.subjects}
    matched = np.logical_and.reduce([counts[lbl]["pairs"] > 0 for lbl in c.subjects])
    out = {}
    for lbl in c.subjects:
        v = counts[lbl]
        scopes = {"all": v["pairs"] > 0, "matched": matched,
                  "own sweet": (own_stratum(c, lbl) == "sweet") & (v["pairs"] > 0)}
        res = {}
        for scope, smask in scopes.items():
            for dom, dmask in _domains(c.items):
                m = smask & dmask
                res[f"{scope}/{dom}"] = {
                    "different": fig(boot.ratio(v["different"] * m, v["pairs"] * m), v["pairs"][m].sum()),
                    "variant_or_different": fig(boot.ratio(v["variant_or_different"] * m, v["pairs"] * m)),
                    "items_any_different": fig(boot.ratio((v["different"] > 0) * m, m), m.sum())}
        res["unparsed"] = int(((sp["model_a"] == lbl) & ~sp["parsed"]).sum())
        out[lbl] = res
    for l1, l2 in itertools.combinations(c.subjects, 2):
        res = {}
        for dom, dmask in _domains(c.items):
            m = matched & dmask
            r1 = boot.ratio(counts[l1]["different"] * m, counts[l1]["pairs"] * m)
            r2 = boot.ratio(counts[l2]["different"] * m, counts[l2]["pairs"] * m)
            res[dom] = fig(r2 - r1, m.sum())
        out[f"{l2} - {l1} (matched, different)"] = res
    return out, counts


def requested(c: Census, boot: Boot) -> dict:
    n = len(c.items)
    out = {}
    for lbl in c.subjects:
        alt = c.alternatives[lbl]
        has = np.isfinite(alt).all(1)
        correct = has & (np.nan_to_num(alt) == 1).any(1)
        rq = c.judged[(c.judged["kind"] == "requested") & (c.judged["model_a"] == lbl)]
        parsed = _item_counts(rq, n, rq["parsed"]) > 0
        diff = _item_counts(rq, n, rq["parsed"] & (rq["relation"] == "different")) > 0
        vd = _item_counts(rq, n, rq["parsed"] & (rq["relation"] != "same")) > 0
        known = has & (~correct | parsed)  # a correct alternative whose verdict did not parse is left out
        res = {}
        for dom, dmask in _domains(c.items):
            m, mk = has & dmask, known & dmask
            res[dom] = {"items": int(m.sum()),
                        "alternative_correct": fig(boot.ratio(correct * m, m)),
                        "correct_and_different": fig(boot.ratio(diff * mk, mk)),
                        "correct_and_variant_or_different": fig(boot.ratio(vd * mk, mk)),
                        "alternative_accuracy_per_sample": fig(boot.ratio(np.nan_to_num(alt).mean(1) * m, m)),
                        "attempt_accuracy_same_items": fig(boot.ratio(c.attempts[lbl]["reward"].mean(1) * m, m)),
                        "unparsed": int((correct & ~parsed & dmask).sum())}
        out[lbl] = res
    return out


def cross(c: Census, boot: Boot, counts: dict) -> dict:
    n = len(c.items)
    out = {}
    for l1, l2 in itertools.combinations(c.subjects, 2):
        cr = c.judged[(c.judged["kind"] == "cross") & (c.judged["model_a"].isin([l1, l2]))
                      & (c.judged["model_b"].isin([l1, l2]))]
        v = _verdict_counts(cr, n)
        both = (v["pairs"] > 0) & (counts[l1]["pairs"] > 0) & (counts[l2]["pairs"] > 0)
        res = {}
        for dom, dmask in _domains(c.items):
            m = both & dmask
            x = boot.ratio(v["different"] * m, v["pairs"] * m)
            w1 = boot.ratio(counts[l1]["different"] * m, counts[l1]["pairs"] * m)
            w2 = boot.ratio(counts[l2]["different"] * m, counts[l2]["pairs"] * m)
            res[dom] = {"items": int(m.sum()), "cross_different": fig(x),
                        "cross_variant_or_different": fig(boot.ratio(v["variant_or_different"] * m, v["pairs"] * m)),
                        f"within_{l1}_different": fig(w1), f"within_{l2}_different": fig(w2),
                        f"cross - within_{l1}": fig(x - w1), f"cross - within_{l2}": fig(x - w2)}
        out[f"{l1} x {l2}"] = res
    return out


def code_distances(c: Census) -> pd.Series:
    """Per judged code pair whose two programs parse, the Jaccard distance of their construct profiles."""
    out = {}
    for row in c.judged[c.judged["domain"] == "code"].itertuples():
        pa = construct_profile(c.sides[(row.model_a, row.stage_a, row.uid, row.sample_a)]["program"])
        pb = construct_profile(c.sides[(row.model_b, row.stage_b, row.uid, row.sample_b)]["program"])
        if pa is not None and pb is not None:
            out[row.id] = jaccard(pa, pb)
    return pd.Series(out, dtype=float)


def spearman_boot(x: np.ndarray, y: np.ndarray, item: np.ndarray, boot: Boot) -> np.ndarray:
    out = []
    for w in boot.w[:, item].astype(int):
        xs, ys = np.repeat(x, w), np.repeat(y, w)
        out.append(sps.spearmanr(xs, ys)[0] if len(set(xs)) > 1 and len(set(ys)) > 1 else np.nan)
    return np.array(out)


def judge_quality(c: Census, boot: Boot, distance: pd.Series, labels: list[Path]) -> dict:
    j, n = c.judged, len(c.items)
    out = {"by_kind": {}, "thinking": c.thinking}
    for kind, df in j.groupby("kind"):
        ok = df[df["parsed"]]
        out["by_kind"][kind] = {"pairs": len(df), "unparsed": int((~df["parsed"]).sum()),
                                "retried": int((df["attempts"] > 1).sum()),
                                "relations": {r: int((ok["relation"] == r).sum()) for r in RELATIONS},
                                "thought": int(ok["thought"].sum()),
                                "solution_truncated": int((df["truncated_a"] | df["truncated_b"]).sum()),
                                "prompt_tokens_mean": round(float(df["prompt_tokens"].mean()), 1),
                                "reply_tokens_mean": round(float(df["reply_tokens"].mean()), 1)}
    rep = j[j["kind"] == "repeat"].merge(j[["id", "parsed", "relation"]], left_on="of", right_on="id",
                                         suffixes=("", "_orig"))
    rep = rep[rep["parsed"] & rep["parsed_orig"]]
    agree = (rep["relation"] == rep["relation_orig"]).to_numpy()
    out["repeat"] = {"pairs": len(rep), "agreement": fig(boot.ratio(_item_counts(rep, n, agree),
                                                                    _item_counts(rep, n))),
                     "kappa": round(kappa(rep["relation"], rep["relation_orig"]), 4) if len(rep) else None}
    code = j[j["id"].isin(distance.index) & j["parsed"] & (j["kind"] != "repeat")]
    dist = distance[code["id"]].to_numpy()
    ordinal = code["relation"].map(ORDINAL).to_numpy(float)
    out["code_mechanical"] = {"pairs": len(code),
                              "spearman": fig(spearman_boot(dist, ordinal, code["item"].to_numpy(), boot))
                              if len(code) > 2 else None,
                              "mean_distance": {r: fig(boot.ratio(_item_counts(code, n, dist * (code["relation"] == r)),
                                                                  _item_counts(code, n, code["relation"] == r)),
                                                       int((code["relation"] == r).sum())) for r in RELATIONS}}
    if labels:
        out["readers"] = readers(c.base, labels)
    return out


def readers(base: Path, files: list[Path]) -> dict:
    """Agreement and Cohen's kappa of each reader with the judge, and between readers, on the reader sample."""
    key = {k["reader_id"]: k for k in map(json.loads, (base / "reader-key.jsonl").read_text().splitlines())}
    labels: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for f in files:
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj["reader_id"] not in key or obj["relation"] not in RELATIONS:
                raise SystemExit(f"{f}: unknown reader id or relation in {line!r}")
            labels[obj.get("reader", Path(f).stem)][obj["reader_id"]] = obj["relation"]
    out = {}
    for who, got in labels.items():
        ids = sorted(got)
        judge, mine = [key[i]["relation"] for i in ids], [got[i] for i in ids]
        out[who] = {"pairs": len(ids), "agreement": round(float(np.mean(np.array(judge) == np.array(mine))), 4),
                    "kappa": round(kappa(judge, mine), 4),
                    "by_judge_verdict": {r: round(float(np.mean([m == r for j_, m in zip(judge, mine) if j_ == r])), 4)
                                         if r in judge else None for r in RELATIONS}}
    for r1, r2 in itertools.combinations(sorted(labels), 2):
        ids = sorted(set(labels[r1]) & set(labels[r2]))
        a, b = [labels[r1][i] for i in ids], [labels[r2][i] for i in ids]
        out[f"{r1} x {r2}"] = {"pairs": len(ids), "agreement": round(float(np.mean(np.array(a) == np.array(b))), 4)
                               if ids else None, "kappa": round(kappa(a, b), 4) if ids else None}
    return out


def typicality_figures(c: Census, boot: Boot) -> dict:
    out = {}
    domain = c.items["domain"].to_numpy()
    for lbl in c.subjects:
        a = c.attempts[lbl]
        correct = a["reward"] == 1
        adjusted, slopes = residualize(a["logprob"], np.log(a["tokens"]), domain)
        res = {"length_slope_per_domain": {d: round(s, 5) for d, s in slopes.items()}}
        for name, x in (("raw", a["logprob"]), ("length_adjusted", adjusted)):
            t = typicality(x, correct)
            mixed = t["mixed"]
            block = {}
            for dom, dmask in _domains(c.items):
                m = mixed & dmask & np.isfinite(t["d"])
                block[dom] = {
                    "mixed_items": int(m.sum()),
                    "standardised_difference": fig(boot.ratio(np.nan_to_num(t["d"]) * m, m)),
                    "accuracy_by_tercile": [fig(boot.ratio(t["c_t"][:, j] * m, t["n_t"][:, j] * m)) for j in range(3)],
                    "correct_in_least_typical_tercile": fig(boot.ratio(t["c_t"][:, 0] * dmask,
                                                                       t["c_t"].sum(1) * dmask)),
                    "chance": round(float(t["n_t"][dmask, 0].sum() / t["n_t"][dmask].sum()), 4)}
            res[name] = block
        out[lbl] = res
    return out


def typicality_and_method(c: Census, boot: Boot) -> dict:
    n = len(c.items)
    out = {}
    sp = c.judged[(c.judged["kind"] == "spontaneous") & c.judged["parsed"]]
    for lbl in c.subjects:
        df = sp[sp["model_a"] == lbl]
        gap = np.array([abs(c.sides[(r.model_a, r.stage_a, r.uid, r.sample_a)]["logprob"]
                            - c.sides[(r.model_b, r.stage_b, r.uid, r.sample_b)]["logprob"]) for r in df.itertuples()])
        draws = {}
        for r in RELATIONS:
            m = (df["relation"] == r).to_numpy()
            draws[r] = boot.ratio(_item_counts(df, n, gap * m), _item_counts(df, n, m))
        out[lbl] = {f"abs_logprob_gap_{r}": fig(draws[r], int((df["relation"] == r).sum())) for r in RELATIONS} | {
            "different - same": fig(draws["different"] - draws["same"])}
    return out


def pair_table(c: Census, distance: pd.Series) -> pd.DataFrame:
    df = c.judged.drop(columns=["item"]).copy()
    df["code_distance"] = df["id"].map(distance)
    df["logprob_gap"] = [abs(c.sides[(r.model_a, r.stage_a, r.uid, r.sample_a)]["logprob"]
                             - c.sides[(r.model_b, r.stage_b, r.uid, r.sample_b)]["logprob"]) for r in df.itertuples()]
    return df


# ---------------------------------------------------------------------------------------------------------------
# Report


def _fmt(f: dict | None) -> str:
    if not f or f["value"] is None:
        return "-"
    lo, hi = f["ci"]
    return f"{f['value']:.3f} [{'-' if lo is None else f'{lo:.3f}'}, {'-' if hi is None else f'{hi:.3f}'}]" + (
        f" n={f['n']}" if "n" in f else "")


def report(figures: dict, c: Census) -> str:
    lines = [f"Method census {c.base}: {len(c.items)} items, subjects {', '.join(c.subjects)}, judge thinking "
             f"{c.thinking}. Intervals: 95% item bootstrap within domain x census stratum.", ""]
    acc = figures["1_accuracy"]
    lines.append("1. Accuracy (item means over samples) by domain / census stratum")
    for lbl in c.subjects:
        lines.append(f"  {lbl}")
        for key, f in acc[lbl]["accuracy"].items():
            lines.append(f"    {key:<20} {_fmt(f)}")
        for f in ("complied", "truncated", "tokens"):
            lines.append(f"    {f:<10} " + "; ".join(f"{d} {_fmt(v)}" for d, v in acc[lbl][f].items()))
        lines.append("    own stratum x census stratum: "
                     + ", ".join(f"{k} {v}" for k, v in acc[lbl]["own_strata"].items()))
    for key, table in acc.items():
        if " x " in key:
            lines.append(f"  {key}: " + ", ".join(f"{k} {v}" for k, v in table.items()))
    lines += ["", "2. Spontaneous method variation (share of judged pairs of distinct correct attempts)"]
    for lbl in c.subjects:
        lines.append(f"  {lbl} (unparsed pairs: {figures['2_spontaneous'][lbl]['unparsed']})")
        for key, v in figures["2_spontaneous"][lbl].items():
            if key != "unparsed":
                lines.append(f"    {key:<16} different {_fmt(v['different'])}; variant or different "
                             f"{_fmt(v['variant_or_different'])}; items with any different "
                             f"{_fmt(v['items_any_different'])}")
    for key, v in figures["2_spontaneous"].items():
        if key not in c.subjects:
            lines.append(f"  {key}: " + "; ".join(f"{d} {_fmt(f)}" for d, f in v.items()))
    lines += ["", "3. Method on request (items with a correct attempt shown as reference)"]
    for lbl in c.subjects:
        for dom, v in figures["3_requested"][lbl].items():
            lines.append(f"  {lbl} {dom:<6} items {v['items']}: alternative correct {_fmt(v['alternative_correct'])}; "
                         f"correct and different {_fmt(v['correct_and_different'])}; correct and variant or different "
                         f"{_fmt(v['correct_and_variant_or_different'])}; per-sample accuracy "
                         f"{_fmt(v['alternative_accuracy_per_sample'])} (attempts "
                         f"{_fmt(v['attempt_accuracy_same_items'])}); unparsed {v['unparsed']}")
    lines += ["", "4. Cross-model pairs against within-model pairs (items with both)"]
    for key, block in figures["4_cross"].items():
        for dom, v in block.items():
            lines.append(f"  {key} {dom:<6} items {v['items']}: " + "; ".join(
                f"{k} {_fmt(f)}" for k, f in v.items() if k != "items"))
    q = figures["5_judge"]
    lines += ["", "5. Judge"]
    for kind, v in q["by_kind"].items():
        lines.append(f"  {kind:<12} {v}")
    lines.append(f"  swapped repeats: {q['repeat']['pairs']} pairs, agreement {_fmt(q['repeat']['agreement'])}, "
                 f"kappa {q['repeat']['kappa']}")
    cm = q["code_mechanical"]
    lines.append(f"  code: {cm['pairs']} pairs with parsable programs; Spearman(construct distance, verdict) "
                 f"{_fmt(cm['spearman'])}; mean distance "
                 + "; ".join(f"{r} {_fmt(f)}" for r, f in cm["mean_distance"].items()))
    for who, v in q.get("readers", {}).items():
        lines.append(f"  reader {who}: {v}")
    lines += ["", "6. Typicality (mean token log-probability) and correctness, items with correct and wrong attempts"]
    for lbl in c.subjects:
        t = figures["6_typicality"][lbl]
        lines.append(f"  {lbl} (within-item slope on log length: {t['length_slope_per_domain']})")
        for name in ("raw", "length_adjusted"):
            for dom, v in t[name].items():
                lines.append(f"    {name:<15} {dom:<6} mixed items {v['mixed_items']}: standardised difference "
                             f"{_fmt(v['standardised_difference'])}; accuracy by tercile (least typical first) "
                             + " / ".join(_fmt(f) for f in v["accuracy_by_tercile"])
                             + f"; correct answers in the least typical tercile "
                             f"{_fmt(v['correct_in_least_typical_tercile'])} (chance {v['chance']})")
    lines += ["", "7. Typicality gap of spontaneous pairs by verdict (|difference of mean log-probability|)"]
    for lbl in c.subjects:
        lines.append(f"  {lbl}: " + "; ".join(f"{k} {_fmt(f)}"
                                               for k, f in figures["7_typicality_and_method"][lbl].items()))
    return "\n".join(lines) + "\n"


def run(base: Path, out: Path | None, labels: list[Path], b: int) -> None:
    c = load(base)
    boot = Boot((c.items["domain"] + "/" + c.items["stratum"]).to_numpy(), b)
    distance = code_distances(c)
    spontaneous, counts = variation(c, boot)
    figures = {"1_accuracy": accuracy(c, boot), "2_spontaneous": spontaneous, "3_requested": requested(c, boot),
               "4_cross": cross(c, boot, counts), "5_judge": judge_quality(c, boot, distance, labels),
               "6_typicality": typicality_figures(c, boot), "7_typicality_and_method": typicality_and_method(c, boot)}
    out = out or base / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures.json").write_text(json.dumps(figures, indent=1))
    pair_table(c, distance).to_csv(out / "pairs.csv", index=False)
    text = report(figures, c)
    (out / "report.txt").write_text(text)
    print(text)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--labels", type=Path, nargs="*", default=[])
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args(argv)
    run(args.base, args.out, args.labels, args.boot)


if __name__ == "__main__":
    main()
