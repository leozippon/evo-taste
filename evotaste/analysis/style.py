"""Style diagnostics of probe responses (docs/design/protocol.md, "Diagnostics of Taste").

    python -m evotaste.analysis.style runs/<name> [--members a,b,...] [--arm plain|scaffold|perturb]
                                                 [--suite train|selection|foundation|all] [--out DIR]

Every lineage and the base model answer the same probe prompts at each probe step (one sample each). From these
responses: style features, a profile per lineage and probe step beside the base model and the seeding pools, the
between/within-member variance decomposition (sharpness), cross-suite identifiability of lineages (coherence) and
recoverability of the seeding bias on held-out lineages. Prints a compact summary and writes CSV/JSON to
results/<name>/style/ (a subset goes to a subdirectory named after it). Definitions: docs/dev/ANALYSIS_NOTES.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits

from evotaste.analysis.report import render as _table
from evotaste.analysis.rundata import SUITES, RunData

THREADS = 8  # BLAS threads: the host's CPUs are shared with running experiments

# -- features -----------------------------------------------------------------------------------------------------

ANSWER_SPAN = re.compile(r"<answer>.*?(?:</answer>|\Z)", re.S)  # an unclosed span runs to the end of the text
WORD = re.compile(r"[^\W\d_]+")
NUMBERED = re.compile(r"^(?:[#*]+\s*)?(?:step\s*\d+|\(?\d+[.)](?=\s|$))", re.I)
BULLET = re.compile(r"^[-*•+]\s+\S")
HEADER = re.compile(r"^(?:#{1,6}\s|\*\*[^*]+\*\*:?$)")
BOLD = re.compile(r"\*\*[^*\n]+\*\*")
MATH = re.compile(r"\\[A-Za-z]+|\\[()\[\]]|\$+")

# Discourse-marker lexicons: name -> (definition, pattern matched case-insensitively between word boundaries).
LEXICONS = {
    "verify": ("verification: check, verify, confirm, plug (in), substitute back, validate, satisfy, sanity",
               r"check(?:s|ed|ing)?|verif\w+|confirm\w*|plug\w*|substitut\w* back|validat\w+|satisf\w+|sanity"),
    "cases": ("case analysis: case, option, candidate, try, possibility, possible, alternatively, otherwise, "
              "either, suppose, assume",
              r"cases?|options?|candidates?|tr(?:y|ies|ied|ying)|possibilit\w+|possible|alternatively|otherwise"
              r"|either|suppose\w*|assum\w+"),
    "plan": ("planning: plan, first, next, then, step, finally, approach, strategy, goal",
             r"plan\w*|first(?:ly)?|next|then|steps?|finally|approach|strategy|goal"),
    "formal": ("formalisation: let (not let's/let me/let us), define, denote, equation, variable, unknown, formula, "
               "represent",
               r"let(?!['’]s|\s+(?:me|us)\b)|defin\w+|denot\w+|equations?|variables?|unknowns?|formulas?|represent\w*"),
    "correct": ("self-correction: wait, actually, mistake, oops, hmm, however, incorrect, wrong, re-check/"
                "reconsider/re-evaluate/recalculate/rethink, on second thought",
                r"wait|actually|mistakes?|oops|hmm+|however|incorrect|wrong"
                r"|re-?(?:check|consider|evaluate|calculate|think)\w*|on second thought"),
    "conclude": ("conclusion: therefore, thus, hence, consequently, in conclusion, final answer, (so) the answer is",
                 r"therefore|thus|hence|consequently|in conclusion|final answer|(?:so )?the (?:final )?answer is"),
    "restate": ("restating: we are given/asked, given that, the problem/question/task asks/states/says/is/gives/"
                "requires, we need/have/want to, our goal is",
                r"(?:we are|we're|you are) (?:given|asked)|given that"
                r"|the (?:problem|question|task) (?:asks|states|says|is|gives|requires)|we (?:need|have|want) to"
                r"|our goal is"),
}
_LEX = {name: re.compile(rf"\b(?:{pat})\b", re.I) for name, (_, pat) in LEXICONS.items()}


def strip_answers(text: str) -> str:
    """The text without its answer spans (closed or not) and without stray closing tags."""
    return ANSWER_SPAN.sub(" ", text).replace("</answer>", " ")


def classifier_text(text: str) -> str:
    """What the classifiers see: answer spans removed, every run of digits replaced by a single 0."""
    return re.sub(r"\d+", "0", strip_answers(text))


@dataclass(frozen=True)
class Parsed:
    text: str
    tokens: int
    body: str  # the text without answer spans
    pre: str  # text before the first answer tag (the whole text without one)
    post: str  # body text after the last closing answer tag ('' without one)
    lines: tuple[str, ...]  # non-empty body lines, stripped
    words: int  # letter-only words of the body
    chars: int  # non-whitespace characters of the body

    @classmethod
    def of(cls, text: str, tokens: int) -> Parsed:
        body = strip_answers(text)
        first, last = text.find("<answer>"), text.rfind("</answer>")
        return cls(text, tokens, body, text if first < 0 else text[:first],
                   "" if last < 0 else strip_answers(text[last + len("</answer>"):]),
                   tuple(s for line in body.splitlines() if (s := line.strip())), len(WORD.findall(body)),
                   sum(not c.isspace() for c in body))


def _per(n: float, d: float, k: float = 100.0) -> float:
    return k * n / d if d else 0.0


def _line_frac(p: Parsed, test) -> float:
    return _per(sum(bool(test(line)) for line in p.lines), len(p.lines), 1.0)


# The feature list: name -> (one-line definition, function of the parsed response). "Body" is the text without
# answer spans; "body characters" are its non-whitespace characters; "words" its letter-only words.
FEATURES = {
    "tokens": ("completion tokens (sampled ids, end-of-turn included)", lambda p: float(p.tokens)),
    "chars": ("characters of the completion", lambda p: float(len(p.text))),
    "direct": ("1 if at most 5 words precede the first answer tag (the whole text if there is none)",
               lambda p: float(len(WORD.findall(p.pre)) <= 5)),
    "pre_answer": ("fraction of the characters that precede the first answer tag (1 if there is none)",
                   lambda p: _per(len(p.pre), len(p.text), 1.0) if p.text else 1.0),
    "after_answer": ("1 if more than 5 words follow the last closing answer tag",
                     lambda p: float(len(WORD.findall(p.post)) > 5)),
    "lines": ("non-empty body lines", lambda p: float(len(p.lines))),
    "line_words": ("words per non-empty body line", lambda p: _per(p.words, len(p.lines), 1.0)),
    "numbered": ("fraction of body lines opening with a step number (1. 1) (1) Step 1, optionally after # or **)",
                 lambda p: _line_frac(p, NUMBERED.match)),
    "bullets": ("fraction of body lines opening with a bullet (- * • + and a space)",
                lambda p: _line_frac(p, BULLET.match)),
    "headers": ("fraction of body lines that are markdown headers (#) or wholly bold",
                lambda p: _line_frac(p, HEADER.match)),
    "bold": ("bold spans (**...**) per 100 body characters", lambda p: _per(len(BOLD.findall(p.body)), p.chars)),
    "math": ("LaTeX markup (\\command, \\( \\) \\[ \\], runs of $) per 100 body characters",
             lambda p: _per(len(MATH.findall(p.body)), p.chars)),
    "equals": ("= signs per 100 body characters", lambda p: _per(p.body.count("="), p.chars)),
    "code": ("1 if the body contains a code fence (```)", lambda p: float("```" in p.body)),
    "table": ("fraction of body lines containing at least two |", lambda p: _line_frac(p, lambda s: s.count("|") >= 2)),
    "digits": ("fraction of body characters that are digits",
               lambda p: _per(sum(c.isdigit() for c in p.body), p.chars, 1.0)),
    **{name: (f"{definition}; matches per 100 body words", lambda p, rx=_LEX[name]: _per(len(rx.findall(p.body)), p.words))
       for name, (definition, _) in LEXICONS.items()},
}
NAMES = list(FEATURES)
TERMINAL = ["tokens", "direct", "lines", "numbered", "bold", "math", "equals", "verify", "cases", "plan", "formal",
            "correct", "conclude"]


def features(text: str, tokens: int) -> dict[str, float]:
    p = Parsed.of(text, tokens)
    return {name: fn(p) for name, (_, fn) in FEATURES.items()}


# -- statistics ---------------------------------------------------------------------------------------------------

def variance_components(z: np.ndarray) -> dict:
    """Two-way layout without replication, z[member, prompt] = mu + a_member + b_prompt + e. Between = the member
    variance component (MS_member - MS_resid) / P, unbiased and possibly negative; within = MS_resid (residual,
    i.e. member x prompt interaction and noise); F = MS_member / MS_resid with (M-1, (M-1)(P-1)) degrees of freedom."""
    m, p = z.shape
    grand, rows, cols = z.mean(), z.mean(1), z.mean(0)
    ms_member = p * float(((rows - grand) ** 2).sum()) / (m - 1)
    ms_resid = float(((z - rows[:, None] - cols[None, :] + grand) ** 2).sum()) / ((m - 1) * (p - 1))
    between = (ms_member - ms_resid) / p
    if ms_resid == 0:
        return {"between": between, "within": 0.0, "ratio": math.nan, "F": math.nan, "p": math.nan}
    f = ms_member / ms_resid
    return {"between": between, "within": ms_resid, "ratio": between / ms_resid, "F": f,
            "p": float(stats.f.sf(f, m - 1, (m - 1) * (p - 1)))}


def accuracy_stats(correct, groups, chance: float, n_boot: int = 1000, seed: int = 0) -> dict:
    """Accuracy, its 95% interval from a bootstrap over prompts (all responses to one prompt resampled together),
    and the one-sided binomial p-value against chance (which treats responses as independent)."""
    correct = np.asarray(correct, float)
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    hits, counts = np.bincount(inv, weights=correct), np.bincount(inv)
    draws = np.random.default_rng(seed).integers(0, len(counts), size=(n_boot, len(counts)))
    lo, hi = np.quantile(hits[draws].sum(1) / counts[draws].sum(1), [0.025, 0.975])
    k, n = int(correct.sum()), len(correct)
    return {"n": n, "accuracy": k / n, "chance": chance, "ci_lo": float(lo), "ci_hi": float(hi),
            "p_binom": float(stats.binomtest(k, n, chance, alternative="greater").pvalue)}


def fit_predict(train_text, train_y, test_text, balanced: bool = False):
    """TF-IDF word 1-2-grams (punctuation runs are tokens) and character 2-4-grams, multinomial logistic regression."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, make_pipeline

    model = make_pipeline(
        FeatureUnion([("w", TfidfVectorizer(token_pattern=r"(?u)\b\w+\b|[^\w\s]+", ngram_range=(1, 2), min_df=2,
                                            sublinear_tf=True)),
                      ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True))]),
        LogisticRegression(max_iter=2000, class_weight="balanced" if balanced else None))
    model.fit(list(train_text), list(train_y))
    return model.predict(list(test_text)), model


def top_ngrams(model, k: int = 5) -> list[dict]:
    names = model[0].get_feature_names_out()
    lr = model[-1]
    coef = lr.coef_ if len(lr.classes_) > 2 else np.vstack([-lr.coef_[0], lr.coef_[0]])
    return [{"lineage": c, "rank": i + 1, "ngram": names[j], "coef": float(row[j])}
            for c, row in zip(lr.classes_, coef) for i, j in enumerate(np.argsort(row)[::-1][:k])]


# -- analyses -----------------------------------------------------------------------------------------------------

def _cube(df: pd.DataFrame) -> tuple[list[str], list[str], np.ndarray]:
    """Lineages, uids and the array x[lineage, uid, feature]; every lineage must have answered the same prompts."""
    df = df.sort_values(["lineage", "uid"])
    lineages = list(dict.fromkeys(df["lineage"]))
    uids = sorted(set(df["uid"]))
    counts = df.groupby("lineage")["uid"].agg(lambda u: (len(u), u.nunique()))
    if any(c != (len(uids), len(uids)) for c in counts):
        raise ValueError(f"lineages did not answer the same probe prompts exactly once: {dict(counts)}")
    return lineages, uids, df[NAMES].to_numpy(float).reshape(len(lineages), len(uids), len(NAMES))


def sharpness(members: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Variance components per probe step, suite (and all suites) and feature, in units of the base model's
    across-prompt standard deviation; the row with feature 'mean' averages the components over features."""
    rows = []
    for step, at in members.groupby("step"):
        if at["lineage"].nunique() < 2:
            continue
        for suite in ("all",) + SUITES:
            sub = at if suite == "all" else at[at["suite"] == suite]
            if sub.empty:
                continue
            lineages, uids, x = _cube(sub)
            sd = base.set_index("uid").loc[uids, NAMES].std(ddof=1).to_numpy()
            comps = []
            for j, name in enumerate(NAMES):
                c = (variance_components(x[:, :, j] / sd[j]) if sd[j] > 0 else
                     {"between": math.nan, "within": math.nan, "ratio": math.nan, "F": math.nan, "p": math.nan})
                comps.append(c)
                rows.append({"step": step, "suite": suite, "feature": name, "members": len(lineages),
                             "prompts": len(uids), **c})
            ok = [c for c in comps if not math.isnan(c["between"])]
            if not ok:  # no feature varies across the base model's answers to these prompts
                continue
            b, w = np.mean([c["between"] for c in ok]), np.mean([c["within"] for c in ok])
            rows.append({"step": step, "suite": suite, "feature": "mean", "members": len(lineages),
                         "prompts": len(uids), "between": b, "within": w, "ratio": b / w, "F": math.nan,
                         "p": math.nan, "features": len(ok), "significant": sum(c["p"] < 0.01 for c in ok)})
    return pd.DataFrame(rows)


def identify(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """Fit a lineage classifier on `train` responses and predict `test` responses (same lineages in both)."""
    pred, model = fit_predict(train["ctext"], train["lineage"], test["ctext"])
    return test.assign(pred=pred), model


def identifiability(members: pd.DataFrame, bias_of: dict[str, str]):
    """Cross-suite lineage identification per probe step in both directions, and across steps (first -> last).
    Returns the accuracy rows, the confusion counts aggregated by bias, and the top n-grams per lineage."""
    rows, confusion, ngrams = [], [], []
    heldout = ("selection", "foundation")

    def score(pred: pd.DataFrame, **keys):
        """Accuracy on each test suite and, with several, on all of them together."""
        parts = [(s, pred[pred["suite"] == s]) for s in SUITES if (pred["suite"] == s).any()]
        if len(parts) > 1:
            parts.append(("+".join(s for s, _ in parts), pred))
        for name, part in parts:
            rows.append({**keys, "test_suite": name, "members": pred["lineage"].nunique(),
                         **accuracy_stats(part["pred"] == part["lineage"], part["uid"],
                                          1 / pred["lineage"].nunique())})

    steps = sorted(members["step"].unique())
    for step in steps:
        at = members[members["step"] == step]
        if at["lineage"].nunique() < 2:
            continue
        fwd, model = identify(at[at["suite"] == "train"], at[at["suite"].isin(heldout)])
        score(fwd, train_step=step, test_step=step, train_suite="train")
        ngrams += [{"step": step, **r} for r in top_ngrams(model)]
        confusion += [{"step": step, "true": bias_of[t], "pred": bias_of[p], "count": n}
                      for (t, p), n in fwd.groupby(["lineage", "pred"]).size().items()]
        rev, _ = identify(at[at["suite"].isin(heldout)], at[at["suite"] == "train"])
        score(rev, train_step=step, test_step=step, train_suite="+".join(heldout))
    if len(steps) > 1:
        first, last = members[members["step"] == steps[0]], members[members["step"] == steps[-1]]
        common = set(first["lineage"]) & set(last["lineage"])
        if len(common) >= 2:
            first, last = first[first["lineage"].isin(common)], last[last["lineage"].isin(common)]
            across, _ = identify(first[first["suite"] == "train"], last[last["suite"].isin(heldout)])
            score(across, train_step=steps[0], test_step=steps[-1], train_suite="train")
    confusion = pd.DataFrame(confusion, columns=["step", "true", "pred", "count"])
    confusion = confusion.groupby(["step", "true", "pred"], as_index=False)["count"].sum()
    return pd.DataFrame(rows), confusion, pd.DataFrame(ngrams)


def scaffold_recovery(members: pd.DataFrame, bias_of: dict[str, str]) -> pd.DataFrame:
    """Per probe step: predict the bias of held-out lineages from a classifier trained on the other lineages of
    the biases that have at least two lineages; fold k holds out lineage k mod n_b of every such bias b."""
    rows = []
    for step, at in members.groupby("step"):
        groups: dict[str, list[str]] = {}
        for lineage in dict.fromkeys(at["lineage"]):
            groups.setdefault(bias_of[lineage], []).append(lineage)
        usable = {b: ls for b, ls in groups.items() if len(ls) >= 2}
        row = {"step": step, "classes": " ".join(f"{b}({len(ls)})" for b, ls in usable.items()),
               "excluded": " ".join(f"{b}({len(ls)})" for b, ls in groups.items() if len(ls) < 2)}
        if len(usable) < 2:
            rows.append({**row, "note": "not possible (needs two biases with two lineages each)"})
            continue
        folds = max(len(ls) for ls in usable.values())
        keep = at[at["lineage"].map(bias_of).isin(usable)]
        preds = []
        for k in range(folds):
            held = {ls[k % len(ls)] for ls in usable.values()}
            train, test = keep[~keep["lineage"].isin(held)], keep[keep["lineage"].isin(held)]
            pred, _ = fit_predict(train["ctext"], train["lineage"].map(bias_of), test["ctext"], balanced=True)
            preds.append(test.assign(pred=pred, fold=k))
        pred = pd.concat(preds)
        truth = pred["lineage"].map(bias_of)
        rows.append({**row, "folds": folds, **accuracy_stats(pred["pred"] == truth,
                                                             pred["fold"].astype(str) + "/" + pred["uid"],
                                                             1 / len(usable))})
    return pd.DataFrame(rows)


# -- loading ------------------------------------------------------------------------------------------------------

def response_frame(run: RunData, lineages) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-response features of the base model's probe answers, the seeding pools and the lineages' probes."""
    base_records = run.base_responses()
    if base_records is None:
        raise SystemExit(f"{run.dir}: the base model's probe responses are not written yet")

    def frame(records, **meta):
        return pd.DataFrame([{**meta, "step": r.get("step"), "suite": r.get("suite", "seeding"), "uid": r["uid"],
                              "question": (hashlib.sha1(r["question"].encode()).hexdigest()[:16] if "question" in r
                                           else None),
                              "family": r["family"], "reward": r.get("reward", 1.0), "ctext": classifier_text(r["text"]),
                              **features(r["text"], r["completion_tokens"] if "completion_tokens" in r
                                         else len(r["completion_ids"]))} for r in records])

    base = frame(base_records, source="base", lineage="base", bias="base", teacher="")
    pools = [frame(run.pool_records(t), source="pool", lineage=f"pool {t}", bias="", teacher=t)
             for t in run.teachers()]
    members = [frame(lin.probe_responses(step), source="member", lineage=lin.id, bias=lin.bias, teacher=lin.teacher)
               for lin in lineages for step in lin.probe_steps()]
    pools = pd.concat(pools, ignore_index=True) if pools else base.iloc[:0]
    members = pd.concat(members, ignore_index=True) if members else base.iloc[:0]
    return base, pools, members


def align(members: pd.DataFrame, base: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    """Keep, at each probe step, the prompts that every lineage present and the base model answered with the same
    uid and the identical question text; return the kept responses and the dropped uids per step."""
    base_keys = set(zip(base["uid"], base["question"]))
    kept, dropped = [], {}
    for step, at in members.groupby("step"):
        lineages = at.groupby(["uid", "question"])["lineage"].nunique()
        common = {k for k, n in lineages.items() if n == at["lineage"].nunique() and k in base_keys}
        mask = np.array([k in common for k in zip(at["uid"], at["question"])])
        kept.append(at[mask])
        dropped[int(step)] = sorted(set(at.loc[~mask, "uid"]))
    return (pd.concat(kept) if kept else members), dropped


def profile(base: pd.DataFrame, pools: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Mean of every feature per source, lineage, probe step and suite (and over all suites)."""
    df = pd.concat([base, pools, members], ignore_index=True)
    keys = ["source", "lineage", "bias", "teacher", "step"]
    by_suite = df.groupby(keys + ["suite"], dropna=False, sort=False)
    out = [by_suite[NAMES].mean().join(by_suite.size().rename("n")).reset_index()]
    probes = df[df["suite"] != "seeding"]
    by_all = probes.groupby(keys, dropna=False, sort=False)
    out.append(by_all[NAMES].mean().join(by_all.size().rename("n")).reset_index().assign(suite="all"))
    return pd.concat(out, ignore_index=True)[keys + ["suite", "n"] + NAMES]


# -- terminal -----------------------------------------------------------------------------------------------------

def _g(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.0f}" if abs(x) >= 100 else f"{x:.2f}" if abs(x) >= 0.995 or x == 0 else f"{x:.3f}".lstrip("0")


def summary_lines(run, lineages, label, prof, sharp, ident, confusion, scaff, suite, teachers, dropped) -> list[str]:
    steps = sorted(prof.loc[prof["source"] == "member", "step"].dropna().astype(int).unique())
    lines = [f"style {run.name} ({label}): {len(lineages)} lineages with probes, probe steps "
             f"{', '.join(map(str, steps)) or 'none'}; {len(NAMES)} features"]
    if any(dropped.values()):
        lines.append("prompts dropped (not answered with the identical question by every lineage and the base): "
                     + "; ".join(f"step {s}: {len(u)} ({', '.join(sorted({x.split('/')[0] for x in u}))})"
                                 for s, u in dropped.items() if u))
    # Profile: base, each teacher pool with the lineages seeded from it at their first and last probe step.
    p = prof[(prof["suite"] == suite) | (prof["source"] == "pool")]
    lines.append(f"profile: means on the {suite} suite (pools: correct seeding samples, train-family prompts)")
    rows = []

    def add(name, step, sel):
        r = p[sel]
        if not r.empty:
            rows.append([name, "-" if step is None else str(step)] + [_g(r.iloc[0][f]) for f in TERMINAL])

    add("base", None, p["source"] == "base")
    for t in teachers + [t for t in dict.fromkeys(lin.teacher for lin in lineages) if t not in teachers]:
        add(f"pool {t}", None, p["lineage"] == f"pool {t}")
        for lin in (lin for lin in lineages if lin.teacher == t):
            mine = sorted(p.loc[p["lineage"] == lin.id, "step"].dropna().astype(int))
            for step in sorted({mine[0], mine[-1]}) if mine else []:
                add(f"  {lin.id}", step, (p["lineage"] == lin.id) & (p["step"] == step))
    lines += _table(["source", "step"] + TERMINAL, rows, left=1)

    lines.append("")
    if sharp.empty:
        lines.append("sharpness: needs two lineages with a probe at the same step")
    else:
        lines.append("sharpness, all suites, base-SD units: between = lineage variance component, within = residual")
        m = sharp[(sharp["suite"] == "all") & (sharp["feature"] == "mean")]
        lines += _table(["step", "lineages", "prompts", "between", "within", "ratio", "p<.01"],
                        [[int(r.step), r.members, r.prompts, f"{r.between:.4f}", f"{r.within:.3f}", f"{r.ratio:.4f}",
                          f"{r.significant:.0f}/{r.features:.0f}"] for r in m.itertuples()], left=0)
        last = sharp[(sharp["suite"] == "all") & (sharp["feature"] != "mean") & (sharp["step"] == m["step"].max())]
        top = last.dropna(subset=["ratio"]).nlargest(6, "ratio")
        lines.append(f"largest ratios at step {int(m['step'].max())}: "
                     + ", ".join(f"{r.feature} {r.ratio:.3f}" for r in top.itertuples()))
        flat = last[last["between"].isna()]["feature"].tolist()
        if flat:
            lines.append(f"  not standardisable (no base spread): {', '.join(flat)}")

    lines.append("")
    if ident.empty:
        lines.append("identifiability: needs two lineages with a probe at the same step")
    else:
        lines.append("identifiability: predict the lineage across suites; 95% CI from a bootstrap over prompts")
        lines += _table(["train", "test", "lineages", "n", "acc", "chance", "95% CI", "p"],
                        [[f"{r.train_suite}@{r.train_step}", f"{r.test_suite}@{r.test_step}", r.members, r.n,
                          f"{r.accuracy:.3f}", f"{r.chance:.3f}", f"{r.ci_lo:.3f}-{r.ci_hi:.3f}", f"{r.p_binom:.0e}"]
                         for r in ident.itertuples() if "+" not in r.test_suite], left=2)
        step = confusion["step"].max()
        c = confusion[confusion["step"] == step].pivot_table(index="true", columns="pred", values="count",
                                                              aggfunc="sum", fill_value=0)
        order = [b for b in dict.fromkeys(lin.bias for lin in lineages) if b in c.index]
        c = c.reindex(index=order, columns=order, fill_value=0)
        lines.append(f"confusion by bias, step {int(step)}, train -> selection+foundation (row %, rows true)")
        lines += _table(["true \\ pred"] + order,
                        [[t] + [f"{100 * c.loc[t, q] / c.loc[t].sum():.0f}" for q in order] for t in order])

    lines.append("")
    lines.append("bias recoverability on held-out lineages (all suites)" + ("" if len(scaff) else ": no probes yet"))
    for r in scaff.itertuples():
        if isinstance(getattr(r, "note", None), str):
            lines.append(f"  step {int(r.step)}: {r.note}; lineages per bias: {' '.join([r.classes, r.excluded]).strip()}")
        else:
            lines.append(f"  step {int(r.step)}: {r.classes}; acc {r.accuracy:.3f} chance {r.chance:.3f} "
                         f"CI {r.ci_lo:.3f}-{r.ci_hi:.3f} p {r.p_binom:.0e} (n {r.n}, {r.folds} folds"
                         + (f"; excluded {r.excluded})" if r.excluded else ")"))
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--members", help="comma-separated lineage ids to restrict the analysis to")
    ap.add_argument("--arm", choices=["plain", "scaffold", "perturb"], help="restrict the analysis to one arm")
    ap.add_argument("--suite", default="train", choices=["all", *SUITES], help="suite of the terminal profile")
    ap.add_argument("--out", help="output directory (default results/<name>/style[/<subset>])")
    args = ap.parse_args(argv)
    run = RunData(args.run_dir)
    lineages = [lin for lin in run.lineages if lin.probe_steps()]
    label = []
    if args.members:
        wanted = args.members.split(",")
        unknown = set(wanted) - {lin.id for lin in run.lineages}
        if unknown:
            raise SystemExit(f"unknown lineages: {sorted(unknown)}")
        lineages = [lin for lin in lineages if lin.id in wanted]
        label.append("members-" + "+".join(wanted))
    if args.arm:
        lineages = [lin for lin in lineages if lin.arm == args.arm]
        label.append(f"arm-{args.arm}")
    label = "_".join(label) or "all"
    out = Path(args.out or f"results/{run.name}/style" + ("" if label == "all" else f"/{label}"))
    out.mkdir(parents=True, exist_ok=True)

    base, pools, members = response_frame(run, lineages)
    members, dropped = align(members, base)
    bias_of = {lin.id: lin.bias for lin in lineages}
    prof = profile(base, pools, members)
    with threadpool_limits(THREADS):
        sharp = sharpness(members, base)
        ident, confusion, ngrams = identifiability(members, bias_of) if len(members) else (
            pd.DataFrame(), pd.DataFrame(columns=["step", "true", "pred", "count"]), pd.DataFrame())
        scaff = scaffold_recovery(members, bias_of) if len(members) else pd.DataFrame()

    pd.concat([base, pools, members]).drop(columns=["ctext", "question"]).to_csv(out / "features.csv.gz", index=False)
    prof.to_csv(out / "profile.csv", index=False)
    sharp.to_csv(out / "sharpness.csv", index=False)
    ident.to_csv(out / "identifiability.csv", index=False)
    confusion.to_csv(out / "confusion.csv", index=False)
    ngrams.to_csv(out / "ngrams.csv", index=False)
    scaff.to_csv(out / "scaffold.csv", index=False)
    def records(df):  # NaN -> null, numpy scalars -> JSON numbers
        return json.loads(df.to_json(orient="records"))

    summary = {"run": run.name, "subset": label, "lineages": {lin.id: lin.bias for lin in lineages},
               "features": {name: definition for name, (definition, _) in FEATURES.items()},
               "dropped_prompts": dropped,
               "sharpness": records(sharp[sharp["feature"] == "mean"]) if len(sharp) else [],
               "identifiability": records(ident), "scaffold": records(scaff)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(summary_lines(run, lineages, label, prof, sharp, ident, confusion, scaff, args.suite,
                                  run.teachers(), dropped)))
    print(f"\nCSV/JSON: {out}/")


if __name__ == "__main__":
    main()
