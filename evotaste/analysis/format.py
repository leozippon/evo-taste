"""Format-lenient re-scoring of final evaluations: how much of the difference between adapters is answer format.

    python -m evotaste.analysis.format runs/<name> (--suite selection|train | --blind blind1|blind2) --tag T
                                                   [--selection-tag S] [--processes 16] [--out DIR]

Every parsed, untruncated, wrong record's extracted answer is rewritten by a fixed list of meaning-preserving
transforms and re-scored by its family's own scorer (the engine's `raw_score` on the regenerated instance); a numeric
or symbolic equality with the gold also counts. A transform that changes some gold answer or correct answer of a
family into one that family's scorer rejects is disabled for that family. Reports strict and lenient macro accuracy
and their difference (format loss) per adapter and family, the decomposition of between-member variance, style
features against outcome, and with --blind and --selection-tag the selection-blind comparison. A blind vault is read
only with --blind and an existing harvest manifest. Writes CSV/JSON to results/<name>/format/<suite>/. Definitions:
docs/dev/ANALYSIS_NOTES.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import multiprocessing as mp
import re
import signal
import time
import warnings
from decimal import Decimal, InvalidOperation
from multiprocessing.connection import wait
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from evotaste.analysis.harvest import (PERMUTATIONS, Suite, adapter_roles, ci, compare_outlier, correlation,
                                       median_member, pick_outlier, spread)
from evotaste.analysis.report import render
from evotaste.analysis.rundata import PROCEDURAL_UID, VAULTS, Evaluation, load_evaluation, member_specs
from evotaste.tasks import REWARD_THRESHOLD, Task, procedural_task, raw_score

CALL_TIMEOUT_S = 5  # one scorer or sympy call inside a worker (SIGALRM)
JOB_TIMEOUT_S = 60  # one answer's whole re-scoring; past it the worker process is killed and replaced
MAX_EXPR_CHARS = 400  # longer answers are not parsed symbolically

# -- transforms ---------------------------------------------------------------------------------------------------

_WRAPPERS = (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]"), ("**", "**"), ("`", "`"), ('"', '"'),
             ("'", "'"))
_COMMAND = re.compile(r"\\(?:boxed|text|textbf|textrm|mathrm|mathbf)\{(.*)\}", re.S)
_SUPERSCRIPTS = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+")
_SUPER = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_FUNCTIONS = r"(?:sin|cos|tan|cot|sec|csc|asin|acos|atan|sinh|cosh|tanh|exp|log|ln|sqrt|pi|Integral)"
_NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d*)?(?:[eE][+-]?\d{1,3})?")


def _braced(s: str, i: int) -> tuple[str, int] | None:
    """Content of the {...} group opening at s[i] and the index after it."""
    if i >= len(s) or s[i] != "{":
        return None
    depth = 0
    for j in range(i, len(s)):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return s[i + 1:j], j + 1
    return None


def _command(s: str, name: str, arity: int, build) -> str:
    """Replace \\name{a}{b}... by build(a, b, ...) wherever its brace groups are complete."""
    out, i = "", 0
    for m in re.finditer(rf"\\{name}(?![A-Za-z])", s):
        if m.start() < i:
            continue
        args, j = [], m.end()
        for _ in range(arity):
            g = _braced(s, j)
            if g is None:
                break
            args.append(g[0])
            j = g[1]
        if len(args) == arity:
            out, i = out + s[i:m.start()] + build(*args), j
    return out + s[i:]


def t_strip(s: str, gold: str) -> str:
    """Surrounding whitespace, a trailing period, and wrappers around the whole answer: $ $$ \\( \\) \\[ \\] ** ` and
    quotes (only when the delimiter does not occur inside), \\boxed{} \\text{} \\textbf{} \\mathrm{} \\mathbf{}."""
    prev = None
    while s != prev:
        prev = s
        s = s.strip()
        if len(s) > 1 and s.endswith("."):
            s = s[:-1]
        for a, b in _WRAPPERS:
            inner = s[len(a):len(s) - len(b)]
            if len(s) >= len(a) + len(b) and s.startswith(a) and s.endswith(b) and a not in inner and b not in inner:
                s = inner
        m = _COMMAND.fullmatch(s)
        if m and _braced("{" + m.group(1) + "}", 0) == (m.group(1), len(m.group(1)) + 2):
            s = m.group(1)
    return s


def t_latex(s: str, gold: str) -> str:
    """LaTeX to plain notation: \\frac \\dfrac \\tfrac{a}{b} -> (a)/(b), \\sqrt{a} -> sqrt(a), \\cdot \\times × · -> *,
    \\left \\right and spacing commands dropped, \\pi and function commands lose the backslash, remaining braces become
    parentheses, the unicode minus becomes -."""
    s = s.replace("−", "-")
    s = re.sub(r"\\left|\\right|\\displaystyle|\\[,;!: ]", "", s)
    for name in ("frac", "dfrac", "tfrac"):
        s = _command(s, name, 2, lambda a, b: f"({a})/({b})")
    s = _command(s, "sqrt", 1, lambda a: f"sqrt({a})")
    s = re.sub(r"\\(?:cdot|times)(?![A-Za-z])|[×·]", "*", s)
    s = re.sub(rf"\\({_FUNCTIONS[3:-1]})(?![A-Za-z])", r"\1", s)
    return s.replace("{", "(").replace("}", ")")


def t_exponent(s: str, gold: str) -> str:
    """Exponent notation: unicode superscripts -> **n (x⁻² -> x**-2), ^ -> **."""
    return _SUPERSCRIPTS.sub(lambda m: "**" + m.group(0).translate(_SUPER), s).replace("^", "**")


def t_implicit(s: str, gold: str) -> str:
    """Implicit multiplication made explicit: a digit before a letter or ( (3x, 2(x+1); not 2e-3), ) before a letter,
    digit or ( , a single-letter variable before ( , and whitespace between two factors (x sin(x))."""
    s = re.sub(r"(\d)\s*(?=[A-Za-z(])(?![eE][+-]?\d)", r"\1*", s)
    s = re.sub(r"\)\s*(?=[\w(])", ")*", s)
    s = re.sub(r"(?<![\w.])([A-Za-z])\s*\(", r"\1*(", s)
    return re.sub(r"(?<=[\w)])\s+(?=[A-Za-z(])", "*", s)


def number(s: str) -> Decimal | None:
    """The value of a string that is exactly one decimal number (thousands separators, sign, exponent allowed)."""
    s = s.strip()
    if not s or len(s) > 60 or not _NUMBER.fullmatch(s) or not re.search(r"\d", s):
        return None
    try:
        return Decimal(s.replace(",", ""))
    except InvalidOperation:
        return None


def t_numeric(s: str, gold: str) -> str:
    """A lone number in canonical decimal form: thousands separators and + sign dropped, trailing zeros and point
    dropped, integer values without a fraction (237.0 -> 237, 1,234.50 -> 1234.5, .5 -> 0.5, 1e3 -> 1000)."""
    v = number(s)
    if v is None:
        return s
    if v == v.to_integral_value():
        return str(int(v))
    return format(v.normalize(), "f")


def t_case_space(s: str, gold: str) -> str:
    """Lower case and single spaces."""
    return re.sub(r"\s+", " ", s.strip()).lower()


def _list_style(gold: str):
    """(open, close, quote, separator) of a flat list gold answer: bracketed, or two or more comma-separated items."""
    g = gold.strip()
    if g.startswith("[") and g.endswith("]") and "[" not in g[1:-1]:
        body, opening, closing = g[1:-1], "[", "]"
    elif "," in g and "[" not in g and "(" not in g:
        body, opening, closing = g, "", ""
    else:
        return None
    items = [i.strip() for i in body.split(",")]
    if len(items) < 2 and opening == "":
        return None
    quote = items[0][0] if items and items[0][:1] in ("'", '"') else ""
    return opening, closing, quote, ", " if ", " in body else ","


def t_list(s: str, gold: str) -> str:
    """When the gold is a flat list: the answer's items (split at commas, semicolons or newlines, or at spaces when
    it has none and the gold's items contain no space), stripped of brackets and quotes, rewritten in the gold's
    brackets, quoting and separator."""
    style = _list_style(gold)
    if style is None:
        return s
    opening, closing, quote, sep = style
    a = s.strip()
    if a[:1] in "[({" and a[-1:] in "])}":
        a = a[1:-1]
    if re.search(r"[,;\n]", a):
        items = re.split(r"\s*[,;\n]\s*", a)
    elif not any(" " in i.strip().strip("'\"") for i in gold.strip("[]").split(",")):
        items = a.split()
    else:
        items = [a]
    items = [i.strip().strip("'\"").strip() for i in items if i.strip()]
    return opening + sep.join(f"{quote}{i}{quote}" for i in items) + closing


# Applied singly to the raw answer and, in this order, as one chain ("combined").
TRANSFORMS = {"strip": t_strip, "latex": t_latex, "exponent": t_exponent, "implicit": t_implicit,
              "numeric": t_numeric, "case_space": t_case_space, "list": t_list}
CRITERIA = ("numeric_equal", "symbolic_equal")  # tested on the combined form after the transforms
ORDER = (*TRANSFORMS, "combined", *CRITERIA)  # the first of these that rescues a record is credited with it


def chain(s: str, gold: str, safe) -> str:
    for name, fn in TRANSFORMS.items():
        if name in safe:
            s = fn(s, gold)
    return s


def numeric_equal(a: str, gold: str) -> bool:
    """Both are single numbers of equal value."""
    x, g = number(a), number(gold)
    return x is not None and g is not None and x == g


_EXPR = re.compile(r"^[\w\s+\-*/().,]*$")


def symbolic_applicable(a: str, gold: str) -> bool:
    """Both look like plain expressions and the gold contains a letter and an operator, digit or parenthesis."""
    def plain(x):
        return (len(x) <= MAX_EXPR_CHARS and _EXPR.match(x) and "__" not in x
                and not re.search(r"[A-Za-z_)\]]\s*\.", x) and not re.search(r"\*\*\s*\(?\s*-?\d{3,}", x))
    return bool(plain(a) and plain(gold) and re.search(r"[A-Za-z]", gold) and re.search(r"[\d+\-*/()]", gold))


def symbolic_equal(a: str, gold: str) -> bool:
    """Sympy: structurally equal after parsing (so an unexpanded product never equals an expanded gold), or, for a
    gold antiderivative ending in + C, equal derivatives in the gold's variable. A standalone e is Euler's number
    unless the gold uses e as a variable."""
    import sympy

    names = {"C": sympy.Symbol("C")}
    if not re.search(r"\be\b", gold):
        names["e"] = sympy.E
    g = sympy.parse_expr(gold, local_dict=dict(names))
    x = sympy.parse_expr(a, local_dict=dict(names))
    free = g.free_symbols - {names["C"]}
    if not free:
        return False
    if re.search(r"\+\s*C\s*$", gold):
        if len(free) != 1:
            return False
        v = free.pop()
        return sympy.simplify(sympy.diff(x, v) - sympy.diff(g, v)) == 0
    return bool(x == g)


# -- worker processes ---------------------------------------------------------------------------------------------

class _CallTimeout(Exception):
    pass


def _alarm(*_):
    raise _CallTimeout


def _bounded(fn, *args):
    """fn(*args) within CALL_TIMEOUT_S; a timeout or an exception gives None."""
    signal.alarm(CALL_TIMEOUT_S)
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001 - arbitrary model text: any failure means "not rescued"
        return None
    finally:
        signal.alarm(0)


class GoldMismatch(Exception):
    """A regenerated instance does not carry the gold answer of its record: the analysis would score the wrong task."""


@functools.lru_cache(maxsize=None)
def make_task(uid: str, family: str, gold: str) -> Task:
    """The instance behind a record: regenerated from its uid (procedural) or built from the gold (external)."""
    if PROCEDURAL_UID.match(uid):
        fam, seed, index = uid.split("/")
        task = procedural_task(fam, int(seed), int(index))
        if task.answer != gold:
            raise GoldMismatch(f"{uid}: regenerated gold {task.answer!r} differs from the record's {gold!r}")
        return task
    return Task(uid, family, "", gold)


def _accepts(task: Task, form: str) -> bool:
    return (_bounded(raw_score, task, form) or 0.0) >= REWARD_THRESHOLD


def guard_job(job) -> dict[str, bool]:
    """For a gold or correct answer: whether each transform (and the chain of the given safe ones) that changes it
    leaves it accepted by the family's scorer."""
    uid, family, gold, text, safe = job
    task = make_task(uid, family, gold)
    out = {}
    for name, fn in TRANSFORMS.items():
        if safe is None and (form := fn(text, gold)) != text:
            out[name] = _accepts(task, form)
    if safe is not None and (form := chain(text, gold, safe)) != text:
        out["combined"] = _accepts(task, form)
    return out


def rescue_job(job) -> tuple[str | None, str]:
    """The first entry of ORDER that makes a wrong answer correct, and the form that did."""
    uid, family, gold, answer, safe = job
    task = make_task(uid, family, gold)
    tried = {answer}
    for name, fn in TRANSFORMS.items():
        form = fn(answer, gold)
        if name in safe and form not in tried:
            tried.add(form)
            if _accepts(task, form):
                return name, form
    form = chain(answer, gold, safe)
    if "combined" in safe and form not in tried and _accepts(task, form):
        return "combined", form
    if numeric_equal(form, gold):
        return "numeric_equal", form
    if symbolic_applicable(form, gold) and _bounded(symbolic_equal, form, gold):
        return "symbolic_equal", form
    return None, form


_WORK = {"guard": guard_job, "rescue": rescue_job}


def _serve(conn) -> None:
    signal.signal(signal.SIGALRM, _alarm)
    warnings.simplefilter("ignore")  # sympy's parser warns about every odd model answer
    while True:
        try:
            kind, job = conn.recv()
        except EOFError:
            return
        try:
            conn.send((_WORK[kind](job), None))
        except Exception as e:  # noqa: BLE001 - reported per job; a gold mismatch aborts in the parent
            conn.send((None, f"{type(e).__name__}: {e}"[:300]))


class _Worker:
    def __init__(self, ctx):
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_serve, args=(child,), daemon=True)
        self.proc.start()
        child.close()

    def kill(self) -> None:
        self.proc.kill()
        self.proc.join()
        self.conn.close()


def run_jobs(kind: str, jobs: list, processes: int) -> list[tuple[object, str | None]]:
    """(result, error) per job, computed by worker processes; a job past JOB_TIMEOUT_S gives (None, "timeout")."""
    if not jobs:
        return []
    ctx = mp.get_context("fork")
    workers = [_Worker(ctx) for _ in range(min(processes, len(jobs)))]
    out: list = [None] * len(jobs)
    todo = list(range(len(jobs)))[::-1]
    busy: dict[int, tuple[int, float]] = {}
    try:
        while todo or busy:
            for w in range(len(workers)):
                if w not in busy and todo:
                    i = todo.pop()
                    workers[w].conn.send((kind, jobs[i]))
                    busy[w] = (i, time.monotonic())
            deadline = min(t0 for _, t0 in busy.values()) + JOB_TIMEOUT_S
            ready = wait([workers[w].conn for w in busy], timeout=max(0.0, deadline - time.monotonic()))
            for w in [w for w in busy if workers[w].conn in ready]:
                i, _ = busy.pop(w)
                try:
                    out[i] = workers[w].conn.recv()
                except (EOFError, OSError):
                    out[i] = (None, "worker died")
                    workers[w].kill()
                    workers[w] = _Worker(ctx)
            now = time.monotonic()
            for w in [w for w, (_, t0) in busy.items() if now - t0 >= JOB_TIMEOUT_S]:
                i, _ = busy.pop(w)
                out[i] = (None, "timeout")
                workers[w].kill()
                workers[w] = _Worker(ctx)
    finally:
        for w in workers:
            w.kill()
    return out


# -- re-scoring ---------------------------------------------------------------------------------------------------

def guard(ev: Evaluation, processes: int) -> tuple[dict[str, set[str]], pd.DataFrame]:
    """Safe transforms per family: those that never turn a gold answer, or an answer the evaluation scored correct,
    into one the family's scorer rejects; then the same for their chain ("combined")."""
    texts = {(u, f, g, g) for u, f, g in zip(ev.uids, ev.families, ev.gold)}
    correct = np.argwhere(ev.reward == 1)
    texts |= {(ev.uids[i], ev.families[i], ev.gold[i], ev.extracted[a, i, s]) for a, i, s in correct}
    texts = sorted(texts)
    rows: dict[tuple[str, str], list[int]] = {}

    def tally(results, jobs):
        for (res, err), job in zip(results, jobs):
            if err is not None and err.startswith("GoldMismatch"):
                raise SystemExit(err)
            for name, ok in (res or {}).items():
                c = rows.setdefault((job[1], name), [0, 0, 0])
                c[0] += 1
                c[1] += not ok
            if err is not None:
                rows.setdefault((job[1], "error"), [0, 0, 0])[2] += 1

    jobs = [(*t, None) for t in texts if any(fn(t[3], t[2]) != t[3] for fn in TRANSFORMS.values())]
    tally(run_jobs("guard", jobs, processes), jobs)
    families = sorted(set(ev.families))
    safe = {f: {n for n in TRANSFORMS if rows.get((f, n), [0, 0])[1] == 0} for f in families}
    jobs = [(*t, frozenset(safe[t[1]])) for t in texts if chain(t[3], t[2], safe[t[1]]) != t[3]]
    tally(run_jobs("guard", jobs, processes), jobs)
    for f in families:
        if rows.get((f, "combined"), [0, 0])[1] == 0:
            safe[f].add("combined")
    table = pd.DataFrame([{"family": f, "transform": n, "changed": c[0], "violations": c[1], "errors": c[2],
                           "disabled": c[1] > 0} for (f, n), c in sorted(rows.items())],
                         columns=["family", "transform", "changed", "violations", "errors", "disabled"])
    return safe, table


def rescore(ev: Evaluation, processes: int) -> tuple[Evaluation, pd.DataFrame, dict]:
    """The lenient evaluation (reward 1 also for rescued records) and one row per distinct rescued answer."""
    safe, guard_table = guard(ev, processes)
    wrong = np.argwhere((ev.reward == 0) & ~ev.truncated & ~pd.isna(ev.extracted))
    distinct = sorted({(ev.uids[i], ev.families[i], ev.gold[i], ev.extracted[a, i, s]) for a, i, s in wrong})
    jobs, quick = [], {}
    for u, f, g, x in distinct:
        if any(n in safe[f] and fn(x, g) != x for n, fn in TRANSFORMS.items()) or symbolic_applicable(chain(
                x, g, safe[f]), g):
            jobs.append((u, f, g, x, frozenset(safe[f])))
        else:
            quick[(u, x)] = ("numeric_equal" if numeric_equal(x, g) else None, x)
    results = run_jobs("rescue", jobs, processes)
    outcome = dict(quick)
    errors = {"timeout": 0, "other": 0}
    for job, (res, err) in zip(jobs, results):
        if err is not None:
            if err.startswith("GoldMismatch"):
                raise SystemExit(err)
            errors["timeout" if err == "timeout" else "other"] += 1
        outcome[(job[0], job[3])] = res if res is not None else (None, job[3])
    lenient = ev.reward.copy()
    by = np.full(ev.reward.shape, None, dtype=object)
    for a, i, s in wrong:
        name, _ = outcome[(ev.uids[i], ev.extracted[a, i, s])]
        if name is not None:
            lenient[a, i, s], by[a, i, s] = 1.0, name
    rescued = pd.DataFrame([{"uid": u, "family": f, "gold": g, "extracted": x, "transform": outcome[(u, x)][0],
                             "form": outcome[(u, x)][1]} for u, f, g, x in distinct if outcome[(u, x)][0]])
    info = {"records": int(ev.reward.size), "wrong_parsed": len(wrong), "distinct_wrong": len(distinct),
            "jobs": len(jobs), "errors": errors, "guard": guard_table, "safe": safe, "by": by}
    return dataclasses.replace(ev, reward=lenient), rescued, info


# -- statistics ---------------------------------------------------------------------------------------------------

def pair(strict: Evaluation, lenient: Evaluation, seed: int) -> tuple[Suite, Suite]:
    s, l = Suite(strict, seed), Suite(lenient, seed)
    assert all(np.array_equal(s.weights[f], l.weights[f]) for f in s.groups)  # paired draws
    return s, l


def adapter_table(s: Suite, l: Suite, roles: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict, lenient and loss (lenient - strict) macro per adapter, and accuracy per adapter and family."""
    rows, fam_rows = [], []
    for a in s.ev.adapters:
        i = s.pos[a]
        row = {"adapter": a, **roles[a]}
        for key, v, d in (("strict", s.macro[i], s.macro_boot[i]), ("lenient", l.macro[i], l.macro_boot[i]),
                          ("loss", l.macro[i] - s.macro[i], l.macro_boot[i] - s.macro_boot[i])):
            row.update({key: v, f"{key}_lo": ci(d)[0], f"{key}_hi": ci(d)[1]})
        rows.append(row)
        for f in s.groups:
            r = {"adapter": a, "family": f, "procedural": f in s.procedural}
            for key, v, d in (("strict", s.acc[f][i], s.boot[f][i]), ("lenient", l.acc[f][i], l.boot[f][i]),
                              ("loss", l.acc[f][i] - s.acc[f][i], l.boot[f][i] - s.boot[f][i])):
                r.update({key: v, f"{key}_lo": ci(d)[0], f"{key}_hi": ci(d)[1]})
            fam_rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(fam_rows)


def decomposition(x, y, xd, yd, rng) -> dict:
    """Across members: strict x, lenient y, loss = y - x, and their (co)variances (ddof 1), which satisfy
    var(x) = var(y) + var(loss) - 2 cov(y, loss); each with its excess over the mean evaluation-noise (co)variance
    of the bootstrap draws (member x draw arrays), with intervals resampling members."""
    loss, ld = y - x, yd - xd
    out = {}
    for key, v, d in (("strict", x, xd), ("lenient", y, yd), ("loss", loss, ld)):
        sp = spread(v, d.var(axis=1, ddof=1), rng)
        out.update({f"var_{key}": float(np.var(v, ddof=1)), f"excess_{key}": sp["excess_var"],
                    f"excess_{key}_lo": sp["excess_lo"], f"excess_{key}_hi": sp["excess_hi"]})
    noise_cov = np.array([np.cov(a, b)[0, 1] for a, b in zip(yd, ld)])
    idx = rng.integers(0, len(x), (2000, len(x)))
    boot = [np.cov(y[i], loss[i])[0, 1] - noise_cov[i].mean() for i in idx]
    out.update({"cov_lenient_loss": float(np.cov(y, loss)[0, 1]),
                "excess_cov": float(np.cov(y, loss)[0, 1] - noise_cov.mean()),
                "excess_cov_lo": ci(np.array(boot))[0], "excess_cov_hi": ci(np.array(boot))[1]})
    return out


def family_decomposition(s: Suite, l: Suite, mem: list[int], rng) -> pd.DataFrame:
    """The decomposition per family, and each procedural family's contribution to the between-member variance of
    the strict macro: (1/F^2) sum_g cov(a_f, a_g) over procedural families g (contributions sum to var(macro))."""
    proc = s.procedural
    a = np.array([s.acc[f][mem] for f in proc])
    cov = np.cov(a)
    contrib = cov.sum(axis=1) / len(proc) ** 2
    total = float(np.var(s.macro[mem], ddof=1))
    rows = []
    for f in s.groups:
        r = {"family": f, "procedural": f in proc,
             **decomposition(s.acc[f][mem], l.acc[f][mem], s.boot[f][mem], l.boot[f][mem], rng)}
        if f in proc:
            r["contribution"] = float(contrib[proc.index(f)])
            r["share"] = r["contribution"] / total
        rows.append(r)
    return pd.DataFrame(rows).sort_values("contribution", ascending=False, na_position="last")


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def style_outcome(profile: pd.DataFrame, outcomes: dict[str, dict[str, float]], rng) -> pd.DataFrame:
    """Spearman correlation across members between each style feature mean (all probe suites, at the member's
    evaluated step) and each outcome, with a two-sided permutation p-value and BH q-values over all tests."""
    labels = list(next(iter(outcomes.values())))
    feats = [c for c in profile.columns if c not in ("source", "lineage", "bias", "teacher", "step", "suite", "n")]
    rows = []
    for name, values in outcomes.items():
        y = stats.rankdata([values[a] for a in labels])
        perm = np.array([rng.permutation(y) for _ in range(PERMUTATIONS)])
        for f in feats:
            x = profile.loc[labels, f].to_numpy(float)
            if np.ptp(x) == 0:
                continue
            xr = stats.rankdata(x)
            r = float(np.corrcoef(xr, y)[0, 1])
            rp = (perm - perm.mean(1, keepdims=True)) @ (xr - xr.mean()) / (
                np.sqrt(((perm - perm.mean(1, keepdims=True)) ** 2).sum(1)) * np.sqrt(((xr - xr.mean()) ** 2).sum()))
            rows.append({"feature": f, "outcome": name, "spearman": r,
                         "p_perm": float((1 + np.sum(np.abs(rp) >= abs(r))) / (PERMUTATIONS + 1))})
    df = pd.DataFrame(rows)
    if len(df):
        df["q_bh"] = bh(df["p_perm"].to_numpy())
    return df


# -- command ------------------------------------------------------------------------------------------------------

def analyse_suite(run_dir: Path, suite: str, tag: str, processes: int, blind: bool) -> dict:
    ev = load_evaluation(run_dir, suite, tag, blind=blind)
    t0 = time.time()
    lenient_ev, rescued, info = rescore(ev, processes)
    s, l = pair(ev, lenient_ev, seed={"selection": 1, "train": 2}.get(suite, 3))
    return {"ev": ev, "strict": s, "lenient": l, "rescued": rescued, "info": info, "seconds": time.time() - t0}


def rescue_table(r: dict) -> pd.DataFrame:
    """Rescued records per transform and family (records, and distinct answers)."""
    ev, by = r["ev"], r["info"]["by"]
    a, i, s = np.nonzero(~pd.isna(by))
    rec = pd.DataFrame({"family": ev.families[i], "transform": by[a, i, s]})
    counts = rec.groupby(["transform", "family"]).size().rename("records")
    distinct = r["rescued"].groupby(["transform", "family"]).size().rename("distinct_answers")
    return pd.concat([counts, distinct], axis=1).reset_index()


def audit_sample(rescued: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    return pd.concat([g.sample(min(k, len(g)), random_state=0) for _, g in rescued.groupby("transform")]) \
        if len(rescued) else rescued


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--suite", choices=["train", "selection"])
    where.add_argument("--blind", choices=VAULTS, help="a blind vault (needs the harvest manifest)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--selection-tag", help="with --blind: also re-score this selection evaluation and compare")
    ap.add_argument("--processes", type=int, default=16)
    ap.add_argument("--style-profile", help="style profile CSV (default results/<name>/style/profile.csv)")
    ap.add_argument("--out", help="output directory (default results/<name>/format/<suite>)")
    args = ap.parse_args(argv)
    name, specs = member_specs(args.run_dir)
    suite = args.blind or args.suite
    runs = {suite: analyse_suite(args.run_dir, suite, args.tag, args.processes, bool(args.blind))}
    if args.blind and args.selection_tag:
        runs["selection"] = analyse_suite(args.run_dir, "selection", args.selection_tag, args.processes, False)
    roles = adapter_roles(sorted({a for r in runs.values() for a in r["ev"].adapters}), specs)
    rng = np.random.default_rng(0)
    out = Path(args.out or f"results/{name}/format/{suite}")
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    summary = {"run": name}
    for k, r in runs.items():
        s, l = r["strict"], r["lenient"]
        mem_labels = [a for a in s.ev.adapters if roles[a]["role"] == "member"]
        if len(mem_labels) < 2:  # e.g. a baseline-only run
            raise SystemExit(f"the format analysis compares population members, and the {k} evaluation has "
                             f"{len(mem_labels)}: per-adapter scores are in the evaluation's summary.json")
        mem = [s.pos[a] for a in mem_labels]
        adapters, families = adapter_table(s, l, roles)
        dec = decomposition(s.macro[mem], l.macro[mem], s.macro_boot[mem], l.macro_boot[mem], rng)
        fam_dec = family_decomposition(s, l, mem, rng)
        loss = l.macro - s.macro
        cors = {f"strict~{key}": {kind: correlation(s.macro[mem], v[mem], s.macro_boot[mem], d[mem], rng,
                                                    kind == "spearman") for kind in ("spearman", "pearson")}
                for key, v, d in (("lenient", l.macro, l.macro_boot), ("loss", loss, l.macro_boot - s.macro_boot))}
        rescues = rescue_table(r)
        prefix = "" if k == suite else f"{k}_"
        r["info"]["guard"].to_csv(out / f"{prefix}guard.csv", index=False)
        rescues.to_csv(out / f"{prefix}rescues.csv", index=False)
        audit_sample(r["rescued"]).to_csv(out / f"{prefix}audit.csv", index=False)
        r["rescued"].to_csv(out / f"{prefix}rescued_answers.csv.gz", index=False)
        adapters.to_csv(out / f"{prefix}adapters.csv", index=False)
        families.to_csv(out / f"{prefix}families.csv", index=False)
        fam_dec.to_csv(out / f"{prefix}decomposition_families.csv", index=False)
        info = {key: v for key, v in r["info"].items() if key not in ("guard", "by", "safe")}
        summary[k] = {"tag": r["ev"].tag, **info, "seconds": r["seconds"], "decomposition": dec,
                      "correlations": cors, "safe_transforms": {f: sorted(v) for f, v in r["info"]["safe"].items()}}
        lines += suite_lines(name, k, r, roles, adapters, rescues, dec, fam_dec, cors, mem_labels)
        if k == "selection" and suite == "selection":
            path = Path(args.style_profile or f"results/{name}/style/profile.csv")
            if path.exists():
                so = style_vs_outcome(path, mem_labels, s, l, rng)
                so.to_csv(out / "style_outcome.csv", index=False)
                lines += style_lines(so)
            else:
                lines.append(f"style vs outcome: no style profile at {path}")
    if len(runs) == 2:
        lines += blind_lines(runs, suite, roles, name, rng, out, summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print("\n".join(lines))
    print(f"\nCSV/JSON: {out}/")


def style_vs_outcome(path: Path, members: list[str], s: Suite, l: Suite, rng) -> pd.DataFrame:
    prof = pd.read_csv(path)
    prof = prof[(prof["source"] == "member") & (prof["suite"] == "all")]
    rows = {}
    for a in members:
        lineage, _, step = a.partition("@")
        hit = prof[(prof["lineage"] == lineage) & (prof["step"] == (int(step) if step else prof["step"].max()))]
        if len(hit):
            rows[a] = hit.iloc[0]
    table = pd.DataFrame(rows).T
    labels = list(table.index)
    outcomes = {"strict": {a: s.macro[s.pos[a]] for a in labels}, "lenient": {a: l.macro[l.pos[a]] for a in labels},
                "loss": {a: l.macro[l.pos[a]] - s.macro[s.pos[a]] for a in labels}}
    df = style_outcome(table, outcomes, rng)
    df["members"] = len(labels)
    return df


# -- terminal -----------------------------------------------------------------------------------------------------

def pct(v, lo=None, hi=None, sign=False) -> str:
    f = "+.1f" if sign else ".1f"
    return f"{100 * v:{f}}" + ("" if lo is None else f" [{100 * lo:{f}},{100 * hi:{f}}]")


def suite_lines(name, k, r, roles, adapters, rescues, dec, fam_dec, cors, members) -> list[str]:
    info, ev = r["info"], r["ev"]
    rescued_records = int(rescues["records"].sum()) if len(rescues) else 0
    lines = [f"format {name} {k} {ev.tag}: {info['records']} records; parsed, untruncated and wrong "
             f"{info['wrong_parsed']} ({info['distinct_wrong']} distinct answers, {info['jobs']} re-scored); rescued "
             f"{rescued_records} records ({100 * rescued_records / max(info['wrong_parsed'], 1):.1f}%); "
             f"{info['errors']['timeout']} timeouts, {info['errors']['other']} errors; {r['seconds']:.0f} s"]
    g = info["guard"]
    disabled = g[g["disabled"].astype(bool)].groupby("transform")["family"].apply(lambda x: ", ".join(x))
    lines.append("guard (gold and correct answers): disabled " + ("; ".join(
        f"{t} for {fams}" for t, fams in disabled.items()) if len(disabled) else "nothing"))
    per = rescues.groupby("transform")[["records", "distinct_answers"]].sum().reindex(ORDER).dropna()
    lines.append("rescued records by first rescuing transform: " + ", ".join(
        f"{t} {int(v.records)}" for t, v in per.iterrows()) + " (examples in audit.csv)")
    top = rescues.groupby("family")["records"].sum().sort_values(ascending=False).head(8)
    lines.append("  by family: " + ", ".join(f"{f} {n}" for f, n in top.items()))

    a = adapters.set_index("adapter")
    m = a.loc[members]
    lines += ["", "macro accuracy %, strict (as recorded) / lenient / format loss = lenient - strict [95% CI]"]
    rows = [["population", f"{len(members)} members", f"{100 * m.strict.mean():.1f} ± {100 * m.strict.std():.1f}",
             f"{100 * m.lenient.mean():.1f} ± {100 * m.lenient.std():.1f}",
             f"{100 * m.loss.mean():+.1f} ± {100 * m.loss.std():.1f}"]]
    shown = [x for x in a.index if roles[x]["role"] != "member"]
    best = m.sort_values("strict", ascending=False).index[:1].tolist()
    for x in best + shown:
        rr = a.loc[x]
        rows.append([x, rr["role"] if rr["role"] != "member" else rr["bias"],
                     pct(rr.strict, rr.strict_lo, rr.strict_hi), pct(rr.lenient, rr.lenient_lo, rr.lenient_hi),
                     pct(rr.loss, rr.loss_lo, rr.loss_hi, sign=True)])
    lines += render(["adapter", "role", "strict", "lenient", "loss"], rows, left=2)

    def ex(key):
        return f"{1e4 * dec[f'excess_{key}']:.2f} [{1e4 * dec[f'excess_{key}_lo']:.2f},{1e4 * dec[f'excess_{key}_hi']:.2f}]"
    lines += ["", f"between-member decomposition of macro ({len(members)} members, pp^2): var strict "
                  f"{1e4 * dec['var_strict']:.2f} = var lenient {1e4 * dec['var_lenient']:.2f} + var loss "
                  f"{1e4 * dec['var_loss']:.2f} - 2 cov(lenient, loss) {1e4 * dec['cov_lenient_loss']:.2f}",
              f"  excess over noise: strict {ex('strict')}, lenient {ex('lenient')}, loss {ex('loss')}, "
              f"cov {ex('cov')}"]
    lines.append("  families by contribution to var(strict macro): " + ", ".join(
        f"{r.family} {100 * r.share:.0f}% (var strict/lenient/loss {1e4 * r.var_strict:.0f}/{1e4 * r.var_lenient:.0f}/"
        f"{1e4 * r.var_loss:.0f})" for r in fam_dec[fam_dec["procedural"]].head(5).itertuples()))
    for key, c in cors.items():
        lines.append(f"  across members {key}: Spearman {c['spearman']['r']:+.2f} [{c['spearman']['members_lo']:+.2f},"
                     f"{c['spearman']['members_hi']:+.2f}], Pearson {c['pearson']['r']:+.2f} "
                     f"[{c['pearson']['members_lo']:+.2f},{c['pearson']['members_hi']:+.2f}]")
    return lines


def style_lines(so: pd.DataFrame) -> list[str]:
    if so.empty:
        return ["", "style vs outcome: no member with a profile at its evaluated step"]
    top = so.reindex(so["spearman"].abs().sort_values(ascending=False).index).head(8)
    return ["", f"style vs outcome ({so['members'].iloc[0]} members, {len(so)} tests; q = Benjamini-Hochberg; "
                f"{int((so['q_bh'] < 0.05).sum())} with q < 0.05): "
                + ", ".join(f"{r.feature}~{r.outcome} {r.spearman:+.2f} (p {r.p_perm:.3f}, q {r.q_bh:.2f})"
                            for r in top.itertuples())]


def blind_lines(runs, vault, roles, name, rng, out, summary) -> list[str]:
    sel, bl = runs["selection"], runs[vault]
    mem = [a for a in sel["ev"].adapters if roles[a]["role"] == "member" and a in bl["strict"].pos]
    lines = ["", f"selection vs {vault} across {len(mem)} members: Spearman / Pearson [member CI]"]
    rows = []
    for key in ("strict", "lenient", "loss"):
        def vals(r):
            s, l = r["strict"], r["lenient"]
            idx = [s.pos[a] for a in mem]
            if key == "loss":
                return l.macro[idx] - s.macro[idx], l.macro_boot[idx] - s.macro_boot[idx]
            x = r[key]
            return x.macro[idx], x.macro_boot[idx]
        (x, xd), (y, yd) = vals(sel), vals(bl)
        c = {kind: correlation(x, y, xd, yd, rng, kind == "spearman") for kind in ("spearman", "pearson")}
        rows.append({"metric": key, **{f"{kind}_{f}": v[f] for kind, v in c.items() for f in v}})
        lines.append(f"  {key}: {c['spearman']['r']:+.2f} [{c['spearman']['members_lo']:+.2f},"
                     f"{c['spearman']['members_hi']:+.2f}] / {c['pearson']['r']:+.2f} "
                     f"[{c['pearson']['members_lo']:+.2f},{c['pearson']['members_hi']:+.2f}]")
    pd.DataFrame(rows).to_csv(out / "selection_blind_correlations.csv", index=False)
    members = [a for a in sel["ev"].adapters if roles[a]["role"] == "member"]
    outlier, _, source = pick_outlier(name, sel["strict"], members)
    median = median_member(sel["strict"], members)
    comp = []
    for key in ("strict", "lenient"):
        df = compare_outlier({"selection": sel[key], vault: bl[key]}, roles, outlier, median)
        comp.append(df.assign(scoring=key))
    comp = pd.concat(comp)
    comp.to_csv(out / "outlier_comparisons.csv", index=False)
    lines += ["", f"outlier {outlier} ({source.split(':')[0]}) minus comparator, macro points [paired CI]"]
    lines += render(["scoring", "suite", "comparator", "kind", "diff [CI]"],
                    [[r.scoring, r.suite, r.comparator, r.kind, pct(r.diff, r.lo, r.hi, sign=True)]
                     for r in comp.itertuples()], left=4)
    summary["selection_blind"] = rows
    return lines


if __name__ == "__main__":
    main()
