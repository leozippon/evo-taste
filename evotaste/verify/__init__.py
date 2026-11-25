"""Verifiers and sandbox of Round 2: the reward of every response passes through `verify_many`.

    verify_many(jobs, timeout=None) -> list[Verdict]     Job(kind, text, gold, wrapper, channels)
    WRAPPERS, TRAIN_WRAPPERS, HELDOUT_WRAPPERS            wrapper id -> instruction, render, extractor
    run_python(code, *, stdin, files, timeout_s, memory_mb) -> Result
    gold_ok(kind, gold) / gold_check(items)               does the reference verify against itself, unambiguously?
    is_undecided(error)                                   a verdict or reason that says nothing about the answer
    RULES_VERSION                                         changes whenever a rule that can change a verdict changes

Gold formats and the extraction and equivalence rules: judge.py, wrappers.py, equivalence.py and
docs/dev/VERIFY_NOTES.md.
"""

from __future__ import annotations

from evotaste.verify import equivalence
from evotaste.verify.judge import CODE_TIME_LIMIT_S, LETTERS, Job, Verdict, check_job, is_undecided
from evotaste.verify.pool import Pool
from evotaste.verify.sandbox import Result, run_python
from evotaste.verify.wrappers import DEFAULT, HELDOUT_WRAPPERS, TRAIN_WRAPPERS, WRAPPERS, Wrapper, for_kind

# Bump on every change to a rule that can change a verdict or a gold_check result (stored verdicts and suites built
# under another version are stale).
RULES_VERSION = "2026-11-25.audit-1"

__all__ = ["CODE_TIME_LIMIT_S", "DEFAULT", "HELDOUT_WRAPPERS", "Job", "LETTERS", "RULES_VERSION", "Result",
           "TRAIN_WRAPPERS", "Verdict", "WRAPPERS", "Wrapper", "gold_check", "gold_ok", "is_undecided", "run_python",
           "verify_many"]

_POOL: Pool | None = None


def verify_many(jobs: list[Job], timeout: float | None = None) -> list[Verdict]:
    """Verdicts of the jobs, in order, computed by a persistent pool of verifier processes. A job's deadline is
    `timeout` seconds if given, else a margin outside its sandbox runs plus the runs' own limits; a job past it gets
    reward 0 with an undecided error (is_undecided). A malformed job raises ValueError before anything runs. Not for
    concurrent use from several threads."""
    global _POOL
    for job in jobs:
        check_job(job)
    if not jobs:
        return []
    _POOL = _POOL or Pool()
    return _POOL.run(jobs, timeout)


def _refusal(kind: str, gold) -> str | None:
    """Why a reference is refused before any verification: ambiguous (open), or its correct option text repeated."""
    if kind == "open":
        return equivalence.ambiguity(*equivalence.reference(gold))
    if kind == "choice":
        if gold["answer"] not in LETTERS[:len(gold["options"])]:
            raise ValueError(f"choice answer {gold['answer']!r} is not one of {len(gold['options'])} options")
        texts = [" ".join(o.split()).lower() for o in gold["options"]]
        if texts.count(texts[LETTERS.index(gold["answer"])]) > 1:
            return "ambiguous: the correct option's text is also another option's"
    if kind == "verdict" and gold["answer"] not in gold["labels"]:
        raise ValueError(f"verdict answer {gold['answer']!r} is not among the labels {gold['labels']!r}")
    return None


def _gold_jobs(kind: str, gold) -> list[Job]:
    """The reference answer rendered in every wrapper of its kind."""
    if kind == "code":
        return [Job("code", WRAPPERS["code"].render(gold["reference"]), gold, "code")]
    if kind == "open":
        answer = equivalence.reference(gold)[0]
    elif kind in ("choice", "verdict"):
        answer = gold["answer"]
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return [Job(kind, w.render(answer), gold, w.id) for w in for_kind(kind)]


def gold_check(items: list[tuple[str, object]], timeout: float | None = None) -> list[str | None]:
    """For each (kind, gold): None if the reference is unambiguous and earns reward 1 in every wrapper of its kind
    (for code: the reference solution passes the item's tests within its time limit), else the first reason it does
    not. A reason for which is_undecided holds decides nothing; the caller repeats or sets aside such items."""
    refusals = [_refusal(kind, gold) for kind, gold in items]
    jobs = [[] if r else _gold_jobs(kind, gold) for r, (kind, gold) in zip(refusals, items)]
    flat = verify_many([j for js in jobs for j in js], timeout)
    out, k = [], 0
    for refusal, js in zip(refusals, jobs):
        verdicts, k = flat[k:k + len(js)], k + len(js)
        bad = next(((j, v) for j, v in zip(js, verdicts) if v.reward != 1.0), None)
        out.append(refusal or (None if bad is None else f"wrapper {bad[0].wrapper}: {bad[1].error or 'not verified'}"))
    return out


def gold_ok(kind: str, gold) -> bool:
    """Whether the reference verifies against itself, unambiguously (see gold_check)."""
    return gold_check([(kind, gold)])[0] is None
