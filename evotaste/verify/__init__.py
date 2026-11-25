"""Verifiers and sandbox of Round 2: the reward of every response passes through `verify_many`.

    verify_many(jobs, timeout=None) -> list[Verdict]     Job(kind, text, gold, wrapper, channels)
    WRAPPERS, TRAIN_WRAPPERS, HELDOUT_WRAPPERS            wrapper id -> instruction, render, extractor
    run_python(code, *, stdin, files, timeout_s, memory_mb) -> Result
    gold_ok(kind, gold) / gold_check(items)               does the reference verify against itself?

Gold formats and the extraction and equivalence rules: judge.py, wrappers.py, equivalence.py and
docs/dev/VERIFY_NOTES.md.
"""

from __future__ import annotations

from evotaste.verify.judge import CODE_TIME_LIMIT_S, LETTERS, Job, Verdict, check_job
from evotaste.verify.pool import Pool
from evotaste.verify.sandbox import Result, run_python
from evotaste.verify.wrappers import DEFAULT, HELDOUT_WRAPPERS, TRAIN_WRAPPERS, WRAPPERS, Wrapper, for_kind

__all__ = ["CODE_TIME_LIMIT_S", "DEFAULT", "HELDOUT_WRAPPERS", "Job", "LETTERS", "Result", "TRAIN_WRAPPERS",
           "Verdict", "WRAPPERS", "Wrapper", "gold_check", "gold_ok", "run_python", "verify_many"]

_POOL: Pool | None = None


def verify_many(jobs: list[Job], timeout: float | None = None) -> list[Verdict]:
    """Verdicts of the jobs, in order, computed by a persistent pool of verifier processes. Each job ends within
    `timeout` seconds of wall-clock time (default: its sandbox time plus a margin) or scores 0 with error "timeout".
    A malformed job raises ValueError before anything runs. Not for concurrent use from several threads."""
    global _POOL
    for job in jobs:
        check_job(job)
    if not jobs:
        return []
    _POOL = _POOL or Pool()
    return _POOL.run(jobs, timeout)


def _gold_jobs(kind: str, gold) -> list[Job]:
    """The reference answer rendered in every wrapper of its kind."""
    if kind == "code":
        return [Job("code", WRAPPERS["code"].render(gold["reference"]), gold, "code")]
    if kind == "open":
        answer = gold
    elif kind == "choice":
        if gold["answer"] not in LETTERS[:len(gold["options"])]:
            raise ValueError(f"choice answer {gold['answer']!r} is not one of {len(gold['options'])} options")
        answer = gold["answer"]
    elif kind == "verdict":
        if gold["answer"] not in gold["labels"]:
            raise ValueError(f"verdict answer {gold['answer']!r} is not among the labels {gold['labels']!r}")
        answer = gold["answer"]
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return [Job(kind, w.render(answer), gold, w.id) for w in for_kind(kind)]


def gold_check(items: list[tuple[str, object]], timeout: float | None = None) -> list[str | None]:
    """For each (kind, gold): None if the reference earns reward 1 in every wrapper of its kind (for code: the
    reference solution passes the item's tests within its time limit), else the first reason it does not. A reason
    of "timeout" means undecided, not wrong; the caller decides whether to drop or retry such items."""
    jobs = [_gold_jobs(kind, gold) for kind, gold in items]
    flat = verify_many([j for js in jobs for j in js], timeout)
    out, k = [], 0
    for js in jobs:
        verdicts, k = flat[k:k + len(js)], k + len(js)
        bad = next(((j, v) for j, v in zip(js, verdicts) if v.reward != 1.0), None)
        out.append(None if bad is None else f"wrapper {bad[0].wrapper}: {bad[1].error or 'not verified'}")
    return out


def gold_ok(kind: str, gold) -> bool:
    """Whether the reference verifies against itself (see gold_check)."""
    return gold_check([(kind, gold)])[0] is None
