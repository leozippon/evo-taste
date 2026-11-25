"""The verdict on one response: where its final answer is, whether it is in the wrapper asked for, and whether it is
correct. Pure function of the job; runs inside a verifier process (see pool.py), which bounds its time.

Gold formats by kind:
- open: str, the reference answer in LaTeX or plain notation ("\\frac{1}{2}", "18", "(1, 2)").
- choice: {"options": [option text, ...], "answer": letter}, options in the order of their letters A, B, C, ...
- verdict: {"labels": [label, ...], "answer": label}, a closed label set.
- code: {"harness": "pytest" | "assert" | "check", "tests": str, "reference": str, optional "entry_point" (required
  for check), "time_limit_s" (default 10), "memory_mb" (default 1024)}. The program is saved as solution.py, and
  every top-level name it defines is imported ahead of the tests; pytest runs `tests` as test_solution.py; assert
  runs `tests` as a script; check does the same and then calls `check(<entry_point>)`.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass

from evotaste.verify import equivalence
from evotaste.verify.equivalence import balanced
from evotaste.verify.sandbox import OVERHEAD_S, run_python
from evotaste.verify.wrappers import WRAPPERS, Found, for_kind, python_block

KINDS = ("open", "choice", "verdict", "code")
CHANNELS = ("direct", "program")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PROGRAM_TIMEOUT_S = 5
PROGRAM_MEMORY_MB = 512
CODE_TIME_LIMIT_S = 10
CODE_MEMORY_MB = 1024


@dataclass(frozen=True)
class Job:
    kind: str
    text: str  # the response
    gold: object  # see the module docstring
    wrapper: str  # id of the wrapper asked for
    channels: tuple[str, ...] = ("direct",)  # kind open with wrapper box may also offer "program"


@dataclass(frozen=True)
class Verdict:
    reward: float  # 1.0 iff the final answer is in the wrapper asked for (and channel offered) and verified correct
    correct_any: bool  # verified correct under tolerant extraction: the last answer in any known wrapper
    complied: bool  # a final answer was found in the wrapper asked for
    channel: str | None  # "direct" | "program" for kind open when an answer was found, else None
    extracted: str | None  # the answer verified for the reward: wrapper content, a program's output line, or code
    error: str | None = None  # why a found answer could not be verified correct beyond being wrong (or "timeout")


def check_job(job: Job) -> None:
    """Refuse a malformed job (a caller's bug, never a verdict)."""
    if job.kind not in KINDS:
        raise ValueError(f"unknown kind {job.kind!r}")
    w = WRAPPERS.get(job.wrapper)
    if w is None or job.kind not in w.kinds:
        raise ValueError(f"wrapper {job.wrapper!r} does not apply to kind {job.kind!r}")
    if not job.channels or set(job.channels) - set(CHANNELS):
        raise ValueError(f"bad channels {job.channels!r}")
    if "program" in job.channels and (job.kind != "open" or job.wrapper != "box"):
        raise ValueError("the program channel exists only for kind open with wrapper box")
    if job.kind != "open" and job.channels != ("direct",):
        raise ValueError(f"kind {job.kind} has only the direct channel")


def budget_s(job: Job) -> float:
    """Sandbox time a job may spend (a program, or the tests run on up to two extracted programs)."""
    if job.kind == "code":
        return 2 * (float(job.gold.get("time_limit_s", CODE_TIME_LIMIT_S)) + OVERHEAD_S)
    return PROGRAM_TIMEOUT_S + OVERHEAD_S if "program" in job.channels else 0.0


# -- the final answer -----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    channel: str  # "direct" or "program"
    found: Found


def _candidates(job: Job, wrappers) -> list[Candidate]:
    out = [Candidate("direct", f) for w in wrappers if (f := w.find(job.text)) is not None]
    if "program" in job.channels:
        if (f := python_block(job.text)) is not None:
            out.append(Candidate("program", f))
    return out


def _last(candidates: list[Candidate]) -> Candidate | None:
    """The candidate ending last; on a tie the earlier one in the list (default wrapper first)."""
    best = None
    for c in candidates:
        if best is None or c.found.end > best.found.end:
            best = c
    return best


def strict_answer(job: Job) -> Candidate | None:
    """The final answer in the wrapper asked for, among the channels offered."""
    if job.kind == "code":
        f = python_block(job.text)
        return Candidate("direct", f) if f else None
    return _last(_candidates(job, [WRAPPERS[job.wrapper]] if "direct" in job.channels else []))


def tolerant_answer(job: Job) -> Candidate | None:
    """The final answer in any wrapper of the kind (the program channel if offered): the one that ends last."""
    if job.kind == "code":
        f = python_block(job.text, tolerant=True)
        return Candidate("direct", f) if f else None
    return _last(_candidates(job, for_kind(job.kind)))


# -- correctness ----------------------------------------------------------------------------------------------------

def _clean_label(s: str) -> str:
    """A letter or label without its dressings (math delimiters, a \\text-like command around it, markdown emphasis,
    quotes, a trailing period), with single spaces."""
    s = s.strip()
    prev = None
    while s != prev:
        prev = s
        s = s.strip()
        for a, b in (("$", "$"), ("\\(", "\\)"), ("**", "**"), ("*", "*"), ("`", "`"), ('"', '"'), ("'", "'")):
            if len(s) > len(a) + len(b) and s.startswith(a) and s.endswith(b):
                s = s[len(a):-len(b)]
        if (m := re.fullmatch(r"\\(?:text|textbf|mathrm|mathbf|boxed)\s*\{(.*)\}", s, re.S)) and balanced(m.group(1)):
            s = m.group(1)
        if s.endswith("."):
            s = s[:-1]
    return re.sub(r"\s+", " ", s.strip())


_BARE_LETTER = re.compile(r"(?:(?i:option|choice)\s*)?(?:\(([A-Z])\)|\[([A-Z])\]|([A-Z])[.:)]?)")
_LETTER_TEXT = re.compile(r"(?:\(([A-Z])\)|([A-Z])[.:)]?)\s+(.+)", re.S)


def resolve_choice(content: str, options: list[str]) -> int | None:
    """The one option a choice answer names. Readings: a letter in its usual dressings (C, (C), [C], C., C), C:,
    Option C); a letter followed by that same option's exact text (C. 42, (C) 42); the exact text of an option
    (case and spacing aside). A reading that does not fit names nothing; the answer counts only if its readings name
    exactly one option (so "C. 43" with 43 under D, or a text shared by two options, names none)."""
    s = _clean_label(content)
    norm = [_clean_label(o).lower() for o in options]
    named: set[int] = set()
    if m := _BARE_LETTER.fullmatch(s):
        named.add(LETTERS.find(next(g for g in m.groups() if g)))
    if m := _LETTER_TEXT.fullmatch(s):
        i = LETTERS.find(m.group(1) or m.group(2))
        if i < len(options) and _clean_label(m.group(3)).lower() == norm[i]:
            named.add(i)
    named |= {i for i, o in enumerate(norm) if o == s.lower()}
    if len(named) != 1:
        return None
    i = named.pop()
    return i if i < len(options) else None


def resolve_label(content: str, labels: list[str]) -> str | None:
    s = _clean_label(content).lower()
    hits = [lab for lab in labels if _clean_label(lab).lower() == s]
    return hits[0] if len(hits) == 1 else None


_PYTEST_MAIN = """\
import pytest
class Tally:
    def __init__(self):
        self.passed = self.bad = 0
    def pytest_runtest_logreport(self, report):
        if report.when == "call" and report.passed:
            self.passed += 1
        if report.failed or report.skipped:
            self.bad += 1
tally = Tally()
code = pytest.main(["-q", "-x", "-p", "no:cacheprovider", "test_solution.py"], plugins=[tally])
if code == 0 and tally.passed > 0 and tally.bad == 0:
    print({nonce!r})
"""


# Every top-level name of the solution, underscored ones included (`from solution import *` skips those)
_IMPORT_ALL = ("import solution as _solution\nglobals().update({k: v for k, v in vars(_solution).items() "
               "if not k.startswith('__')})\n")


def run_tests(program: str, gold: dict) -> str | None:
    """None if the program passes all of the item's tests in the sandbox, else the reason. Success is the harness's
    random token printed after the tests and a zero exit status, so a program that exits early never passes."""
    harness = gold["harness"]
    nonce = secrets.token_hex(16)
    files = {"solution.py": program}
    if harness == "pytest":
        files["test_solution.py"] = _IMPORT_ALL + gold["tests"]
        main = _PYTEST_MAIN.format(nonce=nonce)
    elif harness in ("assert", "check"):
        main = _IMPORT_ALL + gold["tests"]
        if harness == "check":
            main += f"\ncheck({gold['entry_point']})\n"
        main += f"\nprint({nonce!r})\n"
    else:
        raise ValueError(f"unknown harness {harness!r}")
    if "__eq__" in program and "__eq__" not in gold.get("reference", ""):
        return "program defines __eq__"  # an always-equal object would pass equality tests
    r = run_python(main, files=files, timeout_s=float(gold.get("time_limit_s", CODE_TIME_LIMIT_S)),
                   memory_mb=int(gold.get("memory_mb", CODE_MEMORY_MB)))
    if r.timed_out:
        return "timeout"
    if r.returncode == 0 and nonce in r.stdout.split("\n"):
        return None
    tail = (r.stderr.strip() or r.stdout.strip())[-300:]
    return f"tests failed (exit {r.returncode}): {tail}"


def program_output(program: str) -> tuple[str | None, str | None]:
    """(last non-empty line of the program's output, error) for the program channel."""
    r = run_python(program, timeout_s=PROGRAM_TIMEOUT_S, memory_mb=PROGRAM_MEMORY_MB)
    if r.timed_out:
        return None, "program timeout"
    if r.returncode != 0:
        return None, f"program exit {r.returncode}: {r.stderr.strip()[-300:]}"
    lines = [line for line in r.stdout.splitlines() if line.strip()]
    if not lines:
        return None, "program printed nothing"
    return lines[-1].strip(), None


def assess(job: Job, c: Candidate) -> tuple[bool, str | None, str | None]:
    """(correct, extracted answer, error) of one candidate final answer."""
    content = c.found.content
    if job.kind == "code":
        error = run_tests(content, job.gold)
        return error is None, content, error
    if c.channel == "program":
        content, error = program_output(content)
        if content is None:
            return False, None, error
    if job.kind == "open":
        return equivalence.equivalent(content, job.gold), content, None
    if job.kind == "choice":
        i = resolve_choice(content, job.gold["options"])
        return i is not None and LETTERS[i] == job.gold["answer"], content, None
    return resolve_label(content, job.gold["labels"]) == job.gold["answer"], content, None


def judge(job: Job) -> Verdict:
    strict, tolerant = strict_answer(job), tolerant_answer(job)
    correct, extracted, error = assess(job, strict) if strict else (False, None, None)
    if tolerant is None:
        any_correct = False
    elif strict is not None and tolerant == strict:
        any_correct = correct
    else:
        any_correct = correct or assess(job, tolerant)[0]
    channel = strict.channel if strict is not None and job.kind == "open" else None
    return Verdict(float(correct), bool(any_correct), strict is not None, channel, extracted, error)
