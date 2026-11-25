"""The verdict on one response: where its final answer is, whether it is in the wrapper asked for, and whether it is
correct. Pure function of the job; runs inside a verifier process (see pool.py), which bounds its time.

Gold formats by kind:
- open: str, the reference answer in LaTeX or plain notation ("\\frac{1}{2}", "18", "(1, 2)"), or
  {"answer": str, "type": "number" | "expression" | "tuple" | "list" | "interval"} (see equivalence.py).
- choice: {"options": [option text, ...], "answer": letter}, options in the order of their letters A, B, C, ...
- verdict: {"labels": [label, ...], "answer": label}, a closed label set.
- code: {"harness": "pytest" | "assert" | "check", "tests": str, "reference": str, optional "entry_point" (required
  for check), "prelude" (default none), "time_limit_s" (default 10, CPU seconds), "memory_mb" (default 1024)}. The
  program is saved as solution.py, after the prelude if there is one (dataset text such as `from typing import *`
  and ListNode, which the program may rely on), and every top-level name of that file is imported ahead of the tests;
  pytest runs `tests` as test_solution.py; assert runs `tests` as a script; check does the same and then calls
  `check(<entry_point>)`.

An error starting with UNDECIDED means that the verdict says nothing about the answer (the sandbox ran out of wall
clock before CPU time, or the verifier's deadline passed): the reward is 0, but the caller must not take it as wrong.
"""

from __future__ import annotations

import ast
import re
import secrets
from dataclasses import dataclass

from evotaste.verify import equivalence
from evotaste.verify.equivalence import balanced
from evotaste.verify.sandbox import UNDECIDED, run_python
from evotaste.verify.wrappers import WRAPPERS, Found, Unclosed, for_kind, python_block, python_blocks

KINDS = ("open", "choice", "verdict", "code")
CHANNELS = ("direct", "program")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PROGRAM_TIMEOUT_S = 5
PROGRAM_MEMORY_MB = 512
CODE_TIME_LIMIT_S = 10
CODE_MEMORY_MB = 1024
CPU_EXHAUSTED = 128 + 24  # exit status of a job killed by SIGXCPU at its CPU-time limit


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
    error: str | None = None  # why a found answer could not be verified correct beyond being wrong; UNDECIDED...


def is_undecided(error: str | None) -> bool:
    """The verdict (or a gold_check reason) decides nothing about the answer: repeat it or set it aside."""
    return error is not None and UNDECIDED in error


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
    if job.kind == "open":
        equivalence.reference(job.gold)


# -- the final answer -----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    channel: str  # "direct" or "program"
    found: Found


def _answer(job: Job, wrappers) -> Candidate | None:
    """A direct answer takes precedence: the last marker among these wrappers decides (on a tie the earlier wrapper,
    default first), and if that marker is unclosed there is no answer at all. Only a response without any marker is
    a program answer, when the program channel is offered: its last fenced Python block. (A correct box followed by
    an unasked-for program that prints nothing is a correct box.)"""
    best, at = None, -1
    for w in wrappers:
        x = w.locate(job.text)
        if x is not None and (pos := x.end if isinstance(x, Found) else x.start) > at:
            best, at = x, pos
    if best is not None:
        return Candidate("direct", best) if isinstance(best, Found) else None
    if "program" in job.channels and isinstance(f := python_block(job.text), Found):
        return Candidate("program", f)
    return None


def _defined(program: str) -> set[str] | None:
    """The top-level names a program defines (functions, classes, assignments, imports); None if it does not parse."""
    try:
        tree = ast.parse(program)
    except (SyntaxError, ValueError):
        return None
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                names |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return names


def tested_names(gold: dict) -> set[str]:
    """The names the tests use that the reference solution defines."""
    used = set(re.findall(r"[A-Za-z_]\w*", gold["tests"] + " " + gold.get("entry_point", "")))
    return (_defined(gold["reference"]) or set()) & used


def _code_answer(job: Job, tolerant: bool) -> Candidate | None:
    """The last Python block; if it does not define the names the tests use, all Python blocks in order (a solution
    followed by an example of its use)."""
    blocks = python_blocks(job.text, tolerant)
    if isinstance(blocks, Unclosed) or not blocks or not blocks[-1].content.strip():
        return None
    last = blocks[-1]
    if len(blocks) > 1 and not tested_names(job.gold) <= (_defined(last.content) or set()):
        return Candidate("direct", Found("\n\n".join(b.content for b in blocks), last.end))
    return Candidate("direct", last)


def strict_answer(job: Job) -> Candidate | None:
    """The final answer in the wrapper asked for, among the channels offered (with only the program channel, boxes
    are ignored)."""
    if job.kind == "code":
        return _code_answer(job, tolerant=False)
    return _answer(job, [WRAPPERS[job.wrapper]] if "direct" in job.channels else [])


def tolerant_answer(job: Job) -> Candidate | None:
    """The final answer in any wrapper of the kind, with the same precedence over the program (if offered)."""
    if job.kind == "code":
        return _code_answer(job, tolerant=True)
    return _answer(job, for_kind(job.kind))


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
_LETTER_TEXT = re.compile(r"(?:(?i:option|choice)\s*)?(?:\(([A-Z])\)|([A-Z])[.:)])\s*(.+)", re.S)
# Another option letter in the text after a letter: "(B)", "[B]", or a lone letter after a comma, "or", "and", "/"
_OTHER_LETTER = re.compile(r"\(([A-Z])\)|\[([A-Z])\]|(?:^|,|;|/|&|\b(?i:or|and)\b)\s*(?:(?i:option|choice)\s*)?"
                           r"([A-Z])(?:[.:)]|\s*$|\s*(?:,|;|/|&|\b(?i:or|and)\b))")


def resolve_choice(content: str, options: list[str]) -> int | None:
    """The one option a choice answer names. Readings: a letter in its usual dressings (C, (C), [C], C., C), C:,
    Option C); a letter with a separator followed by any text (C. 42, (C) \\frac{2}{5}): the letter decides, and the
    text only vetoes when it is exactly a different option's text or names another option letter ("(A), (B)",
    "A. or B"); the exact text of an option (case and spacing aside). The answer counts only if its readings name
    exactly one existing option (so "C. 43" with 43 under D, or a text shared by two options, names none)."""
    s = _clean_label(content)
    norm = [_clean_label(o).lower() for o in options]
    named: set[int] = set()
    if m := _BARE_LETTER.fullmatch(s):
        named.add(LETTERS.find(next(g for g in m.groups() if g)))
    elif m := _LETTER_TEXT.fullmatch(s):
        i = LETTERS.find(m.group(1) or m.group(2))
        text = _clean_label(m.group(3))
        own = i < len(norm) and norm[i] == text.lower()
        if not own and any(o == text.lower() for o in norm) or _OTHER_LETTER.search(text):
            return None  # the text is another option's (and not this one's), or names another letter: a hedge
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


def _limited(r, limit_s: float) -> bool:
    """The job reached its CPU-time limit (SIGXCPU, or SIGKILL at the hard limit after ignoring it)."""
    return r.returncode == CPU_EXHAUSTED or (r.returncode == 128 + 9 and r.cpu_s >= limit_s)


def run_tests(program: str, gold: dict) -> str | None:
    """None if the program passes all of the item's tests in the sandbox, else the reason. Success is the harness's
    random token printed after the tests and a zero exit status, so a program that exits early never passes."""
    harness = gold["harness"]
    nonce = secrets.token_hex(16)
    prelude = gold.get("prelude", "")  # trusted dataset text, as if written at the top of the solution file
    files = {"solution.py": f"{prelude}\n{program}" if prelude else program}
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
    if "__eq__" in program and "__eq__" not in gold.get("reference", ""):  # the model's code only, not the prelude
        return "program defines __eq__"  # an always-equal object would pass equality tests
    limit = float(gold.get("time_limit_s", CODE_TIME_LIMIT_S))
    r = run_python(main, files=files, timeout_s=limit, memory_mb=int(gold.get("memory_mb", CODE_MEMORY_MB)))
    if r.undecided:
        return r.stderr
    if _limited(r, limit):
        return "timeout"
    if r.returncode == 0 and nonce in r.stdout.split("\n"):
        return None
    tail = (r.stderr.strip() or r.stdout.strip())[-300:]
    return f"tests failed (exit {r.returncode}): {tail}"


def program_output(program: str) -> tuple[str | None, str | None]:
    """(last non-empty line of the program's output, error) for the program channel."""
    r = run_python(program, timeout_s=PROGRAM_TIMEOUT_S, memory_mb=PROGRAM_MEMORY_MB)
    if r.undecided:
        return None, r.stderr
    if _limited(r, PROGRAM_TIMEOUT_S):
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
        text, kind = equivalence.reference(job.gold)
        return equivalence.equivalent(content, text, kind, program=c.channel == "program"), content, None
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
