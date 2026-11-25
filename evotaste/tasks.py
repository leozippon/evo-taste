"""Tasks, prompts and scoring; suites, probe and evaluation sets, and the training stream; the blind vault's receipt.

A `Task` is one prompt: a dataset item (`external.items`) or a procedural instance (reasoning-gym, evaluation only),
with the answer instruction of its kind and wrapper appended to the problem. `score_many` verifies answers through
`evotaste.verify` (procedural instances through their family's own scorer). The training stream is a pure function of
(run seed, step) over the census-stratified training part, shared by every member of a run. The vault (`blind2`) is
opened only with a `BlindReceipt`, which only `evaluate.blind_guard` issues.
"""

from __future__ import annotations

import atexit
import ctypes
import dataclasses
import functools
import hashlib
import logging
import math
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from multiprocessing.connection import Connection, wait
from pathlib import Path

import yaml

# Some reasoning-gym generators (polynomial_multiplication, isomorphic_strings, ransom_note, knight_swap's gold)
# iterate sets of strings, so an instance is a function of the interpreter's hash seed. Fixed at 0 by scripts/env.sh;
# the environment variable is checked so that child processes inherit it, the flag so that this process runs with it.
if os.environ.get("PYTHONHASHSEED") != "0" or sys.flags.hash_randomization:
    raise RuntimeError("evotaste.tasks needs PYTHONHASHSEED=0 set before Python starts (source scripts/env.sh)")

log = logging.getLogger(__name__)

SYSTEM_PROMPT = "Reason step by step, then give your final answer in exactly the form the problem asks for."
PROGRAM_OFFER = ("or end with one Python program in a fenced block whose printed output is the final answer; "
                 "whichever comes last counts")
PROGRAM_ONLY = ("End your response with one Python program in a fenced block (```python ... ```) whose printed "
                "output is the final answer.")
BOTH_CHANNELS = ("direct", "program")
VAULT = "blind2"

# Training stream (docs/dev/ROUND2_SPEC.md, WP-D): shares are exact fractions so that the stream's cumulative counts
# follow them to within one prompt per cell.
DOMAIN_MIX = {"math": Fraction(2, 5), "code": Fraction(1, 4), "logic": Fraction(7, 20)}
STRATUM_MIX = {"sweet": Fraction(7, 10), "mastered": Fraction(3, 20), "frontier": Fraction(3, 20)}
VARIANT_SHARE = Fraction(3, 20)  # of non-code prompts, which carry a training-variant wrapper
MIN_POOL = 50  # a mastered or frontier pool with fewer items gives its share to the source's sweet pool
PROBE_ITEMS = 300  # about this many selection items in the probe set, spread over its units
PROBE_WRAPPED = 60  # further selection items presented with held-out wrappers

REWARD_THRESHOLD = 0.999  # a procedural scorer output at or above this counts as solved
SCORE_TIMEOUT_S = 10  # wall-clock bound per procedural scorer call; past it the call scores 0
BUILD_TIMEOUT_S = 300  # bound on building a family's dataset in a scorer process (acre precomputes 8,192 items)
GOLD_TIMEOUT_S = 300  # scorer bound when checking gold answers; a timeout there is an error, never a verdict
_READY = "ready"
SCORER_PROCESSES = 8


@dataclass(frozen=True)
class Task:
    uid: str  # "<source>/<item id>", or "<family>/<dataset seed>/<index>" for a procedural instance
    family: str  # the source id (procedural: the reasoning-gym family)
    domain: str  # "math" | "code" | "logic"
    kind: str  # "open" | "choice" | "verdict" | "code" | "procedural"
    question: str  # the full user turn: the problem and its answer instruction
    # open: the reference expression; choice: {"options", "answer" (letter)}; verdict: {"labels", "answer"};
    # code: {"harness", "tests", "entry_point", "signature", "prelude", "solution"}; procedural: the reasoning-gym
    # entry, or {"exact", "ignore_spaces"} for a string answer compared exactly.
    gold: object = field(compare=False, repr=False)
    wrapper: str = "box"  # id of the answer wrapper in force (evotaste.verify.WRAPPERS)
    stratum: str | None = None  # "mastered" | "sweet" | "frontier" in the training stream, else None
    channels: tuple[str, ...] = ()  # answer channels of an open task: ("direct", "program"), or one of them
    problem: str = field(default="", compare=False, repr=False)  # the question without its answer instruction


@dataclass(frozen=True)
class Score:
    reward: float  # 1.0 iff the answer is in the wrapper asked for, verified correct, and the completion stopped
    correct_any: bool  # verified correct under tolerant extraction (any known wrapper)
    complied: bool  # a final answer was found in the wrapper asked for
    channel: str | None  # "direct" | "program" for kind open, else None
    truncated: bool  # generation stopped for any reason other than an end-of-sequence token
    extracted: str | None
    error: str | None = None


def stable_seed(*parts) -> int:
    """31-bit seed that is a pure function of its arguments (no Python hash randomization)."""
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


# ---------------------------------------------------------------------------------------------------------------
# Answer instructions and prompts


def default_wrapper(kind: str) -> str:
    return "code" if kind == "code" else "box"


def default_channels(kind: str, wrapper: str) -> tuple[str, ...]:
    """An open task asked for a box may also answer by program; any other wrapper has the direct channel only."""
    if kind != "open":
        return ()
    return BOTH_CHANNELS if wrapper == "box" else ("direct",)


def instruction(kind: str, gold, wrapper: str, channels: tuple[str, ...]) -> str:
    """The answer instruction appended to a problem."""
    from evotaste.verify import WRAPPERS

    if kind == "open" and channels == ("program",):
        return PROGRAM_ONLY
    ask = WRAPPERS[wrapper].instruction
    if kind == "open" and channels == BOTH_CHANNELS:
        return f"{ask.rstrip('.')}, {PROGRAM_OFFER}."
    if kind == "choice":
        return f"Answer with the letter of the correct option. {ask}"
    if kind == "verdict":
        return f"Answer with exactly one of: {', '.join(gold['labels'])}. {ask}"
    if kind == "code":
        entry = gold["entry_point"]
        if entry.startswith("Solution()."):
            target = "the class `Solution` of the starter code"
        elif gold.get("signature"):
            target = f"`{gold['signature'].removesuffix(':')}`"
        else:
            target = f"the function `{entry}`"
        return f"{ask} It must define {target}."
    return ask


def make_task(item, wrapper: str | None = None, stratum: str | None = None,
              channels: tuple[str, ...] | None = None) -> Task:
    """The task of a dataset item (`sources.Item`) under a wrapper (default: the kind's) and answer channels
    (default: both for an open task asked for a box)."""
    wrapper = wrapper or default_wrapper(item.kind)
    channels = default_channels(item.kind, wrapper) if channels is None else tuple(channels)
    question = f"{item.problem}\n\n{instruction(item.kind, item.gold, wrapper, channels)}"
    return Task(item.uid, item.source, item.domain, item.kind, question, item.gold, wrapper, stratum, channels,
                item.problem)


def rewrap(task: Task, wrapper: str | None = None, channels: tuple[str, ...] | None = None,
           stratum: str | None = None) -> Task:
    """The same problem under another wrapper or set of channels (a guardrail variant, a forced channel)."""
    wrapper = wrapper or task.wrapper
    channels = default_channels(task.kind, wrapper) if channels is None else tuple(channels)
    question = f"{task.problem}\n\n{instruction(task.kind, task.gold, wrapper, channels)}"
    return dataclasses.replace(task, question=question, wrapper=wrapper, channels=channels,
                               stratum=stratum if stratum is not None else task.stratum)


def prompt_ids(tokenizer, question: str, system: str = SYSTEM_PROMPT, context: str | None = None) -> list[int]:
    """Chat-template token ids with thinking disabled; a seeding context (the teacher's) is appended to the system
    prompt. The trainer scores exactly these ids."""
    if context:
        system = f"{system}\n\n{context}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer(text, add_special_tokens=False).input_ids


# ---------------------------------------------------------------------------------------------------------------
# Procedural families (reasoning-gym): evaluation-only sources. Every instance is a pure function of (stream key,
# family, occurrence); datasets are created with one fixed size and walked in chunks with their own seeds, and each
# occurrence owns REDRAWS consecutive indices so that an unusable draw is replaced deterministically.

CHUNK = 512
REDRAWS = 16
DATASET_SIZE = CHUNK * REDRAWS


@functools.lru_cache(maxsize=None)
def _dataset(family: str, seed: int):
    import reasoning_gym

    return reasoning_gym.create_dataset(family, size=DATASET_SIZE, seed=seed)


def procedural_task(family: str, seed: int, index: int, domain: str = "logic") -> Task:
    ds = _dataset(family, seed)
    random.seed(stable_seed("global-random", family, seed, index))  # list_functions also draws from module `random`
    entry = ds[index]
    from evotaste.verify import WRAPPERS

    question = f"{entry['question']}\n\n{WRAPPERS['box'].instruction}"
    return Task(f"{family}/{seed}/{index}", family, domain, "procedural", question, entry, "box", None, (),
                entry["question"])


_GENERATOR_FAILURES: set[tuple[str, int, int]] = set()


def draws(key: str, family: str, n: int, domain: str = "logic"):
    """The draws of occurrence n of `family` in stream `key`, in their fixed order; a draw whose generator raises is
    skipped (logged once per process)."""
    chunk, pos = divmod(n, CHUNK)
    seed = stable_seed(key, family, chunk)
    for attempt in range(REDRAWS):
        index = pos * REDRAWS + attempt
        try:
            yield procedural_task(family, seed, index, domain)
        except Exception as e:  # noqa: BLE001 - a generator bug must cost one draw, not a worker
            if (family, seed, index) not in _GENERATOR_FAILURES:
                _GENERATOR_FAILURES.add((family, seed, index))
                log.warning("generator of %s failed for seed %d index %d (%s: %s); using the next draw",
                            family, seed, index, type(e).__name__, str(e)[:200])


def procedural_tasks(families: dict[str, str], key: str, per_family: int) -> list[Task]:
    """The first `per_family` usable instances of each family (family -> domain) in stream `key`: a position whose
    draw raises in its generator or has a gold answer its own scorer rejects takes the position's next draw."""
    positions = [(f, n) for f in sorted(families) for n in range(per_family)]
    candidates = {p: draws(key, p[0], p[1], families[p[0]]) for p in positions}
    chosen = {p: next(candidates[p], None) for p in positions}
    pending = positions
    while pending:
        exhausted = [p for p in pending if chosen[p] is None]
        if exhausted:
            raise RuntimeError(f"{key}: no usable draw among {REDRAWS} for {exhausted[:5]}")
        ok = gold_ok_many([chosen[p] for p in pending])
        pending = [p for p, good in zip(pending, ok) if not good]
        for p in pending:
            log.warning("gold answer of %s is rejected by its own scorer; using the next draw", chosen[p].uid)
            chosen[p] = next(candidates[p], None)
    return [chosen[p] for p in positions]


def _exact(answer: str, gold: dict) -> float:
    norm = (lambda s: "".join(s.split())) if gold["ignore_spaces"] else (lambda s: " ".join(s.split()))
    return float(norm(answer) == norm(gold["exact"]))


def raw_score(task: Task, answer: str) -> float:
    """A procedural task's own scorer on an extracted answer. Runs inside a scorer process (see ScorerPool)."""
    if "exact" in task.gold:
        return _exact(answer, task.gold)
    family, seed, _ = task.uid.split("/")
    return float(_dataset(family, int(seed)).score_answer(answer, task.gold))


def _dataset_key(task: Task) -> str | None:
    return None if "exact" in task.gold else task.uid.rsplit("/", 1)[0]


def _serve(jobs: Connection, results: Connection, parent: int) -> None:
    """Scorer process: (task, answer) in, (raw score, error) out, until the parent goes away."""
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG: die with the parent, even mid-scorer
    if os.getppid() != parent:
        return
    built = set()
    while True:
        try:
            task, answer = jobs.recv()
        except EOFError:
            return
        try:
            key = _dataset_key(task)
            if key is not None and key not in built:
                family, seed = key.split("/")
                _dataset(family, int(seed))  # building is not part of the scorer's time limit
                built.add(key)
                results.send(_READY)
            results.send((raw_score(task, answer), None))
        except Exception as e:  # noqa: BLE001 - a scorer failure scores 0, never crashes a worker
            results.send((0.0, f"{type(e).__name__}: {e}"[:200]))


class _ScorerProcess:
    def __init__(self):
        jobs_r, jobs_w = os.pipe()
        results_r, results_w = os.pipe()
        self.proc = subprocess.Popen([sys.executable, "-m", "evotaste.tasks", str(jobs_r), str(results_w),
                                      str(os.getpid())], pass_fds=(jobs_r, results_w))
        os.close(jobs_r)
        os.close(results_w)
        self.built: set = set()
        self.jobs, self.results = Connection(jobs_w, readable=False), Connection(results_r, writable=False)

    def kill(self) -> None:
        self.proc.kill()
        self.proc.wait()
        self.jobs.close()
        self.results.close()


class ScorerPool:
    """Persistent scorer processes for procedural families. A job running past its timeout (sympy inside C code
    cannot be interrupted by a signal) is ended by killing its process, which is replaced."""

    def __init__(self, size: int = SCORER_PROCESSES):
        self.size = size
        self.procs: list[_ScorerProcess] = []
        atexit.register(self.close)

    def close(self) -> None:
        for p in self.procs:
            p.kill()
        self.procs = []

    def run(self, jobs: list[tuple[Task, str]], timeout: float | None = None) -> list[tuple[float, str | None]]:
        timeout = SCORE_TIMEOUT_S if timeout is None else timeout
        while len(self.procs) < min(self.size, len(jobs)):
            self.procs.append(_ScorerProcess())
        out: list = [None] * len(jobs)
        todo = list(range(len(jobs)))[::-1]
        busy: dict[int, tuple[int, float]] = {}
        while todo or busy:
            for slot in range(len(self.procs)):
                if slot not in busy and todo:
                    i = todo.pop()
                    proc, task = self.procs[slot], jobs[i][0]
                    proc.jobs.send(jobs[i])
                    key = _dataset_key(task)
                    builds = key is not None and key not in proc.built
                    proc.built.add(key)
                    busy[slot] = (i, time.monotonic() + (BUILD_TIMEOUT_S if builds else timeout))
            deadline = min(d for _, d in busy.values())
            ready = wait([self.procs[s].results for s in busy], timeout=max(0.0, deadline - time.monotonic()))
            for slot in [s for s in busy if self.procs[s].results in ready]:
                i, _ = busy[slot]
                try:
                    message = self.procs[slot].results.recv()
                except (EOFError, OSError):
                    busy.pop(slot)
                    out[i] = (0.0, f"scorer process died ({self.procs[slot].proc.poll()})")
                    self._replace(slot)
                    continue
                if message == _READY:
                    busy[slot] = (i, time.monotonic() + timeout)
                else:
                    busy.pop(slot)
                    out[i] = message
            now = time.monotonic()
            for slot in [s for s, (_, d) in busy.items() if now >= d]:
                i, _ = busy.pop(slot)
                out[i] = (0.0, "timeout")
                self._replace(slot)
        return out

    def _replace(self, slot: int) -> None:
        self.procs[slot].kill()
        self.procs[slot] = _ScorerProcess()


_POOL: ScorerPool | None = None


def _procedural_scores(jobs: list[tuple[Task, str, str]], timeout: float | None) -> list[Score]:
    """Strict: the content of the wrapper asked for, scored by the family's scorer; lenient (correct_any): the
    content of any answer wrapper scores at or above the threshold."""
    global _POOL
    from evotaste.verify import WRAPPERS
    from evotaste.verify.wrappers import ANSWER_KINDS

    tolerant = [w for w in WRAPPERS.values() if w.kinds == ANSWER_KINDS]
    contents = []  # per job: (strict content, [contents of any wrapper])
    for task, text, _ in jobs:
        strict = WRAPPERS[task.wrapper].find(text)
        found = [f.content for w in tolerant if (f := w.find(text)) is not None]
        contents.append((strict.content if strict else None, list(dict.fromkeys(found))))
    calls = sorted({(i, c) for i, (s, any_) in enumerate(contents) for c in ([s] if s else []) + any_})
    pooled = [(i, c) for i, c in calls if _dataset_key(jobs[i][0]) is not None]
    results = {}
    if pooled:
        _POOL = _POOL or ScorerPool()
        results = dict(zip(pooled, _POOL.run([(jobs[i][0], c) for i, c in pooled], timeout)))
    for i, c in calls:
        if (i, c) not in results:
            results[(i, c)] = (raw_score(jobs[i][0], c), None)
    out = []
    for i, ((task, _, finish), (strict, any_)) in enumerate(zip(jobs, contents)):
        value, error = results[(i, strict)] if strict else (0.0, None)
        solved = value >= REWARD_THRESHOLD
        out.append(Score(float(solved and finish == "stop"), any(results[(i, c)][0] >= REWARD_THRESHOLD for c in any_),
                         strict is not None, None, finish != "stop", strict, error))
    return out


def score_many(jobs: list[tuple[Task, str, str]], timeout: float | None = None) -> list[Score]:
    """Scores of (task, completion text, finish reason) triples. The reward requires the wrapper asked for, a verified
    correct answer and a completion that stopped by itself; verifier errors and timeouts score 0 with `error` set."""
    from evotaste.verify import Job, verify_many

    out: list[Score | None] = [None] * len(jobs)
    procedural = [i for i, (t, _, _) in enumerate(jobs) if t.kind == "procedural"]
    if procedural:
        for i, s in zip(procedural, _procedural_scores([jobs[i] for i in procedural], timeout)):
            out[i] = s
    verified = [i for i, (t, _, _) in enumerate(jobs) if t.kind != "procedural"]
    if verified:
        verdicts = verify_many([Job(jobs[i][0].kind, jobs[i][1], jobs[i][0].gold, jobs[i][0].wrapper,
                                    jobs[i][0].channels or ("direct",)) for i in verified], timeout)
        for i, v in zip(verified, verdicts):
            stopped = jobs[i][2] == "stop"
            out[i] = Score(float(v.reward == 1.0 and stopped), bool(v.correct_any), bool(v.complied), v.channel,
                           not stopped, v.extracted, v.error)
    return out


def score(task: Task, text: str, finish_reason: str) -> Score:
    return score_many([(task, text, finish_reason)])[0]


GOLD_ATTEMPTS = 3  # a reference check that times out is repeated; one that always times out counts as failed


def _undecided(reason: str | None) -> bool:
    return reason is not None and (reason.endswith("timeout") or "process died" in reason)


def gold_reasons(tasks: list[Task]) -> list[str | None]:
    """For each task None if its reference verifies against itself (code: the reference solution passes its tests;
    procedural: the family's scorer accepts its gold answer), else why not. A check that times out or loses its
    verifier process decides nothing, so it is repeated, up to GOLD_ATTEMPTS times; a reference that times out every
    time is reported as "timeout", and a verifier process that keeps dying is an error, never a verdict."""
    from evotaste.verify import WRAPPERS, gold_check

    out: list[str | None] = [None] * len(tasks)
    procedural = [i for i, t in enumerate(tasks) if t.kind == "procedural"]
    if procedural:
        jobs = [(dataclasses.replace(tasks[i], wrapper="box"),
                 WRAPPERS["box"].render(str(tasks[i].gold.get("exact", tasks[i].gold.get("answer")))), "stop")
                for i in procedural]
        for i, s in zip(procedural, _procedural_scores(jobs, GOLD_TIMEOUT_S)):
            out[i] = None if s.reward == 1.0 else (s.error or "gold answer not accepted by its scorer")
    pending = [i for i, t in enumerate(tasks) if t.kind != "procedural"]
    for _ in range(GOLD_ATTEMPTS):
        if not pending:
            break
        for i, reason in zip(pending, gold_check([(tasks[i].kind, tasks[i].gold) for i in pending])):
            out[i] = reason
        pending = [i for i in pending if _undecided(out[i])]
    died = sum("process died" in out[i] for i in pending)
    if died:  # counts only: the generator checks vault items too
        raise RuntimeError(f"the verifier process died {GOLD_ATTEMPTS} times while checking {died} references")
    return out


def gold_ok_many(tasks: list[Task]) -> list[bool]:
    return [r is None for r in gold_reasons(tasks)]


def check_gold(tasks: list[Task]) -> None:
    """Reward integrity before any generation: every task's reference must verify against itself."""
    bad = [(t.uid, r) for t, r in zip(tasks, gold_reasons(tasks)) if r is not None]
    if bad:
        raise RuntimeError(f"{len(bad)} of {len(tasks)} references do not verify against themselves, e.g. {bad[:3]}")


# ---------------------------------------------------------------------------------------------------------------
# Suites and the vault's receipt


class Suites:
    """The open parts of configs/suites-r2.yaml: training sources, the dataset sources with selection items, and the
    procedural selection families. The vault's units are never read here."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        data = yaml.safe_load(self.path.read_text())
        self.train: tuple[str, ...] = tuple(data["suites"]["train"])
        self.selection: tuple[str, ...] = tuple(data["suites"]["selection"])
        self.procedural: dict[str, str] = dict(data["procedural"]["selection"])  # family -> domain
        if not self.train or not self.selection:
            raise ValueError(f"{self.path}: no training or no selection source")

    def items(self, source: str, part: str) -> list[Task]:
        from evotaste import external

        return external.items(source, part, suites=self.path)


class BlindReceipt:
    """Proof that the blind guard passed and the ledger line was written; created only by evaluate.blind_guard."""

    def __init__(self, vault: str, ledger_line: str):
        self.vault = vault
        self.ledger_line = ledger_line


def check_receipt(receipt: BlindReceipt | None, vault: str) -> None:
    if not isinstance(receipt, BlindReceipt) or receipt.vault != vault:
        raise PermissionError(f"blind vault {vault!r} can only be opened through evaluate.blind_guard")


def blind_suite(path: str | Path, vault: str, receipt: BlindReceipt) -> tuple[dict[str, str], tuple[str, ...]]:
    """(procedural families -> domain, dataset sources) of the vault."""
    check_receipt(receipt, vault)
    data = yaml.safe_load(Path(path).read_text())
    return dict(data["procedural"][vault]), tuple(data["suites"][vault])


# ---------------------------------------------------------------------------------------------------------------
# Probe and evaluation sets (independent of the run seed)


def _probe_layout(suites: Suites) -> tuple[int, dict[str, int]]:
    """Items per probe unit (each selection source and the procedural families together) and, per non-code selection
    source, how many items after those carry a held-out wrapper (round robin to PROBE_WRAPPED)."""
    per_unit = round(PROBE_ITEMS / (len(suites.selection) + 1))
    room = {}  # non-code sources: items left after the plain probe items
    for s in suites.selection:
        part = suites.items(s, "selection")
        if part and part[0].domain != "code":
            room[s] = max(0, len(part) - per_unit)
    wrapped = dict.fromkeys(suites.selection, 0)
    while sum(wrapped.values()) < PROBE_WRAPPED and any(room[s] > wrapped[s] for s in room):
        for s in sorted(room):
            if sum(wrapped.values()) < PROBE_WRAPPED and room[s] > wrapped[s]:
                wrapped[s] += 1
    return per_unit, wrapped


def probe_set(suites: Suites) -> list[Task]:
    """About PROBE_ITEMS selection items spread over the selection sources (the first items of each part's order)
    and the procedural families (one instance each, stream `probe`), then PROBE_WRAPPED further items of the non-code
    sources under held-out wrappers, cycling through them."""
    from evotaste.verify import HELDOUT_WRAPPERS

    per_unit, wrapped = _probe_layout(suites)
    plain = [t for s in suites.selection for t in suites.items(s, "selection")[:per_unit]]
    per_family = max(1, math.ceil(per_unit / max(1, len(suites.procedural))))
    plain += procedural_tasks(suites.procedural, "probe", per_family) if suites.procedural else []
    extra = [t for s in suites.selection for t in suites.items(s, "selection")[per_unit:per_unit + wrapped[s]]]
    return plain + [rewrap(t, HELDOUT_WRAPPERS[k % len(HELDOUT_WRAPPERS)]) for k, t in enumerate(extra)]


def fixed_tasks(suites: Suites, part: str, per_source: int, per_family: int | None = None,
                receipt: BlindReceipt | None = None) -> list[Task]:
    """Evaluation set of a part: for each dataset source its last `per_source` items in the part's order (excluding
    the probe's items; the whole remainder if smaller), and `per_family` (default `per_source`) instances of each
    procedural family (stream `eval`). The vault needs the guard's receipt."""
    from evotaste import external

    per_family = per_source if per_family is None else per_family
    if part == "selection":
        families, srcs = suites.procedural, suites.selection
        per_unit, wrapped = _probe_layout(suites)
        skip = {s: per_unit + wrapped[s] for s in srcs}
    elif part == VAULT:
        families, srcs = blind_suite(suites.path, VAULT, receipt)
        skip = dict.fromkeys(srcs, 0)
    else:
        raise ValueError(f"no evaluation set for part {part!r}")
    out = []
    for s in srcs:
        part_items = external.items(s, part, receipt, suites=suites.path)[skip[s]:]
        out += part_items[max(0, len(part_items) - per_source):]
    return out + (procedural_tasks(families, "eval", per_family) if families else [])


# ---------------------------------------------------------------------------------------------------------------
# Strata (from the census) and the training stream


def stratum_of(passes: int, samples: int) -> str:
    """frontier: never solved; mastered: solved in all or all but one samples; sweet: in between."""
    if passes == 0:
        return "frontier"
    return "mastered" if passes >= samples - 1 else "sweet"


def load_strata(path: str | Path, sha256: str | None = None) -> dict[str, str]:
    """uid -> stratum from a census strata table (lines `uid<TAB>passes<TAB>samples`), refused unless its SHA-256
    is the pinned one."""
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if sha256 is not None and digest != sha256:
        raise ValueError(f"{path}: SHA-256 {digest} is not the pinned {sha256}")
    out = {}
    for line in data.decode().splitlines()[1:]:
        uid, passes, samples = line.split("\t")
        out[uid] = stratum_of(int(passes), int(samples))
    return out


def _round_half_up(x: Fraction) -> int:
    return math.floor(x + Fraction(1, 2))


def _apportion(shares: list[Fraction], n: int, carry: list[Fraction], rng: random.Random) -> list[int]:
    """n split by shares (summing to 1) with the carried remainders of earlier calls: largest remainder, ties in a
    seeded order. Updates `carry`, so that cumulative counts stay within one of their shares."""
    quota = [s * n + c for s, c in zip(shares, carry)]
    base = [math.floor(q) for q in quota]
    ties = [rng.random() for _ in quota]
    for c in sorted(range(len(quota)), key=lambda c: (-(quota[c] - base[c]), ties[c]))[:n - sum(base)]:
        base[c] += 1
    carry[:] = [q - b for q, b in zip(quota, base)]
    return base


class _Mix:
    """Batches drawn from weighted pools in groups (domains). Each batch is split first among the groups, then within
    each group among its cells, both by `_apportion`: every batch follows the group shares to within one prompt, and
    over the first t batches a cell receives its share of them to within two. Each pool is walked in a seeded
    permutation, without replacement until it is exhausted, then in a new permutation."""

    def __init__(self, key: str, cells: list[tuple[str, str, Fraction, list[Task]]]):
        total = sum(w for _, _, w, _ in cells)
        self.key = key
        self.names = [c for c, _, _, _ in cells]
        self.groups = sorted({g for _, g, _, _ in cells})
        self.group_of = [self.groups.index(g) for _, g, _, _ in cells]
        self.weights = [w / total for _, _, w, _ in cells]
        self.group_weights = [sum(w for w, g in zip(self.weights, self.group_of) if g == k)
                              for k in range(len(self.groups))]
        self.pools = [sorted(p, key=lambda t: t.uid) for _, _, _, p in cells]
        empty = [c for c, p in zip(self.names, self.pools) if not p]
        if empty:
            raise ValueError(f"{key}: empty pools {empty}")
        self._history: dict[int, tuple[list[list[int]], list[Fraction], list[Fraction]]] = {}
        self._perms: dict[tuple[int, int], list[int]] = {}

    def counts(self, batch: int, n: int) -> list[list[int]]:
        """Counts per cell for batches 0..batch (inclusive) of size n."""
        hist, group_carry, carry = self._history.setdefault(
            n, ([], [Fraction(0)] * len(self.groups), [Fraction(0)] * len(self.weights)))
        while len(hist) <= batch:
            rng = random.Random(stable_seed(self.key, "ties", n, len(hist)))
            per_group = _apportion(self.group_weights, n, group_carry, rng)
            counts = [0] * len(self.weights)
            for k, m in enumerate(per_group):
                members = [c for c, g in enumerate(self.group_of) if g == k]
                sub_carry = [carry[c] for c in members]
                split = _apportion([self.weights[c] / self.group_weights[k] for c in members], m, sub_carry, rng)
                for c, x, cc in zip(members, split, sub_carry):
                    counts[c], carry[c] = x, cc
            hist.append(counts)
        return hist[:batch + 1]

    def _item(self, c: int, k: int) -> Task:
        pool = self.pools[c]
        epoch, pos = divmod(k, len(pool))
        perm = self._perms.get((c, epoch))
        if perm is None:
            perm = random.Random(stable_seed(self.key, self.names[c], epoch)).sample(range(len(pool)), len(pool))
            self._perms[(c, epoch)] = perm
        return pool[perm[pos]]

    def batch(self, b: int, n: int) -> list[tuple[int, Task]]:
        counts = self.counts(b, n)
        before = [sum(h[c] for h in counts[:-1]) for c in range(len(self.pools))]
        return [(c, self._item(c, before[c] + j)) for c in range(len(self.pools)) for j in range(counts[-1][c])]


class Streams:
    """Training stream, seeding prompts and probe set of one run. The stream needs the census strata (uid ->
    stratum, `load_strata`); probes and seeding prompts do not."""

    def __init__(self, suites: Suites, run_seed: int, *, strata: dict[str, str] | None = None):
        from evotaste import sources

        self.suites, self.run_seed, self.strata = suites, run_seed, strata
        self.domain = {s: sources.SOURCES[s].domain for s in suites.train}
        self._train: _Mix | None = None
        self._seed: dict[tuple, _Mix] = {}

    @functools.cached_property
    def probe_tasks(self) -> list[Task]:
        return probe_set(self.suites)

    def _train_mix(self) -> _Mix:
        if self._train is not None:
            return self._train
        if self.strata is None:
            raise ValueError("the training stream needs the census strata table (Streams(..., strata=...))")
        train_uids = {t.uid for s in self.suites.train for t in self.suites.items(s, "train")}
        stray = [u for u in self.strata if u not in train_uids]
        if stray:
            raise ValueError(f"the strata table names {len(stray)} items outside the training part, e.g. {stray[:3]}")
        cells = []
        for domain, share in DOMAIN_MIX.items():
            srcs = [s for s in self.suites.train if self.domain[s] == domain]
            for s in srcs:
                pools = {st: [] for st in STRATUM_MIX}
                for t in self.suites.items(s, "train"):
                    if t.uid in self.strata:
                        pools[self.strata[t.uid]].append(t)
                weights = dict(STRATUM_MIX)
                for st in ("mastered", "frontier"):
                    if len(pools[st]) < MIN_POOL:
                        weights["sweet"] += weights.pop(st)
                if len(pools["sweet"]) < MIN_POOL:
                    raise ValueError(f"{s}: only {len(pools['sweet'])} sweet-spot items; the stream cannot follow "
                                     "its mix")
                cells += [(f"{s}/{st}", domain, share / len(srcs) * w,
                           [dataclasses.replace(t, stratum=st) for t in pools[st]]) for st, w in weights.items()]
        self._train = _Mix(f"train/{self.run_seed}", cells)
        return self._train

    def train_batch(self, step: int, prompts: int) -> list[Task]:
        """The prompts of training step `step`, the same for every member with this batch size: the domain, source
        and stratum mix of the spec, and VARIANT_SHARE of the non-code prompts under a training-variant wrapper."""
        from evotaste.verify import TRAIN_WRAPPERS

        mix = self._train_mix()
        drawn = [t for _, t in mix.batch(step, prompts)]
        code = [self.domain[name.split("/")[0]] == "code" for name in mix.names]
        noncode = [sum(n for c, n in enumerate(h) if not code[c]) for h in mix.counts(step, prompts)]
        variants = _round_half_up(VARIANT_SHARE * sum(noncode)) - _round_half_up(VARIANT_SHARE * sum(noncode[:-1]))
        rng = random.Random(stable_seed(mix.key, "batch", prompts, step))
        slots = [i for i, t in enumerate(drawn) if t.domain != "code"]
        for i in rng.sample(slots, variants):
            drawn[i] = rewrap(drawn[i], rng.choice(TRAIN_WRAPPERS))
        rng.shuffle(drawn)
        return drawn

    def seed_batch(self, i: int, n: int, domains: tuple[str, ...] | None = None) -> list[Task]:
        """Prompts of seeding update i: the stream's domain shares (restricted to `domains`), equal shares for the
        sources of a domain, default wrappers, drawn from the whole training part."""
        domains = tuple(sorted(domains)) if domains else tuple(DOMAIN_MIX)
        if not set(domains) <= set(DOMAIN_MIX):
            raise ValueError(f"unknown domains {domains}")
        mix = self._seed.get(domains)
        if mix is None:
            cells = []
            for domain in domains:
                srcs = [s for s in self.suites.train if self.domain[s] == domain]
                cells += [(s, domain, DOMAIN_MIX[domain] / len(srcs), self.suites.items(s, "train")) for s in srcs]
            mix = self._seed[domains] = _Mix(f"seed/{self.run_seed}/{','.join(domains)}", cells)
        drawn = [t for _, t in mix.batch(i, n)]
        random.Random(stable_seed(mix.key, "order", n, i)).shuffle(drawn)
        return drawn


if __name__ == "__main__":  # a scorer process started by ScorerPool
    from evotaste.tasks import _serve as serve  # the importable module, whose dataset cache the scorers use

    serve(Connection(int(sys.argv[1]), writable=False), Connection(int(sys.argv[2]), readable=False), int(sys.argv[3]))
