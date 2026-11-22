"""Task instances, suites, deterministic streams, prompt building and the binary reward.

Every instance is a pure function of (stream key, family, occurrence number), so all members and all workers see
identical prompts without sharing state. Blind vaults are deliberately not loaded by `Suites`; only a `BlindReceipt`
issued by `evaluate.blind_guard` opens them (see `blind_families` and `external.items`).
"""

from __future__ import annotations

import atexit
import ctypes
import functools
import hashlib
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from pathlib import Path

import reasoning_gym
import yaml
from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer

# Some generators (polynomial_multiplication, isomorphic_strings, ransom_note, knight_swap's gold) iterate sets of
# strings, so an instance is a function of the interpreter's hash seed. Fixed at 0 by scripts/env.sh; the environment
# variable is checked so that child processes inherit it, the flag so that this process really runs with it.
if os.environ.get("PYTHONHASHSEED") != "0" or sys.flags.hash_randomization:
    raise RuntimeError("evotaste.tasks needs PYTHONHASHSEED=0 set before Python starts (source scripts/env.sh)")

SYSTEM_PROMPT = SYSTEM_PROMPTS["default"]
REWARD_THRESHOLD = 0.999  # a scorer output at or above this counts as solved
SCORE_TIMEOUT_S = 10  # wall-clock bound per scorer call on arbitrary model text; past it the call scores 0
SCORER_PROCESSES = 8  # persistent scorer processes per engine process

# reasoning-gym datasets generate item i from Random(seed + i). Some precompute items as a function of `size`
# (acre) or cap it (word_ladder), so every dataset object is created with the same fixed size and a stream walks
# through a family in chunks of CHUNK occurrences, each with its own dataset seed. Each occurrence owns REDRAWS
# consecutive indices so that a prompt colliding with a probe question can be redrawn deterministically.
CHUNK = 512
REDRAWS = 16
DATASET_SIZE = CHUNK * REDRAWS


@dataclass(frozen=True)
class Task:
    uid: str  # "<family>/<dataset seed>/<index>" (procedural) or "<source>/<item id>" (external)
    family: str  # reasoning-gym dataset name or external source name
    question: str
    answer: str  # gold answer as text, for records; the scorer may accept other correct answers
    entry: dict | None = field(default=None, compare=False, repr=False)  # reasoning-gym entry; None for external


@dataclass(frozen=True)
class Score:
    reward: float  # 1.0 if parsed, untruncated and scored >= REWARD_THRESHOLD, else 0.0
    score: float  # raw scorer output; 0.0 when not scored (unparsed or truncated) or the scorer failed
    parsed: bool  # an <answer>...</answer> span was found
    truncated: bool  # generation stopped for any reason other than an end-of-sequence token
    extracted: str | None
    error: str | None = None  # scorer exception or timeout


def stable_seed(*parts) -> int:
    """31-bit seed that is a pure function of its arguments (no Python hash randomization)."""
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


@functools.lru_cache(maxsize=None)
def _dataset(family: str, seed: int):
    return reasoning_gym.create_dataset(family, size=DATASET_SIZE, seed=seed)


def procedural_task(family: str, seed: int, index: int) -> Task:
    ds = _dataset(family, seed)
    random.seed(stable_seed("global-random", family, seed, index))  # list_functions also draws from module `random`
    entry = ds[index]
    return Task(f"{family}/{seed}/{index}", family, entry["question"], str(entry["answer"]), entry)


def instance(key: str, family: str, n: int, attempt: int = 0) -> Task:
    """Occurrence n of `family` in the stream `key` (attempt > 0 gives its deterministic redraws)."""
    if not 0 <= attempt < REDRAWS:
        raise ValueError(f"attempt {attempt} outside [0, {REDRAWS})")
    chunk, pos = divmod(n, CHUNK)
    return procedural_task(family, stable_seed(key, family, chunk), pos * REDRAWS + attempt)


# ---------------------------------------------------------------------------------------------------------------
# Scoring


def raw_score(task: Task, answer: str) -> float:
    """The family's own scorer on an extracted answer. Runs inside a scorer process (see score_many)."""
    if task.entry is None:
        from evotaste import external

        return float(external.SCORERS[task.family](answer, task.answer))
    family, seed, _ = task.uid.split("/")
    return float(_dataset(family, int(seed)).score_answer(answer, task.entry))


def _serve(jobs: Connection, results: Connection, parent: int) -> None:
    """Scorer process: (task, answer) in, (raw score, error) out, until the parent goes away."""
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG: die with the parent, even mid-scorer
    if os.getppid() != parent:  # the parent died before the line above
        return
    while True:
        try:
            task, answer = jobs.recv()
        except EOFError:
            return
        try:
            results.send((raw_score(task, answer), None))
        except Exception as e:  # noqa: BLE001 - any scorer failure must score 0, never crash a worker
            results.send((0.0, f"{type(e).__name__}: {e}"[:200]))


class _ScorerProcess:
    def __init__(self):
        jobs_r, jobs_w = os.pipe()
        results_r, results_w = os.pipe()
        self.proc = subprocess.Popen([sys.executable, "-m", "evotaste.tasks", str(jobs_r), str(results_w),
                                      str(os.getpid())], pass_fds=(jobs_r, results_w))
        os.close(jobs_r)
        os.close(results_w)
        self.jobs, self.results = Connection(jobs_w, readable=False), Connection(results_r, writable=False)

    def kill(self) -> None:
        self.proc.kill()
        self.proc.wait()
        self.jobs.close()
        self.results.close()


class ScorerPool:
    """Persistent scorer processes. A job running longer than SCORE_TIMEOUT_S, which a signal could not stop inside
    C code (e.g. sympy on '1e999999999'), is ended by killing its process; the process is replaced."""

    def __init__(self, size: int = SCORER_PROCESSES):
        self.size = size
        self.procs: list[_ScorerProcess] = []
        atexit.register(self.close)

    def close(self) -> None:
        for p in self.procs:
            p.kill()
        self.procs = []

    def run(self, jobs: list[tuple[Task, str]]) -> list[tuple[float, str | None]]:
        while len(self.procs) < min(self.size, len(jobs)):
            self.procs.append(_ScorerProcess())
        out: list = [None] * len(jobs)
        todo = list(range(len(jobs)))[::-1]
        busy: dict[int, tuple[int, float]] = {}  # process slot -> (job index, start time)
        while todo or busy:
            for slot in range(len(self.procs)):
                if slot not in busy and todo:
                    i = todo.pop()
                    self.procs[slot].jobs.send(jobs[i])
                    busy[slot] = (i, time.monotonic())
            deadline = min(t0 for _, t0 in busy.values()) + SCORE_TIMEOUT_S
            ready = wait([self.procs[s].results for s in busy], timeout=max(0.0, deadline - time.monotonic()))
            for slot in [s for s in busy if self.procs[s].results in ready]:
                i, _ = busy.pop(slot)
                try:
                    out[i] = self.procs[slot].results.recv()
                except (EOFError, OSError):  # the process died (e.g. out of memory)
                    out[i] = (0.0, f"scorer process died ({self.procs[slot].proc.poll()})")
                    self._replace(slot)
            now = time.monotonic()
            for slot in [s for s, (_, t0) in busy.items() if now - t0 >= SCORE_TIMEOUT_S]:
                i, _ = busy.pop(slot)
                out[i] = (0.0, "timeout")
                self._replace(slot)
        return out

    def _replace(self, slot: int) -> None:
        self.procs[slot].kill()
        self.procs[slot] = _ScorerProcess()


_POOL: ScorerPool | None = None


def score_many(jobs: list[tuple[Task, str, str]]) -> list[Score]:
    """Binary rewards of (task, completion text, finish reason) triples: the last <answer> span is scored only if
    present and the completion stopped by itself; scorer exceptions and timeouts give 0 with `error` set."""
    global _POOL
    parsed = [(extract_answer(text), finish != "stop") for _, text, finish in jobs]
    to_score = [i for i, (extracted, truncated) in enumerate(parsed) if extracted is not None and not truncated]
    raw = {}
    if to_score:
        _POOL = _POOL or ScorerPool()
        raw = dict(zip(to_score, _POOL.run([(jobs[i][0], parsed[i][0]) for i in to_score])))
    out = []
    for i, (extracted, truncated) in enumerate(parsed):
        if i not in raw:
            out.append(Score(0.0, 0.0, extracted is not None, truncated, extracted))
        else:
            value, error = raw[i]
            out.append(Score(float(value >= REWARD_THRESHOLD), value, True, False, extracted, error))
    return out


def score(task: Task, text: str, finish_reason: str) -> Score:
    return score_many([(task, text, finish_reason)])[0]


def check_gold(tasks: list[Task]) -> None:
    """Reward integrity on real instances: every gold answer wrapped in <answer> tags must earn reward 1."""
    scores = score_many([(t, f"<answer>{t.answer}</answer>", "stop") for t in tasks])
    bad = [t.uid for t, s in zip(tasks, scores) if s.reward != 1.0]
    if bad:
        raise RuntimeError(f"{len(bad)} of {len(tasks)} gold answers do not earn reward 1, e.g. {bad[:5]}")


# ---------------------------------------------------------------------------------------------------------------
# Prompts


def load_scaffolds(path: str | Path) -> dict[str, str]:
    """Scaffold name -> the text appended to a seeding teacher's system message (preamble + habit)."""
    data = yaml.safe_load(Path(path).read_text())
    named = {**data["control"], **data["scaffolds"]}
    return {name: f"{data['preamble']} {v['text']}" for name, v in named.items()}


def system_prompt(scaffold_text: str | None = None) -> str:
    return SYSTEM_PROMPT if scaffold_text is None else f"{SYSTEM_PROMPT}\n{scaffold_text}\n"


def prompt_ids(tokenizer, question: str, system: str = SYSTEM_PROMPT) -> list[int]:
    """Chat-template token ids with thinking disabled; the trainer scores exactly these ids."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer(text, add_special_tokens=False).input_ids


# ---------------------------------------------------------------------------------------------------------------
# Suites


class Suites:
    """The non-blind part of configs/suites.yaml: train and selection families and the foundation sources."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        data = yaml.safe_load(self.path.read_text())
        self.train: tuple[str, ...] = tuple(data["suites"]["train"])
        self.selection: tuple[str, ...] = tuple(data["suites"]["selection"])
        self.foundation: tuple[str, ...] = tuple(data["external"]["foundation"])
        if not self.train or set(self.train) & set(self.selection):
            raise ValueError(f"{self.path}: train suite empty or overlapping with selection")


class BlindReceipt:
    """Proof that the blind guard passed and the ledger line was written; created only by evaluate.blind_guard."""

    def __init__(self, vault: str, ledger_line: str):
        self.vault = vault
        self.ledger_line = ledger_line


def check_receipt(receipt: BlindReceipt | None, vault: str) -> None:
    if not isinstance(receipt, BlindReceipt) or receipt.vault != vault:
        raise PermissionError(f"blind vault {vault!r} can only be opened through evaluate.blind_guard")


def blind_suite(path: str | Path, vault: str, receipt: BlindReceipt) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(procedural families, external sources) of a blind vault."""
    check_receipt(receipt, vault)
    data = yaml.safe_load(Path(path).read_text())
    return tuple(data["suites"][vault]), tuple(data["external"][vault])


# ---------------------------------------------------------------------------------------------------------------
# Streams


def family_at(families: tuple[str, ...], key: str, j: int) -> tuple[str, int]:
    """Family of stream item j and its occurrence number: each block of len(families) items is a seeded permutation."""
    block, pos = divmod(j, len(families))
    order = random.Random(stable_seed(key, "order", block)).sample(sorted(families), len(families))
    return order[pos], block


def fixed_tasks(families, key: str, per_family: int) -> list[Task]:
    """The first `per_family` instances of each family in stream `key` (probe and evaluation sets)."""
    return [instance(key, f, n) for f in families for n in range(per_family)]


class Streams:
    """Training stream, seeding pool and probe set of one run. Probes do not depend on the run seed."""

    def __init__(self, suites: Suites, run_seed: int, probes):
        from evotaste import external

        self.suites = suites
        self.run_seed = run_seed
        self.probe_tasks: dict[str, list[Task]] = {
            "train": fixed_tasks(suites.train, "probe", probes.train_per_family),
            "selection": fixed_tasks(suites.selection, "probe", probes.selection_per_family),
            "foundation": [t for s in suites.foundation
                           for t in external.items(s, "foundation")[:probes.foundation_per_source]],
        }
        self._probe_questions = {t.question for t in self.probe_tasks["train"]}

    def _train_item(self, key: str, j: int) -> Task:
        family, n = family_at(self.suites.train, key, j)
        for attempt in range(REDRAWS):
            task = instance(key, family, n, attempt)
            if task.question not in self._probe_questions:
                return task
        raise RuntimeError(f"{key} item {j} ({family}): {REDRAWS} draws all equal probe questions")

    def train_batch(self, step: int, prompts: int) -> list[Task]:
        """Prompts of training step `step`; identical for every member with the same batch size."""
        key = f"train/{self.run_seed}"
        return [self._train_item(key, j) for j in range(step * prompts, (step + 1) * prompts)]

    def seed_pool(self, size: int) -> list[Task]:
        key = f"seedpool/{self.run_seed}"
        return [self._train_item(key, j) for j in range(size)]


if __name__ == "__main__":  # a scorer process started by ScorerPool
    from evotaste.tasks import _serve as serve  # the importable module, whose dataset cache the scorers use

    serve(Connection(int(sys.argv[1]), writable=False), Connection(int(sys.argv[2]), readable=False), int(sys.argv[3]))
