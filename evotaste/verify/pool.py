"""Persistent verifier processes with a CPU-time budget and a wall-clock bound per job.

Each process is a fresh interpreter (`python -c`, not a fork of the caller, which may hold CUDA state or threads)
that judges one job at a time. Outside sandbox runs (which bound themselves, report when they start and end, and do
not inherit the budget), a job may spend COMPARISON_CPU_S seconds of CPU time in the process: parsing and comparison.
The budget is a soft RLIMIT_CPU, so the kernel ends a process that exhausts it even inside C code (sympy on a power
tower); that verdict is decided, not verified, because CPU time does not depend on the machine's load. The wall clock
outside sandbox runs is MARGIN_S, WALL_FACTOR times the budget: a process it stops with budget to spare was starved,
and that verdict is undecided. A process that crashes (segmentation fault, abort) on a job gives a decided verdict
too; one killed by anything else (the kernel's out-of-memory killer) an undecided one. A process dies with its parent
(PR_SET_PDEATHSIG), and a sandbox dies with its process.
"""

from __future__ import annotations

import atexit
import ctypes
import math
import os
import resource
import signal
import subprocess
import sys
import time
from multiprocessing.connection import Connection, wait

from evotaste.verify import sandbox
from evotaste.verify.judge import Job, Verdict, judge, unfinished
from evotaste.verify.sandbox import UNDECIDED, WALL_FACTOR

PROCESSES = 32  # verifier processes per calling process (the machine is shared: at most 48)
COMPARISON_CPU_S = 10  # CPU seconds a job may spend in its verifier process outside sandbox runs
MARGIN_S = WALL_FACTOR * COMPARISON_CPU_S  # wall-clock seconds outside sandbox runs before a job counts as starved
MEMORY_BYTES = 4 << 30  # address-space limit of a verifier process (sandboxes lower their own further)
CRASHES = {signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGFPE, signal.SIGILL}
_TICK = os.sysconf("SC_CLK_TCK")


def _cpu() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _serve(jobs: Connection, results: Connection, parent: int) -> None:
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    if os.getppid() != parent:
        return
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    budget = {"left": 0.0, "since": 0.0}

    def arm():  # the kernel sends SIGXCPU (default action: terminate) once the budget is spent
        budget["since"] = _cpu()
        resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(budget["since"] + max(budget["left"], 0.0)),
                                                 resource.RLIM_INFINITY))

    def disarm():
        budget["left"] -= _cpu() - budget["since"]
        resource.setrlimit(resource.RLIMIT_CPU, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    def progress(seconds):  # a sandbox run (started under it) has its own limits and must not inherit the budget
        if seconds is not None:
            disarm()
        results.send(("sandbox", seconds))
        if seconds is None:
            arm()

    sandbox.PROGRESS = progress
    while True:
        try:
            job = jobs.recv()
        except EOFError:
            return
        budget["left"] = COMPARISON_CPU_S
        arm()
        try:
            verdict = judge(job)
        except Exception as e:  # noqa: BLE001 - an unverifiable job scores 0 with the reason, never crashes the pool
            verdict = Verdict(0.0, False, False, None, None, f"{type(e).__name__}: {e}"[:300])
        disarm()
        results.send(verdict)


_WORKER = ("import sys\nfrom multiprocessing.connection import Connection\nfrom evotaste.verify.pool import _serve\n"
           "_serve(Connection(int(sys.argv[1]), writable=False), Connection(int(sys.argv[2]), readable=False), "
           "int(sys.argv[3]))\n")


class _Process:
    def __init__(self):
        jobs_r, jobs_w = os.pipe()
        results_r, results_w = os.pipe()
        self.proc = subprocess.Popen([sys.executable, "-c", _WORKER, str(jobs_r), str(results_w), str(os.getpid())],
                                     pass_fds=(jobs_r, results_w))
        os.close(jobs_r)
        os.close(results_w)
        self.jobs, self.results = Connection(jobs_w, readable=False), Connection(results_r, writable=False)

    def cpu(self) -> float:
        """CPU seconds the process has used so far (its own, not its sandboxes')."""
        try:
            with open(f"/proc/{self.proc.pid}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            return (int(fields[11]) + int(fields[12])) / _TICK
        except (OSError, IndexError, ValueError):
            return 0.0

    def kill(self) -> None:
        self.proc.kill()
        self.proc.wait()
        self.jobs.close()
        self.results.close()


def _after_death(job: Job, returncode: int | None) -> Verdict:
    """The verdict for a job during which its verifier process died on its own."""
    if returncode == -signal.SIGXCPU:
        return unfinished(job, f"comparison exceeded its CPU budget ({COMPARISON_CPU_S} s)")
    if returncode is not None and -returncode in CRASHES:
        return unfinished(job, f"comparison crashed the verifier process ({signal.Signals(-returncode).name})")
    return Verdict(0.0, False, False, None, None, f"{UNDECIDED} verifier process died ({returncode})")


class Pool:
    def __init__(self, size: int = PROCESSES):
        if not 1 <= size <= 48:
            raise ValueError(f"pool size {size} outside 1..48")
        self.size = size
        self.procs: list[_Process] = []
        atexit.register(self.close)

    def close(self) -> None:
        for p in self.procs:
            p.kill()
        self.procs = []

    def run(self, jobs: list[Job], timeout: float | None = None) -> list[Verdict]:
        """Verdicts in job order. A job's wall-clock deadline is `timeout` seconds in all if given, else MARGIN_S
        outside its sandbox runs and, during each run, the longest time the run may take; past it the job was
        starved (its CPU budget is enforced by the kernel) and its verdict is undecided."""
        while len(self.procs) < min(self.size, len(jobs)):
            self.procs.append(_Process())
        out: list[Verdict | None] = [None] * len(jobs)
        todo = list(range(len(jobs)))[::-1]
        busy: dict[int, tuple[int, float, float, float]] = {}  # slot -> (job index, deadline, start, start CPU)
        while todo or busy:
            for slot in range(len(self.procs)):
                if slot not in busy and todo:
                    i = todo.pop()
                    self.procs[slot].jobs.send(jobs[i])
                    now = time.monotonic()
                    busy[slot] = (i, now + (timeout if timeout is not None else MARGIN_S), now, self.procs[slot].cpu())
            deadline = min(b[1] for b in busy.values())
            ready = wait([self.procs[s].results for s in busy], timeout=max(0.0, deadline - time.monotonic()))
            for slot in [s for s in busy if self.procs[s].results in ready]:
                i = busy[slot][0]
                try:
                    message = self.procs[slot].results.recv()
                except (EOFError, OSError):
                    busy.pop(slot)
                    try:
                        returncode = self.procs[slot].proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        returncode = None
                    out[i] = _after_death(jobs[i], returncode)
                    self._replace(slot)
                    continue
                if isinstance(message, Verdict):
                    busy.pop(slot)
                    out[i] = message
                elif timeout is None:  # ("sandbox", longest run time, or None after the run)
                    busy[slot] = (i, time.monotonic() + (MARGIN_S if message[1] is None else message[1]),
                                  *busy[slot][2:])
            now = time.monotonic()
            for slot in [s for s, b in busy.items() if now >= b[1]]:
                i, _, start, cpu0 = busy.pop(slot)
                used = self.procs[slot].cpu() - cpu0
                out[i] = Verdict(0.0, False, False, None, None, f"{UNDECIDED} verifier deadline: the verifier process "
                                 f"used {used:.1f} s of CPU in {now - start:.0f} s (starved)")
                self._replace(slot)
        return out  # type: ignore[return-value]

    def _replace(self, slot: int) -> None:
        self.procs[slot].kill()
        self.procs[slot] = _Process()
