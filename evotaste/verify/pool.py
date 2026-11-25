"""Persistent verifier processes with a hard wall-clock bound per job.

Each process is a fresh interpreter (`python -c`, not a fork of the caller, which may hold CUDA state or threads)
that judges one job at a time. A job still running at its deadline, for instance sympy stuck in C code on a power
tower that no signal interrupts, is ended by killing its process, which is replaced; the job's verdict is reward 0
with error "timeout". A process dies with its parent (PR_SET_PDEATHSIG), and a sandbox dies with its process.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import resource
import signal
import subprocess
import sys
import time
from multiprocessing.connection import Connection, wait

from evotaste.verify.judge import Job, Verdict, budget_s, judge

PROCESSES = 32  # verifier processes per calling process (the machine is shared: at most 48)
MARGIN_S = 20  # per-job allowance for parsing and comparison on top of the job's sandbox time
MEMORY_BYTES = 4 << 30  # address-space limit of a verifier process (sandboxes lower their own further)


def _serve(jobs: Connection, results: Connection, parent: int) -> None:
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    if os.getppid() != parent:
        return
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    while True:
        try:
            job = jobs.recv()
        except EOFError:
            return
        try:
            verdict = judge(job)
        except Exception as e:  # noqa: BLE001 - an unverifiable job scores 0 with the reason, never crashes the pool
            verdict = Verdict(0.0, False, False, None, None, f"{type(e).__name__}: {e}"[:300])
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

    def kill(self) -> None:
        self.proc.kill()
        self.proc.wait()
        self.jobs.close()
        self.results.close()


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
        """Verdicts in job order. A job's deadline is `timeout` seconds, by default MARGIN_S plus its sandbox time."""
        while len(self.procs) < min(self.size, len(jobs)):
            self.procs.append(_Process())
        out: list[Verdict | None] = [None] * len(jobs)
        todo = list(range(len(jobs)))[::-1]
        busy: dict[int, tuple[int, float]] = {}  # process slot -> (job index, deadline)
        while todo or busy:
            for slot in range(len(self.procs)):
                if slot not in busy and todo:
                    i = todo.pop()
                    self.procs[slot].jobs.send(jobs[i])
                    limit = timeout if timeout is not None else MARGIN_S + budget_s(jobs[i])
                    busy[slot] = (i, time.monotonic() + limit)
            deadline = min(d for _, d in busy.values())
            ready = wait([self.procs[s].results for s in busy], timeout=max(0.0, deadline - time.monotonic()))
            for slot in [s for s in busy if self.procs[s].results in ready]:
                i, _ = busy.pop(slot)
                try:
                    out[i] = self.procs[slot].results.recv()
                except (EOFError, OSError):
                    out[i] = Verdict(0.0, False, False, None, None,
                                     f"verifier process died ({self.procs[slot].proc.poll()})")
                    self._replace(slot)
            now = time.monotonic()
            for slot in [s for s, (_, d) in busy.items() if now >= d]:
                i, _ = busy.pop(slot)
                out[i] = Verdict(0.0, False, False, None, None, "timeout")
                self._replace(slot)
        return out  # type: ignore[return-value]

    def _replace(self, slot: int) -> None:
        self.procs[slot].kill()
        self.procs[slot] = _Process()
