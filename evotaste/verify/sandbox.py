"""Sandboxed execution of model-written Python: long-lived bubblewrap sandboxes that run one job at a time.

A sandbox sees a new, read-only root holding only read-only `/usr` (with the merged-usr symlinks), the project's
virtual environment and the uv-managed interpreter it links to, the supervisor script, a fresh `/proc`, a minimal
read-only `/dev`, and a size-capped tmpfs at `/tmp`, the jobs' working directory and the only writable place.
Nothing else of the host exists inside, so `.env`, the repository, home directories, other users' data, host `/tmp`
and host sockets are unreachable. All namespaces are unshared (no network but a private loopback, own pid and IPC
namespaces), the environment is replaced, and capabilities are dropped.

bwrap's mount setup is serialized by the kernel machine-wide (about 30 sandbox starts per second here, fewer under
other users' load), so a sandbox is kept and reused: its PID 1 is the supervisor (supervisor.py), which runs each job
as a fresh interpreter under hard per-job limits, then kills every other process in the namespace and empties /tmp.
Each thread of a process has its own sandbox (one per verifier process in the pool). A sandbox whose job timed out,
died of a signal, left a process behind, or could not be reset, or that answered irregularly, is killed and replaced.

The host keeps its own deadline on every job and kills the whole sandbox past it: bwrap runs in its own session and
is SIGKILLed with its process group; the supervisor dies with it (--die-with-parent), and with the namespace's init
the kernel kills every process left in it. bwrap is started under `setpriv --pdeathsig KILL`, so the death of the
thread that started it (as when the pool kills a stuck verifier process) also tears the sandbox down.
"""

from __future__ import annotations

import json
import math
import os
import re
import select
import signal
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from evotaste.verify import supervisor

TMPFS_BYTES = 64 << 20  # size of the writable /tmp
FILE_BYTES = 8 << 20  # largest file a job may write, stdout and stderr included (RLIMIT_FSIZE)
TASKS = 32  # processes plus threads of a job, the supervisor included (RLIMIT_NPROC, counted per user namespace)
# A job's CPU-time limit decides a timeout: CPU time does not depend on the machine's load (this account's
# processes share a quota of 32 CPUs that other jobs saturate at times, when a busy process gets a fifth of a CPU or
# less). The wall clock, WALL_FACTOR times as long, stops programs that sleep or block (a decided timeout too); but
# if the job was mostly waiting for a CPU (starved) it is run again in a fresh sandbox, up to ATTEMPTS times, and
# stays undecided if it never gets to run.
WALL_FACTOR = 20
ATTEMPTS = 3
STARVED = 0.25  # starved: waited on a run queue for at least this share of its wall-clock time
SETUP_LIMIT_S = 60  # longest wait for a new sandbox to be ready
EXIT_S = 5  # allowed after a job's wall-clock limit for the supervisor's answer
RESET_S = 30  # allowed after the answer for emptying /tmp (a late reset costs the sandbox, never the verdict)
UNDECIDED = "undecided:"  # opens the stderr of a result that decides nothing about the program
# Called with the longest wall-clock time a sandbox run may take, before it, and with None after it (the verifier pool
# keeps its own deadline for the time outside sandbox runs).
PROGRESS = None
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_MAIN = "main.py"
_SUPERVISOR = "/supervisor.py"

_ENV = {"HOME": "/tmp", "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_MAX_THREADS": "1", "MPLBACKEND": "Agg", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}


@dataclass(frozen=True)
class Result:
    returncode: int | None  # None when stopped by the wall clock; 128 + N when the program died of signal N
    stdout: str
    stderr: str
    timed_out: bool  # the job ran past its wall-clock limit, or the sandbox did not answer
    seconds: float  # wall-clock time of the job, from the start of its interpreter
    cpu_s: float = 0.0  # CPU time of all the job's processes
    undecided: bool = False  # nothing is known about the program (stderr says why, starting with UNDECIDED)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _readonly_roots() -> list[Path]:
    """The interpreter's directories bound into the sandbox: the virtual environment and the directory holding the
    base interpreter (for uv, the root of its managed Pythons, where the venv's python symlink points). Refuses any
    root that would expose the repository or a home directory as a whole."""
    roots = [Path(sys.prefix)]
    if sys.prefix != sys.base_prefix:
        roots.append(Path(sys.base_prefix).parent)
    exposed = (Path(__file__).resolve().parents[2], Path.home().resolve(), Path("/home"))
    for root in roots:
        r = root.resolve()
        if any(e == r or r in e.parents for e in exposed):
            raise RuntimeError(f"sandbox: binding {root} would expose {', '.join(map(str, exposed))}")
    return roots


def _bwrap_args() -> list[str]:
    args = ["/usr/bin/setpriv", "--pdeathsig", "KILL", "--", "/usr/bin/bwrap", "--unshare-all", "--die-with-parent",
            "--as-pid-1", "--new-session", "--clearenv", "--ro-bind", "/usr", "/usr"]
    for name in ("bin", "lib", "lib64", "sbin"):
        if Path("/", name).is_symlink():
            args += ["--symlink", os.readlink(Path("/", name)), f"/{name}"]
    for root in _readonly_roots():
        args += ["--ro-bind", str(root), str(root)]
    # bwrap's root and its /dev are uncapped tmpfs mounts of their own: made read-only, so that /tmp is the only
    # writable place (device nodes such as /dev/null still work)
    args += ["--ro-bind", supervisor.__file__, _SUPERVISOR, "--proc", "/proc", "--dev", "/dev", "--size",
             str(TMPFS_BYTES), "--tmpfs", "/tmp", "--remount-ro", "/dev", "--remount-ro", "/", "--chdir", "/tmp"]
    for k, v in _ENV.items():
        args += ["--setenv", k, v]
    return args


class SandboxError(Exception):
    """The sandbox did not start, or answered irregularly."""


class Sandbox:
    """One bwrap sandbox with its supervisor, running jobs one at a time."""

    def __init__(self):
        req_r, self.requests = os.pipe()
        self.responses, resp_w = os.pipe()
        python = str(Path(sys.prefix) / "bin" / "python")
        self.proc = subprocess.Popen(_bwrap_args() + [python, "-I", "-B", _SUPERVISOR, str(req_r), str(resp_w), python],
                                     pass_fds=(req_r, resp_w), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     start_new_session=True, env={})
        os.close(req_r)
        os.close(resp_w)
        try:
            if self._receive(time.monotonic() + SETUP_LIMIT_S) != {"ready": True}:
                raise SandboxError("sandbox: no ready message")
        except BaseException:
            self.close()
            raise

    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.proc.wait()
        for fd in (self.requests, self.responses):
            try:
                os.close(fd)
            except OSError:
                pass

    def _receive(self, deadline: float):
        def exactly(n):
            chunks = []
            while n:
                if not select.select([self.responses], [], [], max(0.0, deadline - time.monotonic()))[0]:
                    raise SandboxError("sandbox: no answer in time")
                chunk = os.read(self.responses, min(n, 1 << 20))
                if not chunk:
                    raise SandboxError("sandbox: supervisor gone")
                chunks.append(chunk)
                n -= len(chunk)
            return b"".join(chunks)
        return json.loads(exactly(struct.unpack("<Q", exactly(8))[0]))

    def answer(self, job: dict) -> dict:
        """The supervisor's answer for a job just sent; raises SandboxError if it does not answer within the job's
        limit plus EXIT_S, or answers irregularly."""
        answer = self._receive(time.monotonic() + job["timeout_s"] + EXIT_S)
        if set(answer) != {"returncode", "stdout", "stderr", "timed_out", "seconds", "cpu_s", "wait_s", "clean"}:
            raise SandboxError("sandbox: malformed answer")
        return answer


_local = threading.local()


def _discard() -> None:
    box, _local.sandbox = getattr(_local, "sandbox", None), None
    if box is not None:
        box.close()


def _attempt(job: dict) -> dict:
    """One run of the job in this thread's sandbox (a new one if needed); raises SandboxError when the sandbox cannot
    run it or does not answer properly, after discarding it. A sandbox that went away between jobs is replaced and
    the job sent again (it never started)."""
    for _ in range(2):
        box = getattr(_local, "sandbox", None)
        try:
            if box is None or not box.alive():
                _discard()
                box = _local.sandbox = Sandbox()
            supervisor.send(box.requests, job)
        except OSError:
            _discard()
            continue
        except SandboxError:
            _discard()
            raise
        break
    else:
        raise SandboxError("sandbox: could not send the job")
    try:
        answer = box.answer(job)
    except (SandboxError, OSError, ValueError) as e:
        _discard()
        raise SandboxError(str(e)) from e
    except BaseException:  # e.g. an alarm while waiting: the job's state is unknown, so the sandbox goes
        _discard()
        raise
    try:
        reset = box._receive(time.monotonic() + RESET_S)
    except (SandboxError, OSError, ValueError):
        reset = {}
    if not answer["clean"] or reset != {"reset": True}:
        _discard()
    return answer


def run_python(code: str, *, stdin: str | None = None, files: dict[str, str] | None = None, timeout_s: float,
               memory_mb: int) -> Result:
    """Run `code` as /tmp/main.py with the project's interpreter in this thread's sandbox. `files` are placed next to
    it (plain file names, read-only); `stdin` is the program's standard input (empty when None). Each process may use
    ceil(`timeout_s`) seconds of CPU time (then SIGXCPU, exit status 128 + 24) and `memory_mb` MiB of address space.
    The wall clock stops the job, with everything it started, WALL_FACTOR * `timeout_s` seconds after its interpreter
    started: a run that slept or blocked is then a timeout (`timed_out`), a starved one is repeated (ATTEMPTS in all).
    Runs that stayed starved, or a sandbox that could not run the job, give an undecided result (`undecided`, the
    reason on stderr): never a verdict on the program."""
    files = dict(files or {})
    if _MAIN in files or any(not _NAME.fullmatch(n) for n in files):
        raise ValueError(f"sandbox file names must be plain names other than {_MAIN}: {sorted(files)}")
    job = {"code": code, "files": files, "stdin": stdin or "", "timeout_s": WALL_FACTOR * timeout_s,
           "memory_mb": memory_mb, "cpu_s": math.ceil(timeout_s), "fsize": FILE_BYTES, "tasks": TASKS}
    reason = ""
    for _ in range(ATTEMPTS):
        if PROGRESS is not None:
            PROGRESS(SETUP_LIMIT_S + job["timeout_s"] + EXIT_S)
        try:
            answer = _attempt(job)
        except SandboxError as e:
            reason = str(e)
            continue
        finally:
            if PROGRESS is not None:
                PROGRESS(None)
        if not answer["timed_out"]:
            return Result(answer["returncode"], answer["stdout"], answer["stderr"], False, answer["seconds"],
                          answer["cpu_s"])
        wait = answer["wait_s"]
        if wait is not None and wait < STARVED * answer["seconds"]:  # decided: repeating cannot change a sleep
            return Result(None, answer["stdout"], f"the wall clock ran out while the program slept or blocked "
                          f"({answer['cpu_s']:.2f} s of CPU)", True, answer["seconds"], answer["cpu_s"])
        reason = f"the wall clock ran out before the CPU-time limit ({answer['cpu_s']:.2f} s of CPU in " \
                 f"{answer['seconds']:.0f} s)"
    return Result(None, "", f"{UNDECIDED} {reason}, {ATTEMPTS} attempts", True, 0.0, 0.0, True)
