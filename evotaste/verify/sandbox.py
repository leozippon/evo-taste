"""Sandboxed execution of model-written Python with bubblewrap.

The sandbox sees a new, read-only root holding only read-only `/usr` (with the merged-usr symlinks), the project's
virtual environment and the uv-managed interpreter it links to, a fresh `/proc`, a minimal read-only `/dev`, and a
size-capped tmpfs at `/tmp`, the working directory and the only writable place. Nothing else of the host exists
inside, so `.env`, the repository, home directories, other users' data, host `/tmp` and unix sockets are
unreachable. All namespaces are unshared (no network but a private loopback, own pid namespace), the environment is
replaced, capabilities are dropped, and the supplied files are bind-mounted read-only. Inside, before the program
runs, hard limits are set on address space, CPU time, file size (which also caps stdout and stderr), process and
thread count, and core dumps.

The wall-clock limit counts from the moment the program starts, which the bootstrap reports on a pipe: bwrap's
mount setup is serialized by the kernel machine-wide, so the wait for it depends on the machine's load and is given
its own, generous limit. The limit is enforced from outside: bwrap runs in its own session and is SIGKILLed with its
process group; its child, the pid namespace's init, dies with it (--die-with-parent), and with init the kernel kills
every process left in the namespace. bwrap itself is started under `setpriv --pdeathsig KILL`, so killing the calling
process (as the verifier pool does to a stuck worker) also tears the sandbox down.
"""

from __future__ import annotations

import math
import os
import re
import select
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

TMPFS_BYTES = 64 << 20  # size of the writable /tmp
FILE_BYTES = 8 << 20  # largest file the program may write, stdout and stderr included (RLIMIT_FSIZE)
TASKS = 32  # processes plus threads inside the sandbox (RLIMIT_NPROC, counted per user namespace)
SETUP_LIMIT_S = 60  # longest wait for bwrap to start the program (about 30 sandboxes start per second machine-wide)
EXIT_S = 2  # allowed after the program's time for its exit and the namespace teardown
OVERHEAD_S = SETUP_LIMIT_S + EXIT_S  # worst-case wall-clock time of a sandbox beyond the program's own limit
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_MAIN = "main.py"

# Runs first inside the sandbox: hard limits the program cannot raise, then the program as __main__ from /tmp.
_BOOT = """\
import os, resource, runpy, sys
mb, cpu, fsize, tasks, started = map(int, sys.argv[1:6])
for r, v in ((resource.RLIMIT_AS, mb << 20), (resource.RLIMIT_FSIZE, fsize), (resource.RLIMIT_NPROC, tasks),
             (resource.RLIMIT_CORE, 0)):
    resource.setrlimit(r, (v, v))
resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
sys.argv = ["main.py"]
os.write(started, b"1")
os.close(started)
del os, resource, mb, cpu, fsize, tasks, started, r, v
runpy.run_path("/tmp/main.py", run_name="__main__")
"""

_ENV = {"HOME": "/tmp", "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_MAX_THREADS": "1", "MPLBACKEND": "Agg", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}


@dataclass(frozen=True)
class Result:
    returncode: int | None  # None when killed at a time limit; 128 + N when the program died of signal N
    stdout: str
    stderr: str
    timed_out: bool  # the program ran past its limit, or bwrap did not start it within SETUP_LIMIT_S
    seconds: float  # wall-clock time from the program's start (from the launch if it never started)

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
            "--new-session", "--clearenv", "--ro-bind", "/usr", "/usr"]
    for name in ("bin", "lib", "lib64", "sbin"):
        if Path("/", name).is_symlink():
            args += ["--symlink", os.readlink(Path("/", name)), f"/{name}"]
    for root in _readonly_roots():
        args += ["--ro-bind", str(root), str(root)]
    # bwrap's root and its /dev are uncapped tmpfs mounts of their own: made read-only, so that /tmp is the only
    # writable place (device nodes such as /dev/null still work)
    args += ["--proc", "/proc", "--dev", "/dev", "--size", str(TMPFS_BYTES), "--tmpfs", "/tmp", "--remount-ro", "/dev",
             "--remount-ro", "/", "--chdir", "/tmp"]
    for k, v in _ENV.items():
        args += ["--setenv", k, v]
    return args


def _memfd(name: str, data: bytes = b"") -> int:
    fd = os.memfd_create(name)  # close-on-exec; the ones bwrap reads are passed explicitly (pass_fds)
    os.write(fd, data)
    os.lseek(fd, 0, os.SEEK_SET)
    return fd


def _read(fd: int) -> str:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while chunk := os.read(fd, 1 << 20):
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


def run_python(code: str, *, stdin: str | None = None, files: dict[str, str] | None = None, timeout_s: float,
               memory_mb: int) -> Result:
    """Run `code` as /tmp/main.py with the project's interpreter in the sandbox. `files` are placed next to it
    (plain file names, read-only); `stdin` is the program's standard input (empty when None). Each process may use
    ceil(`timeout_s`) seconds of CPU time and `memory_mb` MiB of address space; the program is killed, with
    everything it started, `timeout_s` + EXIT_S seconds after it started, or if it has not started after
    SETUP_LIMIT_S seconds."""
    files = dict(files or {})
    if _MAIN in files or any(not _NAME.fullmatch(n) for n in files):
        raise ValueError(f"sandbox file names must be plain names other than {_MAIN}: {sorted(files)}")
    fds: list[int] = []
    started_r, started_w = os.pipe()
    fds += [started_r, started_w]
    try:
        args = _bwrap_args()
        passed = [started_w]
        for name, content in ((_MAIN, code), *files.items()):
            passed.append(_memfd(name, content.encode()))
            fds.append(passed[-1])
            args += ["--ro-bind-data", str(passed[-1]), f"/tmp/{name}"]
        args += [str(Path(sys.prefix) / "bin" / "python"), "-B", "-c", _BOOT, str(memory_mb), str(math.ceil(timeout_s)),
                 str(FILE_BYTES), str(TASKS), str(started_w)]
        out, err, src = _memfd("stdout"), _memfd("stderr"), _memfd("stdin", (stdin or "").encode())
        fds += [out, err, src]
        t0 = time.monotonic()
        proc = subprocess.Popen(args, stdin=src, stdout=out, stderr=err, pass_fds=passed, start_new_session=True,
                                env={})
        os.close(started_w)
        fds.remove(started_w)
        start, note = None, ""
        try:
            ready, _, _ = select.select([started_r], [], [], SETUP_LIMIT_S)
            if ready and os.read(started_r, 1):
                start = time.monotonic()
                proc.wait(timeout=timeout_s + EXIT_S)
            elif ready:  # bwrap ended without starting the program (its error is on stderr)
                proc.wait(timeout=EXIT_S)
            else:
                note = f"\nsandbox: the program did not start within {SETUP_LIMIT_S} s"
                raise subprocess.TimeoutExpired(args, SETUP_LIMIT_S)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:  # also on an exception (e.g. an alarm) while waiting: never leave the sandbox running
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:  # exited just now
                    pass
                proc.wait()
        seconds = time.monotonic() - (start if start is not None else t0)
        return Result(None if timed_out else proc.returncode, _read(out), _read(err) + note, timed_out, seconds)
    finally:
        for fd in fds:
            os.close(fd)
