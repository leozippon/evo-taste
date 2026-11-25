"""Sandboxed execution of model-written Python with bubblewrap.

The sandbox sees a new, empty root: read-only `/usr` (with the merged-usr symlinks), the project's virtual
environment and the uv-managed interpreter it links to, a fresh `/proc` and minimal `/dev`, and a size-capped tmpfs at
`/tmp`, which is the working directory. Nothing else of the host exists inside, so `.env`, the repository, home
directories, other users' data, host `/tmp` and unix sockets are unreachable. All namespaces are unshared (no network
but a private loopback, own pid namespace), the environment is cleared, capabilities are dropped, and the supplied
files are bind-mounted read-only. Inside, before the program runs, hard limits are set on address space, CPU time,
file size (which also caps stdout and stderr), process and thread count, and core dumps.

The wall-clock limit is enforced from outside: bwrap runs in its own session and is SIGKILLed with its process group;
its child, the pid namespace's init, dies with it (--die-with-parent), and with init the kernel kills every process
left in the namespace. bwrap itself is started under `setpriv --pdeathsig KILL`, so killing the calling process (as
the verifier pool does to a stuck worker) also tears the sandbox down.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

TMPFS_BYTES = 64 << 20  # size of the writable /tmp
FILE_BYTES = 8 << 20  # largest file the program may write, stdout and stderr included (RLIMIT_FSIZE)
TASKS = 32  # processes plus threads inside the sandbox (RLIMIT_NPROC, counted per user namespace)
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_MAIN = "main.py"

# Runs first inside the sandbox: hard limits the program cannot raise, then the program as __main__ from /tmp.
_BOOT = """\
import resource, runpy, sys
mb, cpu, fsize, tasks = map(int, sys.argv[1:5])
for r, v in ((resource.RLIMIT_AS, mb << 20), (resource.RLIMIT_FSIZE, fsize), (resource.RLIMIT_NPROC, tasks),
             (resource.RLIMIT_CORE, 0)):
    resource.setrlimit(r, (v, v))
resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
sys.argv = ["main.py"]
del resource, mb, cpu, fsize, tasks, r, v
runpy.run_path("/tmp/main.py", run_name="__main__")
"""

_ENV = {"HOME": "/tmp", "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_MAX_THREADS": "1", "MPLBACKEND": "Agg", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}


@dataclass(frozen=True)
class Result:
    returncode: int | None  # None when killed at the wall-clock limit; negative: killed by that signal
    stdout: str
    stderr: str
    timed_out: bool
    seconds: float

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
    args += ["--proc", "/proc", "--dev", "/dev", "--size", str(TMPFS_BYTES), "--tmpfs", "/tmp", "--chdir", "/tmp"]
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
    (plain file names, read-only); `stdin` is the program's standard input (empty when None). The program is killed,
    with everything it started, after `timeout_s` seconds of wall-clock time; its address space is limited to
    `memory_mb` MiB and its CPU time to about `timeout_s`."""
    files = dict(files or {})
    if _MAIN in files or any(not _NAME.fullmatch(n) for n in files):
        raise ValueError(f"sandbox file names must be plain names other than {_MAIN}: {sorted(files)}")
    fds: list[int] = []
    try:
        args = _bwrap_args()
        for name, content in ((_MAIN, code), *files.items()):
            fds.append(_memfd(name, content.encode()))
            args += ["--ro-bind-data", str(fds[-1]), f"/tmp/{name}"]
        cpu = int(timeout_s) + 1
        args += [str(Path(sys.prefix) / "bin" / "python"), "-B", "-c", _BOOT, str(memory_mb), str(cpu),
                 str(FILE_BYTES), str(TASKS)]
        out, err = _memfd("stdout"), _memfd("stderr")
        fds += [out, err]
        src = _memfd("stdin", (stdin or "").encode())
        fds.append(src)
        t0 = time.monotonic()
        proc = subprocess.Popen(args, stdin=src, stdout=out, stderr=err, pass_fds=fds[:len(files) + 1],
                                start_new_session=True, env={})
        try:
            proc.wait(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:  # also on an exception (e.g. the pool's alarm) while waiting: never leave the sandbox running
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:  # exited just now
                    pass
                proc.wait()
        seconds = time.monotonic() - t0
        return Result(None if timed_out else proc.returncode, _read(out), _read(err), timed_out, seconds)
    finally:
        for fd in fds:
            os.close(fd)
