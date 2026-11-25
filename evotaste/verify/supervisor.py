"""PID 1 of a long-lived sandbox: runs one job at a time, each in a fresh interpreter, and resets the sandbox after it.

Started by sandbox.py inside bwrap as `python -I -B /supervisor.py <request fd> <response fd> <python>`; standard
library only. Messages are length-prefixed JSON. Per job: write the files read-only into the empty /tmp, start the
interpreter with the job bootstrap (hard limits on address space, CPU, file size, task count and POSIX message queue
bytes), wait for it up to its wall-clock limit, then kill every other process in the namespace, reap them, and empty
/tmp. The answer, sent before /tmp is emptied so that a slow reset never delays a verdict, gives the job's CPU time
(all its processes, reaped) and, when the wall clock ran out, how long its main process had been waiting for a CPU,
and says whether the sandbox is still clean; a second message says whether the reset worked. The host discards the
sandbox after a timeout, a death by signal, a process left behind, a failed or late reset, or left-over System V IPC
objects.

The supervisor is not dumpable, so jobs (same uid, no capabilities) cannot ptrace it or open its /proc entries, and it
installs no signal handlers, so as the namespace's init it ignores every signal a job can send it.
"""

import ctypes
import json
import os
import resource
import shutil
import signal
import struct
import subprocess
import sys
import time

# Runs first in every job: hard limits it cannot raise, then /tmp/main.py as __main__.
BOOT = """\
import resource, runpy, sys
mb, cpu, fsize, tasks = map(int, sys.argv[1:5])
for r, v in ((resource.RLIMIT_AS, mb << 20), (resource.RLIMIT_FSIZE, fsize), (resource.RLIMIT_NPROC, tasks),
             (resource.RLIMIT_CORE, 0), (resource.RLIMIT_MSGQUEUE, 0)):
    resource.setrlimit(r, (v, v))
resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
sys.argv = ["main.py"]
del resource, mb, cpu, fsize, tasks, r, v
runpy.run_path("/tmp/main.py", run_name="__main__")
"""
REAP_S = 2  # time allowed for killed processes to disappear


def send(fd: int, obj) -> None:
    data = json.dumps(obj).encode()
    data = struct.pack("<Q", len(data)) + data
    while data:
        data = data[os.write(fd, data):]


def receive(fd: int):
    """The next message, or None at end of file."""
    def exactly(n):
        chunks = []
        while n:
            chunk = os.read(fd, min(n, 1 << 20))
            if not chunk:
                return None
            chunks.append(chunk)
            n -= len(chunk)
        return b"".join(chunks)
    head = exactly(8)
    return None if head is None else json.loads(exactly(struct.unpack("<Q", head)[0]))


def others() -> list[int]:
    return [int(p) for p in os.listdir("/proc") if p.isdigit() and p != "1"]


def kill_all() -> bool:
    """Kill and reap every process but this one; whether none is left."""
    deadline = time.monotonic() + REAP_S
    while True:
        try:
            os.kill(-1, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            while os.waitpid(-1, os.WNOHANG)[0] > 0:
                pass
        except ChildProcessError:
            pass
        if not others():
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(0.002)


def reset_tmp() -> bool:
    """Empty /tmp; whether it is empty."""
    try:
        for name in os.listdir("/tmp"):
            path = os.path.join("/tmp", name)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.unlink(path)
        return os.listdir("/tmp") == []
    except OSError:
        return False


def no_ipc() -> bool:
    """No System V shared memory, message queue or semaphore set exists in the sandbox's IPC namespace."""
    for kind in ("shm", "msg", "sem"):
        with open(f"/proc/sysvipc/{kind}") as f:
            if len(f.read().splitlines()) > 1:
                return False
    return True


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def waiting_seconds(pid: int) -> float | None:
    """How long the process has waited on a run queue (from /proc/<pid>/schedstat), or None."""
    try:
        with open(f"/proc/{pid}/schedstat") as f:
            return int(f.read().split()[1]) / 1e9
    except (OSError, ValueError, IndexError):
        return None


def memfd(name: str, data: bytes = b"") -> int:
    fd = os.memfd_create(name)
    os.write(fd, data)
    os.lseek(fd, 0, os.SEEK_SET)
    return fd


def read(fd: int) -> str:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while chunk := os.read(fd, 1 << 20):
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


def run(job: dict, python: str) -> dict:
    for name, content in [("main.py", job["code"]), *job["files"].items()]:
        path = os.path.join("/tmp", name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        os.chmod(path, 0o444)
    fds = [memfd("stdin", job["stdin"].encode()), memfd("stdout"), memfd("stderr")]
    try:
        cpu0, wait = cpu_seconds(), None
        t0 = time.monotonic()
        proc = subprocess.Popen([python, "-B", "-c", BOOT, str(job["memory_mb"]), str(job["cpu_s"]), str(job["fsize"]),
                                 str(job["tasks"])], cwd="/tmp", stdin=fds[0], stdout=fds[1], stderr=fds[2])
        try:
            returncode = proc.wait(timeout=job["timeout_s"])
            timed_out = False
        except subprocess.TimeoutExpired:
            returncode, timed_out, wait = None, True, waiting_seconds(proc.pid)
        seconds = time.monotonic() - t0
        left_behind = bool(others())
        reaped = kill_all()
        cpu = cpu_seconds() - cpu0
        if returncode is not None and returncode < 0:
            returncode = 128 - returncode  # as bwrap reports a signal death
        stdout, stderr = read(fds[1]), read(fds[2])
    finally:
        for fd in fds:
            os.close(fd)
    clean = reaped and not left_behind and not timed_out and (returncode or 0) < 128
    return {"returncode": returncode, "stdout": stdout, "stderr": stderr, "timed_out": timed_out, "seconds": seconds,
            "cpu_s": cpu, "wait_s": wait, "clean": clean}


def main() -> None:
    requests, responses, python = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    ctypes.CDLL(None).prctl(4, 0)  # PR_SET_DUMPABLE: not traceable by jobs, /proc entries closed to them
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # no handler at all: init ignores what the namespace sends it
    send(responses, {"ready": True})
    while (job := receive(requests)) is not None:
        send(responses, run(job, python))
        send(responses, {"reset": reset_tmp() and no_ipc()})


if __name__ == "__main__":
    main()
