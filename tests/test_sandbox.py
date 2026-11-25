"""The sandbox's guarantees, on the real bwrap: no secrets, no writes outside its tmpfs, no network, bounded time,
memory and output, and nothing left running afterwards."""

import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from evotaste.verify.sandbox import FILE_BYTES, run_python

REPO = Path(__file__).resolve().parents[1]


def run(code, **kw):
    kw.setdefault("timeout_s", 10)
    kw.setdefault("memory_mb", 512)
    return run_python(code, **kw)


def stray(marker: str) -> str:
    time.sleep(0.3)
    return subprocess.run(["pgrep", "-af", marker], capture_output=True, text=True).stdout.strip()


def test_runs_code_with_files_and_stdin():
    r = run("import sys, os\nprint(open('data.txt').read(), sys.stdin.read(), sorted(os.listdir('.')))",
            files={"data.txt": "x"}, stdin="in")
    assert r.ok and r.stdout == "x in ['data.txt', 'main.py']\n"


def test_cannot_read_secrets_repository_or_home():
    assert (REPO / ".env").is_file()  # the secret exists on the host
    probe = f"""
import os
for p in [{str(REPO / '.env')!r}, {str(Path.home() / '.bashrc')!r}, '/etc/passwd', {str(REPO / 'evotaste')!r}]:
    print(p, os.path.exists(p))
print(sorted(os.listdir({str(REPO)!r})), sorted(os.listdir({str(Path.home())!r})))
"""
    r = run(probe)
    assert r.ok, r.stderr
    lines = r.stdout.splitlines()
    assert all(line.endswith(" False") for line in lines[:4]), r.stdout
    assert lines[4] == "['.venv'] ['.local']"  # only the interpreter's directories are bound back


def test_cannot_write_outside_its_tmpfs():
    r = run(f"""
import os
for p in ['/usr/x', {str(Path(sys.prefix) / 'x')!r}, {str(REPO / 'x')!r}, '/x', '/dev/x', '/dev/shm/x', 'main.py',
          'data.txt']:
    try:
        open(p, 'w').write('x')
        print('wrote', p)
    except OSError as e:
        print('refused', p, e.errno)
open('/dev/null', 'w').write('discarded')
open('/tmp/scratch', 'w').write('ok')
print(open('/tmp/scratch').read())
""", files={"data.txt": "x"})
    assert r.ok, r.stderr
    assert "wrote" not in r.stdout and r.stdout.count("refused") == 8 and r.stdout.endswith("ok\n")
    assert not (REPO / "x").exists() and not (Path(sys.prefix) / "x").exists()


def test_no_network():
    """Neither the outside, nor a host service on loopback, nor a host abstract unix socket is reachable."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    port = server.getsockname()[1]
    unix = socket.socket(socket.AF_UNIX)
    unix.bind("\0evotaste-test-abstract")
    unix.listen()
    try:
        r = run(f"""
import socket
for addr in [('127.0.0.1', {port}), ('1.1.1.1', 80)]:
    s = socket.socket()
    s.settimeout(2)
    try:
        s.connect(addr)
        print('connected', addr)
    except OSError as e:
        print('refused', e.errno)
u = socket.socket(socket.AF_UNIX)
try:
    u.connect('\\0' + 'evotaste-test-abstract')
    print('connected unix')
except OSError as e:
    print('refused', e.errno)
""")
    finally:
        server.close()
        unix.close()
    assert r.ok, r.stderr
    assert "connected" not in r.stdout and r.stdout.count("refused") == 3


def test_hanging_program_is_killed_with_everything_it_started():
    marker = f"evotaste-{uuid.uuid4().hex}"
    code = f"""
import os, subprocess, sys, time
if os.fork() == 0:  # a detached grandchild in its own session
    os.setsid()
    if os.fork() == 0:
        os.execv(sys.executable, [sys.executable, '-c', 'import time; time.sleep(60)', '{marker}'])
    os._exit(0)
subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', '{marker}'], start_new_session=True)
time.sleep(60)
"""
    t0 = time.monotonic()
    r = run(code, timeout_s=2)
    assert r.timed_out and r.returncode is None and time.monotonic() - t0 < 5
    assert stray(marker) == ""


def test_fork_bomb_is_contained_and_killed():
    marker = f"evotaste-{uuid.uuid4().hex}"
    code = f"""
import os, sys
try:
    os.execv(sys.executable, [sys.executable, '-c', 'import os\\nwhile True:\\n    try: os.fork()\\n    except OSError: pass', '{marker}'])
except OSError:
    pass
"""
    t0 = time.monotonic()
    r = run(code, timeout_s=2, memory_mb=256)
    assert r.timed_out and time.monotonic() - t0 < 5
    assert stray(marker) == ""


def test_killing_the_caller_kills_the_sandbox():
    marker = f"evotaste-{uuid.uuid4().hex}"
    inner = f"import os, sys; os.execv(sys.executable, [sys.executable, '-c', 'import time; time.sleep(60)', {marker!r}])"
    caller = subprocess.Popen([sys.executable, "-c", "import sys; from evotaste.verify.sandbox import run_python; "
                               "run_python(sys.argv[1], timeout_s=60, memory_mb=256)", inner])
    deadline = time.monotonic() + 10
    while marker not in subprocess.run(["pgrep", "-af", marker], capture_output=True, text=True).stdout:
        assert time.monotonic() < deadline, "the sandboxed sleeper never started"
        time.sleep(0.1)
    caller.kill()
    caller.wait()
    assert stray(marker) == ""


def test_memory_and_output_are_bounded():
    r = run("x = bytearray(600 << 20)", memory_mb=512)
    assert r.returncode != 0 and "MemoryError" in r.stderr
    r = run("while True:\n    print('x' * 1000)")
    assert r.returncode != 0 and not r.timed_out and len(r.stdout) <= FILE_BYTES
    r = run(f"for i in range(100):\n    open(f'f{{i}}', 'wb').write(b'0' * {FILE_BYTES})")  # past the tmpfs size
    assert r.returncode != 0 and "No space left on device" in r.stderr


def test_busy_loop_is_killed_at_the_wall_clock_limit():
    t0 = time.monotonic()
    r = run("while True:\n    pass", timeout_s=2)
    assert r.timed_out and time.monotonic() - t0 < 4


def test_file_names_must_be_plain():
    for bad in ("../x", "a/b", "main.py", ".hidden/.."):
        with pytest.raises(ValueError):
            run("pass", files={bad: ""})
