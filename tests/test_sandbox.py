"""The sandbox's guarantees, on the real bwrap: no secrets, no writes outside its tmpfs, no network, bounded time,
memory and output, nothing left running afterwards, and nothing carried from one job to the next in a reused
sandbox."""

import signal
import socket
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from evotaste.verify import sandbox, supervisor
from evotaste.verify.sandbox import EXIT_S, FILE_BYTES, TMPFS_BYTES, WALL_FACTOR, Sandbox, run_python

REPO = Path(__file__).resolve().parents[1]
PROCESSES = "import os\nprint(sorted(int(p) for p in os.listdir('/proc') if p.isdigit()) == [1, os.getpid()])"


def run(code, **kw):
    kw.setdefault("timeout_s", 10)
    kw.setdefault("memory_mb", 512)
    return run_python(code, **kw)


def current():
    return getattr(sandbox._local, "sandbox", None)


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
for p in ['/usr/x', {str(Path(sys.prefix) / 'x')!r}, {str(REPO / 'x')!r}, '/x', '/dev/x', '/dev/shm/x',
          '/supervisor.py', 'main.py', 'data.txt']:
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
    assert "wrote" not in r.stdout and r.stdout.count("refused") == 9 and r.stdout.endswith("ok\n")
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


def test_hanging_program_is_killed_with_everything_it_started_and_its_sandbox_replaced():
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
    run("pass")
    before = current()
    t0 = time.monotonic()
    r = run(code, timeout_s=0.25)  # sleeping: the wall clock (5 s) stops it, a decided timeout
    assert r.timed_out and not r.undecided and r.returncode is None
    assert time.monotonic() - t0 < WALL_FACTOR * 0.25 + EXIT_S
    assert stray(marker) == "" and current() is None and before is not None and not before.alive()


def test_fork_bomb_is_contained_and_killed():
    marker = f"evotaste-{uuid.uuid4().hex}"
    code = f"""
import os, sys
os.execv(sys.executable, [sys.executable, '-c',
         'import os\\nwhile True:\\n    try: os.fork()\\n    except OSError: pass', '{marker}'])
"""
    r = run(code, timeout_s=1, memory_mb=256)
    assert not r.ok
    assert stray(marker) == "" and run(PROCESSES).stdout == "True\n"


def test_killing_the_caller_kills_the_sandbox():
    marker = f"evotaste-{uuid.uuid4().hex}"
    inner = (f"import os, sys; os.execv(sys.executable, [sys.executable, '-c', 'import time; time.sleep(60)', "
             f"{marker!r}])")
    caller = subprocess.Popen([sys.executable, "-c", "import sys; from evotaste.verify.sandbox import run_python; "
                               "run_python(sys.argv[1], timeout_s=60, memory_mb=256)", inner])
    deadline = time.monotonic() + 30
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


def test_cpu_time_limit_kills_a_busy_process():
    """run_python sets the CPU limit to its time limit; here, through the supervisor, well below the wall clock."""
    box = Sandbox()
    try:
        job = {"code": "while True:\n    pass", "files": {}, "stdin": "", "timeout_s": 30, "memory_mb": 256,
               "cpu_s": 1, "fsize": FILE_BYTES, "tasks": 32}
        supervisor.send(box.requests, job)
        answer = box.answer(job)
        assert answer["returncode"] == 128 + signal.SIGXCPU and not answer["timed_out"] and not answer["clean"]
    finally:
        box.close()


def test_setup_of_a_sandbox_does_not_count_against_its_first_job():
    """bwrap's setup is serialized machine-wide: 48 new sandboxes queue for it, but no job may time out for that."""
    with ThreadPoolExecutor(48) as pool:
        results = list(pool.map(lambda _: run("import time\ntime.sleep(1.5)\nprint('done')", timeout_s=0.6),
                                range(48)))
    assert all(r.ok and r.stdout == "done\n" for r in results), [r for r in results if not r.ok][:2]


def test_file_names_must_be_plain():
    for bad in ("../x", "a/b", "main.py", ".hidden/.."):
        with pytest.raises(ValueError):
            run("pass", files={bad: ""})


# -- consecutive jobs in one sandbox ----------------------------------------------------------------------------------

def test_clean_jobs_reuse_the_sandbox_and_see_only_their_own_processes_and_files():
    run("pass")
    box = current()
    for _ in range(3):
        assert run(PROCESSES).stdout == "True\n"
        assert run("import os\nprint(sorted(os.listdir('/tmp')))", files={"a.txt": "x"}).stdout == \
            "['a.txt', 'main.py']\n"
    assert run("import sys\nsys.exit(3)").returncode == 3  # an ordinary failure keeps the sandbox
    assert current() is box and box.alive()


def test_a_job_that_leaves_a_process_behind_costs_its_sandbox():
    marker = f"evotaste-{uuid.uuid4().hex}"
    run("pass")
    box = current()
    r = run(f"import os, sys\nif os.fork() == 0:\n    os.setsid()\n"
            f"    os.execv(sys.executable, [sys.executable, '-c', 'import time; time.sleep(60)', '{marker}'])\n")
    assert r.ok  # the job itself exited normally
    assert stray(marker) == "" and current() is None and not box.alive()
    assert run(PROCESSES).stdout == "True\n"


def test_a_job_that_fills_tmp_leaves_the_next_job_an_empty_tmp():
    run("pass")
    box = current()
    r = run(f"import os\nos.makedirs('d/e')\nopen('.hidden', 'w').write('x')\nfor i in range(100):\n"
            f"    open(f'd/e/f{{i}}', 'wb').write(b'0' * {FILE_BYTES})")
    assert r.returncode == 1 and "No space left on device" in r.stderr
    files = TMPFS_BYTES // FILE_BYTES - 1  # almost the whole tmpfs again
    r = run(f"import os\nprint(sorted(os.listdir('/tmp')))\nfor i in range({files}):\n"
            f"    open(f'f{{i}}', 'wb').write(b'0' * {FILE_BYTES})\nprint('wrote')")
    assert r.stdout == "['main.py']\nwrote\n" and current() is box


def test_a_job_cannot_plant_files_for_a_later_job():
    run("pass")
    box = current()
    r = run("""
import os
for name, text in [('numpy.py', 'raise SystemExit(7)'), ('sitecustomize.py', 'raise SystemExit(8)'),
                   ('solution.py', 'X = "planted"'), ('test_solution.py', 'def test_ok():\\n    pass\\n')]:
    open(name, 'w').write(text)
os.chmod('main.py', 0o644)
open('main.py', 'w').write('raise SystemExit(9)')
print('planted')
""")
    assert r.stdout == "planted\n"
    r = run("import os, numpy, solution\nprint(sorted(os.listdir('.')), solution.X, numpy.__file__.startswith('/tmp'))",
            files={"solution.py": "X = 'own'"})
    assert r.ok and r.stdout == "['main.py', 'solution.py'] own False\n" and current() is box


def test_a_job_leaving_a_system_v_ipc_object_costs_its_sandbox():
    run("pass")
    box = current()
    r = run("import ctypes\nprint(ctypes.CDLL(None).shmget(0, 4096, 0o1600) >= 0)")  # IPC_PRIVATE, IPC_CREAT
    assert r.stdout == "True\n" and current() is None and not box.alive()
    assert run("print(len(open('/proc/sysvipc/shm').read().splitlines()))").stdout == "1\n"


def test_jobs_cannot_reach_or_signal_the_supervisor():
    run("pass")
    box = current()
    r = run("""
import ctypes, os, signal
for path in ('/proc/1/environ', '/proc/1/mem'):
    try:
        open(path, 'rb').read(1)
        print('read', path)
    except OSError:
        pass
try:
    os.listdir('/proc/1/fd')
    print('listed fds')
except OSError:
    pass
if ctypes.CDLL(None, use_errno=True).ptrace(16, 1, 0, 0) == 0:  # PTRACE_ATTACH
    print('attached')
for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
    os.kill(1, sig)
print('done')
""")
    assert r.ok and r.stdout == "done\n"
    assert current() is box and box.alive() and run(PROCESSES).stdout == "True\n"


def test_a_message_queue_cannot_carry_data_to_the_next_job():
    run("pass")
    box = current()
    mq = ("import ctypes, os\nrt = ctypes.CDLL('librt.so.1', use_errno=True)\n"
          "rt.mq_open.restype = ctypes.c_int\n")
    r = run(mq + "q = rt.mq_open(b'/evo_leak', os.O_CREAT | os.O_WRONLY, 0o600, None)\n"
                 "print(q, ctypes.get_errno() if q < 0 else rt.mq_send(q, b'secret', 6, 0))")
    assert r.ok and r.stdout.split()[0] == "-1"  # RLIMIT_MSGQUEUE 0: the queue cannot be created
    r = run(mq + "print(rt.mq_open(b'/evo_leak', os.O_RDONLY))")
    assert r.stdout == "-1\n" and current() is box


BUSY = "import time\nwhile time.process_time() < 0.7:\n    pass\nprint('done')"  # 0.7 s of CPU, limit 1 s


def _pinned(n_spinners: int, code: str, timeout_s: float):
    """run_python in a new thread (so a new sandbox) pinned to one CPU that n_spinners busy processes share."""
    import os
    import threading

    cpu = max(os.sched_getaffinity(0))
    spin = [subprocess.Popen(["taskset", "-c", str(cpu), sys.executable, "-c", "while True: pass"])
            for _ in range(n_spinners)]
    out = []

    def body():
        os.sched_setaffinity(0, {cpu})
        out.append(run_python(code, timeout_s=timeout_s, memory_mb=256))
    try:
        t = threading.Thread(target=body)
        t.start()
        t.join()
    finally:
        for p in spin:
            p.kill()
            p.wait()
    return out[0]


def test_a_correct_program_near_its_cpu_limit_gets_the_same_verdict_when_starved():
    idle = run(BUSY, timeout_s=1)
    starved = _pinned(11, BUSY, 1)  # about a twelfth of a CPU: 0.7 s of CPU takes several seconds of wall clock
    assert (idle.ok, idle.stdout) == (starved.ok, starved.stdout) == (True, "done\n")
    assert starved.seconds > 3 * idle.seconds and starved.cpu_s < 1


def test_a_run_starved_past_the_wall_clock_is_repeated_and_then_undecided():
    t0 = time.monotonic()
    r = _pinned(11, "while True:\n    pass", 0.1)  # 1 s of CPU, 2 s of wall clock: a twelfth of a CPU is too little
    assert r.undecided and r.timed_out and "before the CPU-time limit" in r.stderr and "3 attempts" in r.stderr
    assert time.monotonic() - t0 > 3 * 2


def test_a_slow_reset_of_tmp_costs_the_sandbox_not_the_verdict():
    run("pass")
    r = run("import os\nfor i in range(30000):\n    open(f'f{i}', 'w').close()\nprint('made')", timeout_s=10)
    assert r.ok and r.stdout == "made\n" and not r.undecided
    assert run("import os\nprint(sorted(os.listdir('/tmp')))").stdout == "['main.py']\n"
