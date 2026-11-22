"""Small file helpers shared by the worker and the evaluator. Whole files are written atomically (temp + rename)."""

from __future__ import annotations

import datetime
import gzip
import json
import os
from pathlib import Path


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    atomic_write(path, (json.dumps(obj, indent=1) + "\n").encode())


def read_json(path: Path):
    return json.loads(path.read_text())


def write_jsonl_gz(path: Path, records) -> None:
    atomic_write(path, gzip.compress("".join(json.dumps(r) + "\n" for r in records).encode(), compresslevel=6))


def read_jsonl_gz(path: Path) -> list[dict]:
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]


def append_jsonl(path: Path, record) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Records of an appended jsonl file; a final line without newline (a write cut short by a kill) is dropped."""
    if not path.exists():
        return []
    lines = path.read_text().split("\n")
    return [json.loads(line) for line in lines[:-1]]  # the last element is "" or an incomplete line


def filter_jsonl(path: Path, keep) -> None:
    """Rewrite a jsonl file keeping only the records for which keep(record) is true."""
    if path.exists():
        kept = [r for r in read_jsonl(path) if keep(r)]
        atomic_write(path, "".join(json.dumps(r) + "\n" for r in kept).encode())
