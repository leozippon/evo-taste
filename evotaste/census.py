"""Census of the base model on the Round 2 environment (docs/design/round-2.md, pilot A); no training.

  python -m evotaste.census run configs/census.yaml --part train [--shard i/n] [--limit N] [server overrides]
  python -m evotaste.census summarize configs/census.yaml [--limit N]

Parts, each a fixed list of jobs (task, samples, context) cut into chunks of `chunk_items` jobs:
  train      a seeded draw from every training source's train part (capped by domain), `train.samples` per item
  selection  the first items of every selection unit (probe, then evaluation items) and of the procedural families'
             evaluation stream, `selection.samples` per item
  contexts   the probe set's plain items, `contexts.samples` under each context of the contexts file

Generation is at temperature 1 with `max_tokens`, each request seeded by (part, uid, variant). A chunk's records are
written atomically once all of its jobs are scored, so a restart skips finished chunks; shard i of n takes the chunks
whose index is i mod n, so shards run on several GPUs at once and may be re-sharded between runs. Each process uses
the one GPU in CUDA_VISIBLE_DEVICES and refuses to start unless it has `need_free_mib` free.

The census reads items only through the open accessors (`Suites.items` of train and selection, the probe set) and
never holds a blind receipt, so it cannot reach the vault; every job's uid is also checked against the open parts.
`summarize` writes the strata table of the training items (uid, passes, samples; pin its SHA-256 in run configs) and
summary.json, under results/census/<name>/ (outside git).
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import logging
import os
import resource
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from evotaste.records import now, read_jsonl_gz, write_jsonl_gz
from evotaste.tasks import Suites, Task, probe_set, procedural_tasks, prompt_ids, score_many, stable_seed

log = logging.getLogger("evotaste.census")
PARTS = ("train", "selection", "contexts")
MAX_DEATHS = 5  # a chunk with more samples whose verifier or scorer process died (or 1%) is refused


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    required = {"name", "model", "suites", "contexts", "max_tokens", "chunk_items", "need_free_mib", "server", "parts"}
    missing = required - set(cfg)
    if missing or set(cfg) - required or set(cfg["parts"]) != set(PARTS):
        raise ValueError(f"{path}: missing keys {sorted(missing)}, unknown keys {sorted(set(cfg) - required)}, or "
                         f"parts other than {PARTS}")
    return cfg


def out_dir(cfg: dict, limit: int | None) -> Path:
    return Path("results/census") / (cfg["name"] + (f"-limit{limit}" if limit else ""))


# ---------------------------------------------------------------------------------------------------------------
# Jobs


class CensusJob:
    __slots__ = ("task", "samples", "context", "variant")

    def __init__(self, task: Task, samples: int, context: str | None, variant: str):
        self.task, self.samples, self.context, self.variant = task, samples, context, variant


def _open_uids(suites: Suites) -> set[str]:
    return {t.uid for p in ("train", "selection") for s in (suites.train if p == "train" else suites.selection)
            for t in suites.items(s, p)}


def train_items(cfg: dict, suites: Suites, limit: int | None) -> list[Task]:
    """The first `cap[domain]` items of each training source's train part, whose order is a seeded draw (at most
    `limit`)."""
    from evotaste.sources import SOURCES

    caps, out = cfg["parts"]["train"]["cap"], []
    for s in suites.train:
        cap = caps[SOURCES[s].domain]
        out += suites.items(s, "train")[:min(cap, limit or cap)]
    return out


def selection_items(cfg: dict, suites: Suites, limit: int | None) -> list[Task]:
    """The first `cap` items of each selection source's part, the probe items and then the evaluation items
    (tasks.fixed_tasks draws from these first), and the procedural families' first evaluation instances."""
    p = cfg["parts"]["selection"]
    cap, per_family = min(p["cap"], limit or p["cap"]), min(p["per_family"], limit or p["per_family"])
    tasks = [t for s in suites.selection for t in suites.items(s, "selection")[:cap]]
    return tasks + (procedural_tasks(suites.procedural, "eval", per_family) if suites.procedural else [])


def strata_table(records: list[dict], train_jobs: list[CensusJob]) -> dict[str, tuple[int, int]]:
    """uid -> (passes, samples) of the train census, refused unless the records are exactly those of the jobs."""
    passes: dict[str, list[int]] = {}
    for r in records:
        passes.setdefault(r["uid"], []).append(int(r["reward"] == 1.0))
    if set(passes) != {j.task.uid for j in train_jobs} or any(
            len(passes[j.task.uid]) != j.samples for j in train_jobs):
        raise SystemExit("the train census records are not those of the current train part (the suites file "
                         "changed since?); rerun the train part into a new output directory")
    return {uid: (sum(v), len(v)) for uid, v in passes.items()}


def jobs_of(part: str, cfg: dict, suites: Suites, limit: int | None) -> list[CensusJob]:
    """The part's job list, a pure function of the config and the suites file."""
    p = cfg["parts"][part]
    if part == "train":
        return [CensusJob(t, p["samples"], None, "default") for t in train_items(cfg, suites, limit)]
    if part == "selection":
        return [CensusJob(t, p["samples"], None, "default") for t in selection_items(cfg, suites, limit)]
    if part == "contexts":
        from evotaste.config import load_contexts

        contexts = load_contexts(cfg["contexts"])
        probe = [t for t in probe_set(suites) if t.wrapper in ("box", "code")]  # the plain probe items
        probe = probe[:limit] if limit else probe
        return [CensusJob(t, p["samples"], name, name) for t in probe for name in sorted(contexts)]
    raise ValueError(f"unknown part {part!r}")


def chunks(jobs: list[CensusJob], size: int) -> list[list[CensusJob]]:
    return [jobs[i:i + size] for i in range(0, len(jobs), size)]


def chunk_path(base: Path, part: str, c: int) -> Path:
    return base / part / f"chunk-{c:05d}.jsonl.gz"


def read_part(base: Path, part: str, n_chunks: int) -> list[dict]:
    missing = [c for c in range(n_chunks) if not chunk_path(base, part, c).exists()]
    if missing:
        raise SystemExit(f"census part {part} incomplete: {len(missing)} of {n_chunks} chunks missing "
                         f"(e.g. {missing[:5]}); run it first")
    return [r for c in range(n_chunks) for r in read_jsonl_gz(chunk_path(base, part, c))]


# ---------------------------------------------------------------------------------------------------------------
# Running a part


def free_mib() -> int:
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu or "," in gpu:
        raise SystemExit("pin exactly one GPU with CUDA_VISIBLE_DEVICES")
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", gpu],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip())


def sample_records(part: str, job: CensusJob, samples, scores) -> list[dict]:
    """The records of one job's samples and their scores (the census record shape, also the method census's)."""
    t = job.task
    return [{"part": part, "uid": t.uid, "family": t.family, "domain": t.domain, "kind": t.kind, "wrapper": t.wrapper,
             "context": job.context, "variant": job.variant, "sample": k,
             "text": s.text, "completion_tokens": len(s.token_ids), "finish_reason": s.finish_reason,
             "reward": sc.reward, "correct_any": sc.correct_any, "complied": sc.complied, "channel": sc.channel,
             "truncated": sc.truncated, "extracted": sc.extracted, "error": sc.error}
            for k, (s, sc) in enumerate(zip(samples, scores))]


def run_chunks(path_of, todo: list[int], generate, score, timing: Path, meta: dict) -> None:
    """Generate and score the chunks `todo` in order: generate(c) returns chunk c's generation results (the GPU's
    work), score(c, results) its records (the CPU's). The next chunk generates while one is scored, two in flight. A
    chunk's records are written atomically to path_of(c) once all of them are scored, so a restart skips finished
    chunks; a chunk with more than MAX_DEATHS (or 1%) records whose verifier or scorer process died is refused. One
    timing line per chunk, with `meta`, is appended to `timing`."""

    def timed(c: int):
        t0 = time.time()
        return generate(c), t0, time.time()

    pool = concurrent.futures.ThreadPoolExecutor(2)
    try:
        pending = collections.deque((c, pool.submit(timed, c)) for c in todo[:2])
        queue = collections.deque(todo[2:])
        while pending:
            c, future = pending.popleft()
            results, start, end = future.result()
            if queue:
                nxt = queue.popleft()
                pending.append((nxt, pool.submit(timed, nxt)))
            t0 = time.perf_counter()
            records = score(c, results)
            died = sum("process died" in (r.get("error") or "") for r in records)
            if died > max(MAX_DEATHS, len(records) // 100):  # a broken verifier, not wrong answers
                raise RuntimeError(f"chunk {c}: {died} of {len(records)} samples lost their verifier or scorer "
                                   "process; not writing it")
            write_jsonl_gz(path_of(c), records)
            tokens = sum(r["completion_tokens"] for r in records)
            line = {"time": now(), **meta, "chunk": c, "pid": os.getpid(), "samples": len(records), "tokens": tokens,
                    "start": round(start, 1), "end": round(end, 1), "score_s": round(time.perf_counter() - t0, 1)}
            with open(timing, "a") as f:
                f.write(json.dumps(line) + "\n")
            log.info("chunk %d: %d records, %d tokens, generated in %.0f s (overlapping the next chunk), scored in "
                     "%.0f s", c, len(records), tokens, end - start, line["score_s"])
    finally:
        pool.shutdown(wait=False, cancel_futures=True)  # an error stops the run now, not after the chunk in flight


def run_part(config: Path, part: str, shard: str, limit: int | None, attach: str | None, server: dict) -> None:
    from transformers import AutoTokenizer

    from evotaste.sampler import Request, Sampler, Server

    cfg = load_config(config)
    base = out_dir(cfg, limit)
    suites = Suites(cfg["suites"])
    jobs = jobs_of(part, cfg, suites, limit)
    allowed = _open_uids(suites) | {t.uid for t in probe_set(suites)} | {
        j.task.uid for j in jobs if j.task.kind == "procedural" and j.task.family in suites.procedural}
    stray = sum(j.task.uid not in allowed for j in jobs)
    if stray:  # a count only: a stray uid could be a vault item
        raise RuntimeError(f"{stray} census jobs are outside the open parts")
    index, count = (int(x) for x in shard.split("/"))
    all_chunks = chunks(jobs, cfg["chunk_items"])
    todo = [c for c in range(index, len(all_chunks), count) if not chunk_path(base, part, c).exists()]
    log.info("census %s part %s shard %s: %d jobs in %d chunks, %d to do here", base, part, shard, len(jobs),
             len(all_chunks), len(todo))
    if not todo:
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)  # two chunks in flight hold up to 1,024 connections
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 65536), hard))
    contexts = {}
    if part == "contexts":
        from evotaste.config import load_contexts

        contexts = load_contexts(cfg["contexts"])
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    srv = None
    if not attach:
        if free_mib() < cfg["need_free_mib"]:
            raise SystemExit(f"GPU {os.environ['CUDA_VISIBLE_DEVICES']} has {free_mib()} MiB free, fewer than "
                             f"need_free_mib {cfg['need_free_mib']}")
        settings = SimpleNamespace(**{**cfg["server"], **server})
        srv = Server(cfg["model"], settings, 32, base / "logs" / f"vllm-{part}-{os.getpid()}.log")
        srv.start()
    sampler = None
    try:
        if srv:
            srv.wait_ready()
        sampler = Sampler(srv.url if srv else attach, cfg["model"], f"census-{cfg['name']}")

        def generate(c: int):
            return sampler.generate([Request(cfg["model"], prompt_ids(tok, j.task.question,
                                                                      context=contexts.get(j.context)),
                                             j.samples, cfg["max_tokens"],
                                             stable_seed("census", part, j.task.uid, j.variant))
                                     for j in all_chunks[c]])

        def score(c: int, results) -> list[dict]:
            flat = [(j.task, s.text, s.finish_reason) for j, group in zip(all_chunks[c], results) for s in group]
            scores = iter(score_many(flat))
            return [r for j, group in zip(all_chunks[c], results)
                    for r in sample_records(part, j, group, [next(scores) for _ in group])]

        run_chunks(lambda c: chunk_path(base, part, c), todo, generate, score, base / part / "timing.jsonl",
                   {"part": part, "shard": shard})
    finally:
        if sampler:
            sampler.close()
        if srv:
            srv.stop()


# ---------------------------------------------------------------------------------------------------------------
# Summary


def _lengths(rs: list[dict]) -> dict:
    x = np.array([r["completion_tokens"] for r in rs])
    return {f"p{q}": int(np.percentile(x, q)) for q in (10, 50, 90, 99)} | {"max": int(x.max()), "mean": round(float(x.mean()), 1)}


def _block(rs: list[dict]) -> dict:
    n = len(rs)
    return {"samples": n, "accuracy": round(sum(r["reward"] for r in rs) / n, 4),
           "correct_any": round(sum(r["correct_any"] for r in rs) / n, 4),
           "complied": round(sum(r["complied"] for r in rs) / n, 4),
           "truncated": round(sum(r["truncated"] for r in rs) / n, 4), "errors": sum(r["error"] is not None for r in rs),
           "tokens": _lengths(rs)}


def summarize(config: Path, limit: int | None) -> None:
    cfg = load_config(config)
    base = out_dir(cfg, limit)
    suites = Suites(cfg["suites"])
    summary: dict = {"time": now(), "config": cfg, "limit": limit, "parts": {}}
    for part in PARTS:
        try:
            jobs = jobs_of(part, cfg, suites, limit)
            records = read_part(base, part, len(chunks(jobs, cfg["chunk_items"])))
        except SystemExit as e:
            log.warning("%s", e)
            continue
        by = collections.defaultdict(list)
        key = {"train": "family", "selection": "family", "contexts": "context"}[part]
        for r in records:
            by[r[key]].append(r)
        block = {k: _block(v) for k, v in sorted(by.items())}
        if part == "train":
            table = strata_table(records, jobs)
            for source, rs in by.items():
                uids = {r["uid"] for r in rs}
                hist = collections.Counter(table[u][0] for u in uids)
                block[source]["passes_histogram"] = [hist.get(k, 0)
                                                     for k in range(cfg["parts"]["train"]["samples"] + 1)]
                strata = collections.Counter(_stratum(*table[u]) for u in uids)
                block[source]["strata"] = {s: strata.get(s, 0) for s in ("frontier", "sweet", "mastered")}
            path = base / "strata.tsv"
            text = "uid\tpasses\tsamples\n" + "".join(f"{u}\t{p}\t{n}\n" for u, (p, n) in sorted(table.items()))
            path.write_text(text)
            summary["strata"] = {"path": str(path), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                                 "items": len(table)}
        if part == "contexts":
            for name, rs in by.items():
                block[name]["by_domain"] = {d: _block([r for r in rs if r["domain"] == d])
                                            for d in sorted({r["domain"] for r in rs})}
        summary["parts"][part] = {"jobs": len(jobs), "samples": len(records), "groups": block,
                                  "overall": _block(records), "throughput": throughput(base / part / "timing.jsonl")}
    (base / "summary.json").write_text(json.dumps(summary, indent=1))
    log.info("wrote %s%s", base / "summary.json",
             f"; strata table {summary['strata']['path']} sha256 {summary['strata']['sha256']}" if "strata" in summary
             else "")
    if "strata" in summary:
        print(f"strata table: {summary['strata']['path']}\nsha256: {summary['strata']['sha256']}")


def throughput(path: Path) -> list[dict]:
    """Per process: completion tokens generated and their rate over the wall time from its first chunk's start to its
    last chunk's end (chunks overlap, so per-chunk rates would understate the server's)."""
    if not path.exists():
        return []
    by_pid = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        t = json.loads(line)
        by_pid[t["pid"]].append(t)
    out = []
    for pid, ts in sorted(by_pid.items()):
        span = max(t["end"] for t in ts) - min(t["start"] for t in ts)
        tokens = sum(t["tokens"] for t in ts)
        out.append({"pid": pid, "chunks": len(ts), "samples": sum(t["samples"] for t in ts), "tokens": tokens,
                    "wall_s": round(span), "tokens_per_s": round(tokens / span) if span else None})
    return out


def _stratum(passes: int, samples: int) -> str:
    from evotaste.tasks import stratum_of

    return stratum_of(passes, samples)


def main(argv=None) -> None:
    from evotaste.worker import logged, server_options, server_overrides, setup_logging

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("config", type=Path)
    r.add_argument("--part", required=True, choices=PARTS)
    r.add_argument("--shard", default="0/1", help="i/n: this process runs the chunks whose index is i mod n")
    r.add_argument("--limit", type=int, help="smoke run: at most N items per source (output under <name>-limitN)")
    r.add_argument("--attach", help="URL of a running vLLM server of the model (development)")
    server_options(r)
    s = sub.add_parser("summarize")
    s.add_argument("config", type=Path)
    s.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    name = load_config(args.config)["name"] + (f"-limit{args.limit}" if args.limit else "")
    if args.cmd == "run":
        setup_logging(f"census-{name}", f"census-{args.part}-{args.shard.replace('/', 'of')}")
        logged(run_part, args.config, args.part, args.shard, args.limit, args.attach, server_overrides(args))
    else:
        setup_logging(f"census-{name}", "census-summarize")
        logged(summarize, args.config, args.limit)


if __name__ == "__main__":
    main()
