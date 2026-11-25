"""Worker CLI: prepare a run's shared artefacts, or advance a list of member slots on one GPU (resumable).

  python -m evotaste.worker prepare configs/run/<name>.yaml     # once per run, before any worker
  python -m evotaste.worker run runs/<name> --members m00,m01   # one process per GPU; rerun to resume
  python -m evotaste.worker fork runs/<source> configs/run/<name>.yaml [--dry-run]   # reuse the source's seeded members

Run from the repository root with exactly one GPU visible (CUDA_VISIBLE_DEVICES). `--attach URL` reuses an
already running vLLM server of the same model instead of starting one (development only).
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

from evotaste import config
from evotaste.member import (Member, Run, current_lineage, first_lineage, lineage_dir, probe_requests,
                             response_records, summarize_probe)
from evotaste.records import atomic_write, now, read_json, read_jsonl, write_json, write_jsonl_gz
from evotaste.sampler import Sampler, Server
from evotaste.tasks import check_gold
from evotaste.trainer import Policy, gpu_processes

log = logging.getLogger("evotaste.worker")


_STOP_SIGNAL: str | None = None


def _stop(signum, _frame) -> None:
    global _STOP_SIGNAL
    _STOP_SIGNAL = signal.Signals(signum).name
    log.warning("received %s; stopping", _STOP_SIGNAL)
    sys.exit(128 + signum)  # run finally-blocks: stop vLLM, clean /dev/shm


def setup_logging(run_name: str, label: str) -> Path:
    path = Path("logs") / run_name / f"{label}-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers = [logging.FileHandler(path), logging.StreamHandler()]
    for h in handlers:
        h.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    for noisy in ("httpx", "httpcore", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, _stop)
    return path


def logged(command, *args) -> None:
    """Run a CLI command so that whatever ends it lands in its log file: the traceback of an exception, or the
    signal that stopped it (a deliberate stop is then distinguishable from a crash)."""
    try:
        command(*args)
    except SystemExit as e:
        if _STOP_SIGNAL:
            log.warning("stopped by %s", _STOP_SIGNAL)
        elif e.code not in (None, 0):
            log.error("exiting: %s", e.code, exc_info=True)
        raise
    except BaseException:
        log.critical("terminated by an exception", exc_info=True)
        raise


def server_options(parser) -> None:
    """Launch-time overrides of the run's server settings; they change memory footprint and speed, never results."""
    parser.add_argument("--kv-cache", help="vLLM KV budget for this process, e.g. 12G (default: server.kv_cache)")
    parser.add_argument("--max-loras", type=int, help="adapters per vLLM batch (default: server.max_loras)")
    parser.add_argument("--gpu-memory-utilization", type=float,
                        help="vLLM's startup free-memory check (default: server.gpu_memory_utilization)")


def server_overrides(args) -> dict:
    overrides = {k: v for k, v in (("kv_cache", args.kv_cache), ("max_loras", args.max_loras),
                                   ("gpu_memory_utilization", args.gpu_memory_utilization)) if v is not None}
    if overrides.get("max_loras", 1) < 1 or not 0 < overrides.get("gpu_memory_utilization", 0.5) <= 1:
        raise SystemExit("--max-loras must be >= 1 and --gpu-memory-utilization in (0, 1]")
    return overrides


def apply_server(run: Run, overrides: dict | None) -> None:
    """Use launch-time server settings for this process only; the frozen run config is not changed."""
    run.cfg = dataclasses.replace(run.cfg, server=dataclasses.replace(run.cfg.server, **(overrides or {})))
    log.info("server settings (footprint and speed only, never results): %s%s", dataclasses.asdict(run.cfg.server),
             f"; set at launch: {sorted(overrides)}" if overrides else "")


def log_uncommitted_changes(run_name: str) -> None:
    """If the working tree differs from HEAD, log the modified files and the SHA-256 of `git diff HEAD`, and keep
    the diff itself next to the logs, so that a run from uncommitted code can be reconstructed."""
    files = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.splitlines()
    if not files:
        log.info("working tree clean at launch")
        return
    diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True).stdout
    digest = hashlib.sha256(diff).hexdigest()
    path = Path("logs") / run_name / f"uncommitted-{digest[:12]}.patch"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(diff)
    log.warning("working tree dirty at launch: %s; git diff HEAD sha256 %s (saved to %s)",
                [f.strip() for f in files], digest, path)


def revision() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    return head + ("-dirty" if dirty else "")


def start_server(run: Run, attach: str | None, label: str) -> tuple[Server | None, str]:
    if attach:
        return None, attach
    server = Server(run.cfg.model, run.cfg.server, run.cfg.lora.rank,
                    Path("logs") / run.cfg.name / f"vllm-{label}-{os.getpid()}.log")
    server.start()
    return server, ""


# -- prepare -------------------------------------------------------------------------------------------------------

FROZEN = ("suites", "contexts")  # files copied next to the frozen config


def freeze(config_path: Path) -> Path:
    """Copy the config, suites, contexts and the strata table into runs/<name>/ once; later calls must find them
    unchanged. The strata table must have its pinned SHA-256."""
    cfg = config.load(config_path)  # validate before writing anything
    data = yaml.safe_load(config_path.read_text())
    run_dir = Path("runs") / data["name"]
    sources = {f"{key}.yaml": Path(getattr(cfg, key)) for key in FROZEN}
    data.update({key: str(run_dir / f"{key}.yaml") for key in FROZEN})
    if cfg.strata is not None:
        if hashlib.sha256(Path(cfg.strata.path).read_bytes()).hexdigest() != cfg.strata.sha256:
            raise SystemExit(f"{cfg.strata.path} does not have the pinned SHA-256 {cfg.strata.sha256}")
        sources["strata.tsv"] = Path(cfg.strata.path)
        data["strata"] = {**data["strata"], "path": str(run_dir / "strata.tsv")}
    text = yaml.safe_dump(data, sort_keys=False)
    if (run_dir / "config.yaml").exists():
        same = (run_dir / "config.yaml").read_text() == text and all(
            (run_dir / name).read_bytes() == src.read_bytes() for name, src in sources.items())
        if not same:
            raise SystemExit(f"{run_dir} exists with a different config, suites, contexts or strata; use a new run "
                             "name")
        return run_dir
    run_dir.mkdir(parents=True)
    for name, src in sources.items():
        shutil.copyfile(src, run_dir / name)
    (run_dir / "config.yaml").write_text(text)
    return run_dir


def base_probe(run: Run, sampler: Sampler, context: str | None = None) -> None:
    """The base model's answers to the probe set: plain (every member's reference), or behind the prompt that carries
    a context (the instructed reference of the context score). The summary is written last and marks completion."""
    prompt = run.prompt if context is None else (lambda q: run.prompt(q, context))
    seed_parts = ("base",) if context is None else ("base", context)
    samples = sampler.generate(probe_requests(run.cfg.model, run.probe_tasks, prompt, run.cfg.probes.max_tokens,
                                              *seed_parts))
    records = [{"step": None, "context": context, **r}
               for r in response_records([(t, g[0]) for t, g in zip(run.probe_tasks, samples)], token_ids=True)]
    summary = {"time": now(), "context": context, **summarize_probe(records)}
    responses, summary_path = base_files(run.dir, context)
    write_jsonl_gz(responses, records)
    write_json(summary_path, summary)
    log.info("base probe%s: accuracy %.3f %s", f" under context {context}" if context else "", summary["accuracy"],
             {k: round(v, 3) for k, v in summary["by_domain"].items()})


def base_files(run_dir: Path, context: str | None) -> tuple[Path, Path]:
    """(responses, summary) of the base model's plain probe answers or of its answers under a context."""
    if context is None:
        return run_dir / "base" / "probe_responses.jsonl.gz", run_dir / "base" / "probe_summary.json"
    return run_dir / "base" / "contexts" / f"{context}.jsonl.gz", run_dir / "base" / "contexts" / f"{context}.json"


def prepare(config_path: Path, attach: str | None, server: dict | None = None) -> None:
    run_dir = freeze(config_path)
    run = Run(run_dir)
    apply_server(run, server)
    check_gold(run.probe_tasks)
    todo = [c for c in [None, *run.contexts] if not base_files(run.dir, c)[1].exists()]
    if todo:
        server, url = start_server(run, attach, "prepare")
        sampler = None
        try:
            if server:
                server.wait_ready()
                url = server.url
            sampler = Sampler(url, run.cfg.model, run.cfg.name)
            for context in todo:
                base_probe(run, sampler, context)
        finally:
            if sampler:
                sampler.close()
            if server:
                server.stop()
    run.check_probe_set()
    write_json(run.dir / "prepared.json", {"time": now(), "revision": revision(), "contexts": sorted(run.contexts)})
    log.info("run %s prepared", run.dir)


# -- fork ----------------------------------------------------------------------------------------------------------

FORK_IDENTICAL = ("model", "seed", "lora", "seeding", "probes", "trainer")  # plus suites and contexts content


def _fork_member(src: Path, src_cfg, new_cfg, spec) -> str | None:
    """None if the source lineage's seeding can be reused for `spec`, else the reason why not."""
    if spec.parent is not None:
        return "a clone starts from its parent in this run"
    try:
        old = src_cfg.member(spec.id)
    except KeyError:
        return "not in the source run"
    if old.parent is not None or (old.context, old.seed) != (spec.context, spec.seed):
        return "context or seed differ, or a clone in the source"
    seeding = new_cfg.settings(spec)[0]
    if src_cfg.settings(old)[0] != seeding:
        return "effective seeding settings differ"
    d = src / "members" / spec.id
    state = read_json(d / "state.json") if (d / "state.json").exists() else None
    if state is None or state["status"] == "new" or not state["probes"] or state["probes"][0]["step"] != 0:
        return "no completed step-0 checkpoint in the source (still seeding?)"
    ckpt = d / "checkpoints" / "step_000000"
    files = [ckpt / "adapter_config.json", ckpt / "adapter_model.safetensors",
             d / "probe_responses" / "step_000000.jsonl.gz"]
    if not all(f.exists() for f in files):
        return "step-0 checkpoint or probe answers missing in the source"
    steps = 0 if seeding.method == "none" else seeding.steps
    if sum(r["phase"] == "seed" for r in read_jsonl(d / "history.jsonl")) != steps:
        return "incomplete seeding history in the source"
    return None


def plan_fork(src: Path, config_path: Path) -> dict:
    """What a fork of `src` under `config_path` reuses. Refuses unless everything that determines the base model's
    reference answers and the step-0 adapters is identical; amplification, culling, server and member overrides of
    those may differ."""
    src_cfg, new_cfg = config.load(src / "config.yaml"), config.load(config_path)
    differ = [f for f in FORK_IDENTICAL if getattr(src_cfg, f) != getattr(new_cfg, f)]
    differ += [f"{f} content" for f in FROZEN
               if Path(getattr(src_cfg, f)).read_bytes() != Path(getattr(new_cfg, f)).read_bytes()]
    if differ:
        raise SystemExit(f"cannot fork {src}: {', '.join(differ)} differ from the source run")
    if new_cfg.name == src_cfg.name or (Path("runs") / new_cfg.name).exists():
        raise SystemExit(f"runs/{new_cfg.name} is the source or exists already; a fork needs a new run name")
    if not (src / "prepared.json").exists():
        raise SystemExit(f"{src} is not prepared")
    return {"source": src, "source_revision": read_json(src / "prepared.json")["revision"], "name": new_cfg.name,
            "members": {m.id: _fork_member(src, src_cfg, new_cfg, m) for m in new_cfg.members}}


def fork(src: Path, config_path: Path, dry_run: bool) -> None:
    """Create a prepared run that shares the source's base reference answers and starts every reusable member at
    amplification step 0 from the source lineage's step-0 checkpoint. Only copies; the source may be live, and only
    committed step-0 artefacts are read."""
    plan = plan_fork(src, config_path)
    reuse = [m for m, why in plan["members"].items() if why is None]
    base = sorted(p for p in (src / "base").rglob("*") if p.is_file())
    print(f"fork {src} (prepared at {plan['source_revision']}) -> runs/{plan['name']}")
    print(f"identical: {', '.join(FORK_IDENTICAL)}, suites and contexts content")
    print(f"copy {len(base)} base reference files ({sum(f.stat().st_size for f in base) / 2**20:.0f} MiB)")
    ckpt_mib = sum(f.stat().st_size for m in reuse
                   for f in (src / "members" / m / "checkpoints" / "step_000000").iterdir()) / 2**20
    print(f"start at amplification step 0 from the source's step-0 checkpoint ({len(reuse)}, {ckpt_mib:.0f} MiB): "
          f"{', '.join(reuse)}")
    for m, why in plan["members"].items():
        if why is not None:
            print(f"start normally: {m} ({why})")
    if dry_run:
        return
    run_dir = freeze(config_path)
    for f in base:
        target = run_dir / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
    run = Run(run_dir)
    run.check_probe_set()
    for slot in reuse:
        s, d = src / "members" / slot, lineage_dir(run, slot)
        probe0 = read_json(s / "state.json")["probes"][0]
        seed_history = [r for r in read_jsonl(s / "history.jsonl") if r["phase"] == "seed"]
        shutil.copytree(s / "checkpoints" / "step_000000", d / "checkpoints" / "step_000000")
        (d / "probe_responses").mkdir()
        shutil.copy2(s / "probe_responses" / "step_000000.jsonl.gz", d / "probe_responses" / "step_000000.jsonl.gz")
        atomic_write(d / "history.jsonl", "".join(json.dumps(r) + "\n" for r in seed_history).encode())
        atomic_write(d / "probes.jsonl", (json.dumps(probe0) + "\n").encode())
        lin = first_lineage(run.cfg, slot)
        lin.status = "active" if run.cfg.settings(run.cfg.member(slot))[1].steps > 0 else "done"
        lin.probes = [probe0]
        write_json(d / "state.json", dataclasses.asdict(lin))
    write_json(run.dir / "prepared.json", {"time": now(), "revision": revision(), "contexts": sorted(run.contexts),
                                           "fork": {"source": str(src), "source_revision": plan["source_revision"],
                                                    "members": reuse}})
    log.info("run %s forked from %s; %d members start at amplification step 0", run.dir, src, len(reuse))


# -- run -----------------------------------------------------------------------------------------------------------

def generate_for(run: Run, members: list[Member], requests_of) -> tuple[list, float]:
    """One concurrent generation for several members; returns per-member sample lists and the wall time."""
    t0 = time.perf_counter()
    requests, spans = [], []
    for m in members:
        reqs = requests_of(m)
        spans.append((len(requests), len(requests) + len(reqs)))
        requests += reqs
    samples = run.sampler.generate(requests)
    return [samples[a:b] for a, b in spans], time.perf_counter() - t0


def probe(run: Run, members: list[Member]) -> None:
    per_member, seconds = generate_for(run, members, Member.probe_requests)
    log.info("probe generation for %d member(s): %.0f s", len(members), seconds)
    for m, samples in zip(members, per_member):
        m.finish_probe(samples)


def seed_members(run: Run, members: list[Member]) -> None:
    """Seed several members in lockstep: round i makes update i of every member that has one, with one joint
    generation for all of them."""
    for m in members:
        m.begin_seeding()
    for i in range(max(m.seeding_steps for m in members)):
        active = [m for m in members if i < m.seeding_steps]
        for m in active:
            if m.seeding.method == "opcd" and m.version is None:
                m.publish()
        tasks = {m.lin.id: m.seed_tasks(i) for m in active}
        per_member, seconds = generate_for(run, active, lambda m: m.seed_requests(i, tasks[m.lin.id]))
        for m, samples in zip(active, per_member):
            m.seed_update(i, tasks[m.lin.id], samples, seconds, len(active))
        log.info("seeding update %d: %s (generation %.0f s)", i, ", ".join(m.lin.id for m in active), seconds)
    for m in members:
        m.end_seeding()


POLL_S = 60  # a worker whose only remaining members are clones waiting for a parent in another worker polls this often


def advance(run: Run, slots: list[str], checkpoint_every: int | None = None) -> None:
    """Advance the slots' members to the end of their courses: seed new members, start clones once their parent has
    committed the clone point, then step. Besides the probe-step checkpoints, a light checkpoint (no probe) is written
    every `checkpoint_every` steps and at every clone point."""
    checkpoint_every = checkpoint_every or run.cfg.probes.every
    members: dict[str, Member] = {}

    def living() -> list[Member]:
        out = []
        for slot in slots:
            lin = current_lineage(run, slot)
            if lin is None or lin.status == "done":
                continue
            if lin.id not in members:
                members[lin.id] = Member(run, lin)
                members[lin.id].restore()
            out.append(members[lin.id])
        return out

    waiting_logged = False
    while live := living():
        seeding = [m for m in live if m.lin.status == "new" and not m.is_clone]
        clones = [m for m in live if m.lin.status == "new" and m.is_clone]
        starting = []
        for m in clones:
            if m.weights is None and (ckpt := m.parent_checkpoint()) is not None:
                m.start_clone(ckpt)
            if m.weights is not None:
                starting.append(m)
        if seeding:
            seed_members(run, seeding)
        fresh = seeding + starting
        for m in fresh:
            if m.version is None:
                m.publish()
        if fresh:
            probe(run, fresh)
        stepping = [m for m in live if m.lin.status == "active" and m.lin.step < m.amp.steps]
        if not fresh and not stepping:
            if len(starting) < len(clones):  # clones whose parents another worker advances
                if not waiting_logged:
                    log.info("waiting for parents: %s", ", ".join(f"{m.lin.id} <- {m.lin.parent['lineage']}@"
                                                                  f"{m.lin.parent['step']}" for m in clones))
                    waiting_logged = True
                time.sleep(POLL_S)
                continue
            raise RuntimeError(f"no member can make progress: {live}")
        waiting_logged = False
        for m in stepping:
            if m.version is None:
                m.publish()
        if stepping:
            tasks = {m.lin.id: m.step_tasks() for m in stepping}
            per_member, seconds = generate_for(run, stepping, lambda m: m.rollout_requests(tasks[m.lin.id]))
            for m, samples in zip(stepping, per_member):
                m.amplify(tasks[m.lin.id], samples, seconds, len(stepping))
            log.info("step: %s (rollouts %.0f s)", ", ".join(f"{m.lin.id}@{m.lin.step}" for m in stepping), seconds)
            due = [m for m in stepping if m.probe_due()]
            if due:
                probe(run, due)
            for m in stepping:
                if m not in due and m.checkpoint_due(checkpoint_every):
                    m.light_checkpoint()
        for m in live:
            if m.finished:
                m.retire()


def lock_slots(run: Run, slots: list[str]) -> list:
    """Exclusive per-slot locks for the life of this process: two workers can never advance the same member."""
    handles = []
    for slot in slots:
        run.cfg.member(slot)  # unknown ids fail here
        path = run.dir / "locks" / f"{slot}.lock"
        path.parent.mkdir(exist_ok=True)
        f = open(path, "w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"member {slot} is being advanced by another worker") from None
        handles.append(f)
    return handles


# CPU threads of a worker's torch: its CPU work is small (dtype conversion of an adapter, copies), and with torch's
# default of one thread per core several workers' OpenMP regions stall one another for tens of seconds.
WORKER_CPU_THREADS = 4


def run_worker(run_dir: Path, slots: list[str], attach: str | None, checkpoint_every: int | None = None,
               server: dict | None = None) -> None:
    torch.set_num_threads(WORKER_CPU_THREADS)
    run = Run(run_dir)
    apply_server(run, server)
    if not (run.dir / "prepared.json").exists():
        raise SystemExit(f"{run.dir} is not prepared; run `python -m evotaste.worker prepare` first")
    run.check_probe_set()
    locks = lock_slots(run, slots)
    log.info("worker pid %d, GPU %s, members %s, revision %s", os.getpid(),
             os.environ.get("CUDA_VISIBLE_DEVICES"), slots, revision())
    log_uncommitted_changes(run.cfg.name)
    server, url = start_server(run, attach, "worker-" + "-".join(slots))
    try:
        run.policy = Policy(run.cfg.model, run.cfg.lora, run.cfg.trainer)  # loads while vLLM starts
        if server:
            server.wait_ready()
            url = server.url
        run.sampler = Sampler(url, run.cfg.model, run.cfg.name)
        advance(run, slots, checkpoint_every)
        log.info("all members finished")
    except torch.OutOfMemoryError:  # before vLLM stops: co-tenants are the usual cause on this machine
        log.error("CUDA out of memory in pid %d; GPU memory now (find this pid's GPU):\n%s", os.getpid(),
                  gpu_processes())
        raise
    finally:
        if run.sampler:
            run.sampler.close()
        if server:
            server.stop()
        for f in locks:
            f.close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="freeze the config and record the base model's probe answers, plain and "
                                       "under every context")
    p.add_argument("config", type=Path)
    p.add_argument("--attach", help="URL of a running vLLM server of the run's model (development)")
    server_options(p)
    f = sub.add_parser("fork", help="a prepared run that reuses another run's seeding and starts at step 0")
    f.add_argument("source", type=Path)
    f.add_argument("config", type=Path)
    f.add_argument("--dry-run", action="store_true", help="print what would be copied and reused, create nothing")
    r = sub.add_parser("run", help="advance member slots on the visible GPU; resumes from checkpoints")
    r.add_argument("run_dir", type=Path)
    r.add_argument("--members", required=True, help="comma-separated member ids of this worker")
    r.add_argument("--attach", help="URL of a running vLLM server of the run's model (development)")
    r.add_argument("--checkpoint-every", type=int, help="also write a light checkpoint (no probe) every N steps "
                                                         "(default: probes.every)")
    server_options(r)
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        setup_logging(yaml.safe_load(args.config.read_text())["name"], "prepare")
        logged(prepare, args.config, args.attach, server_overrides(args))
    elif args.cmd == "fork" and args.dry_run:
        fork(args.source, args.config, True)
    elif args.cmd == "fork":
        setup_logging(yaml.safe_load(args.config.read_text())["name"], "fork")
        logged(fork, args.source, args.config, False)
    else:
        slots = [s for s in args.members.split(",") if s]
        setup_logging(args.run_dir.name, "worker-" + "-".join(slots))
        if args.checkpoint_every is not None and args.checkpoint_every < 1:
            raise SystemExit("--checkpoint-every must be at least 1")
        logged(run_worker, args.run_dir, slots, args.attach, args.checkpoint_every, server_overrides(args))


if __name__ == "__main__":
    main()
