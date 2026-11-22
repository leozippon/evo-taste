"""Worker CLI: prepare a run's shared artefacts, or advance a list of member slots on one GPU (resumable).

  python -m evotaste.worker prepare configs/run/<name>.yaml     # once per run, before any worker
  python -m evotaste.worker run runs/<name> --members m00,m01   # one process per GPU; rerun to resume

Run from the repository root with exactly one GPU visible (CUDA_VISIBLE_DEVICES). `--attach URL` reuses an
already running vLLM server of the same model instead of starting one (development only).
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from evotaste import config
from evotaste.member import Member, Run, current_lineage, response_records, summarize_probe
from evotaste.records import now, read_json, write_json, write_jsonl_gz
from evotaste.sampler import Request, Sampler, Server
from evotaste.tasks import check_gold, score_many
from evotaste.trainer import Policy

log = logging.getLogger("evotaste.worker")


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
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # run finally-blocks: stop vLLM, clean /dev/shm
    return path


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

def freeze(config_path: Path) -> Path:
    """Copy the config, suites and scaffolds into runs/<name>/ once; later calls must find them unchanged."""
    config.load(config_path)  # validate before writing anything
    data = yaml.safe_load(config_path.read_text())
    run_dir = Path("runs") / data["name"]
    sources = {"suites.yaml": Path(data.get("suites", config.RunConfig.suites)),
               "scaffolds.yaml": Path(data.get("scaffolds", config.RunConfig.scaffolds))}
    data.update(suites=str(run_dir / "suites.yaml"), scaffolds=str(run_dir / "scaffolds.yaml"))
    text = yaml.safe_dump(data, sort_keys=False)
    if (run_dir / "config.yaml").exists():
        same = (run_dir / "config.yaml").read_text() == text and all(
            (run_dir / name).read_bytes() == src.read_bytes() for name, src in sources.items())
        if not same:
            raise SystemExit(f"{run_dir} exists with a different config, suites or scaffolds; use a new run name")
        return run_dir
    run_dir.mkdir(parents=True)
    for name, src in sources.items():
        shutil.copyfile(src, run_dir / name)
    (run_dir / "config.yaml").write_text(text)
    return run_dir


def teachers(cfg) -> list[str]:
    return sorted({"plain"} | {m.scaffold for m in cfg.members if m.scaffold})


def build_pool(run: Run, sampler: Sampler, teacher: str, tasks) -> None:
    """The base model answers the seeding pool once under the teacher prompt; correct, stopped samples are kept."""
    s = run.cfg.seeding
    t0 = time.perf_counter()
    samples = sampler.generate([Request(run.cfg.model, run.prompt(t.question, teacher), s.samples, s.max_tokens)
                                for t in tasks])
    records, families = [], {}
    rewards = iter(s.reward for s in score_many([(t, x.text, x.finish_reason) for t, g in zip(tasks, samples) for x in g]))
    for task, group in zip(tasks, samples):
        fam = families.setdefault(task.family, {"prompts": 0, "samples": 0, "correct": 0, "prompts_with_correct": 0})
        correct = [x for x in group if next(rewards) == 1.0]
        fam["prompts"] += 1
        fam["samples"] += len(group)
        fam["correct"] += len(correct)
        fam["prompts_with_correct"] += bool(correct)
        records += [{"uid": task.uid, "family": task.family, "prompt_ids": run.prompt(task.question),
                     "completion_ids": x.token_ids, "text": x.text, "teacher_logprob": x.logprob} for x in correct]
    summary = {"teacher": teacher, "time": now(), "generate_s": round(time.perf_counter() - t0, 1),
               "prompts": len(tasks), "samples": sum(f["samples"] for f in families.values()),
               "correct": len(records), "prompts_with_correct": len({r["uid"] for r in records}),
               "by_family": families}
    write_jsonl_gz(run.dir / "seeding" / f"{teacher}.jsonl.gz", records)
    write_json(run.dir / "seeding" / f"{teacher}.json", summary)
    log.info("seeding pool %s: %d correct samples on %d of %d prompts (%.0f s)", teacher, summary["correct"],
             summary["prompts_with_correct"], len(tasks), summary["generate_s"])


def base_probe(run: Run, sampler: Sampler) -> None:
    samples = sampler.generate([Request(run.cfg.model, run.prompt(t.question), 1, run.cfg.probes.max_tokens)
                                for _, t in run.probe_tasks])
    records = [{"step": None, "suite": suite, **r} for (suite, _), r in
               zip(run.probe_tasks, response_records([(t, g[0]) for (_, t), g in zip(run.probe_tasks, samples)]))]
    write_jsonl_gz(run.dir / "base" / "probe_responses.jsonl.gz", records)
    summary = {"time": now(), **summarize_probe(records)}
    write_json(run.dir / "base" / "probe_summary.json", summary)
    log.info("base probe: %s", {k: round(v, 3) for k, v in summary["accuracy"].items()})


def prepare(config_path: Path, attach: str | None) -> None:
    run_dir = freeze(config_path)
    run = Run(run_dir)
    setup_logging(run.cfg.name, "prepare")
    pool_tasks = run.streams.seed_pool(run.cfg.seeding.pool_prompts)
    check_gold([t for _, t in run.probe_tasks] + pool_tasks)
    todo = [t for t in teachers(run.cfg) if not (run.dir / "seeding" / f"{t}.jsonl.gz").exists()]
    need_base = not (run.dir / "base" / "probe_summary.json").exists()
    if todo or need_base:
        server, url = start_server(run, attach, "prepare")
        sampler = None
        try:
            if server:
                server.wait_ready()
                url = server.url
            sampler = Sampler(url, run.cfg.model, run.cfg.name)
            for teacher in todo:
                build_pool(run, sampler, teacher, pool_tasks)
            if need_base:
                base_probe(run, sampler)
        finally:
            if sampler:
                sampler.close()
            if server:
                server.stop()
    run.check_probe_set()
    write_json(run.dir / "prepared.json", {"time": now(), "revision": revision(), "teachers": teachers(run.cfg)})
    log.info("run %s prepared", run.dir)


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


def advance(run: Run, slots: list[str]) -> None:
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

    while live := living():
        fresh = [m for m in live if m.lin.status == "new"]
        for m in fresh:
            m.seed()
        for m in live:
            if m.version is None:
                m.publish()
        if fresh:
            probe(run, fresh)
        stepping = [m for m in live if m.lin.status == "active" and m.lin.step < m.amp.steps]
        if not fresh and not stepping:
            raise RuntimeError(f"no member can make progress: {live}")
        if stepping:
            tasks = {m.lin.id: m.step_tasks() for m in stepping}
            per_member, seconds = generate_for(run, stepping, lambda m: m.rollout_requests(tasks[m.lin.id]))
            for m, samples in zip(stepping, per_member):
                m.amplify(tasks[m.lin.id], samples, seconds, len(stepping))
            log.info("step: %s (rollouts %.0f s)", ", ".join(f"{m.lin.id}@{m.lin.step}" for m in stepping), seconds)
            due = [m for m in stepping if m.probe_due()]
            if due:
                probe(run, due)
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


def run_worker(run_dir: Path, slots: list[str], attach: str | None) -> None:
    run = Run(run_dir)
    log_path = setup_logging(run.cfg.name, "worker-" + "-".join(slots))
    prepared = run.dir / "prepared.json"
    if not prepared.exists() or set(teachers(run.cfg)) - set(read_json(prepared)["teachers"]):
        raise SystemExit(f"{run.dir} is not prepared; run `python -m evotaste.worker prepare` first")
    run.check_probe_set()
    locks = lock_slots(run, slots)
    log.info("worker pid %d, GPU %s, members %s, revision %s, log %s", os.getpid(),
             os.environ.get("CUDA_VISIBLE_DEVICES"), slots, revision(), log_path)
    server, url = start_server(run, attach, "worker-" + "-".join(slots))
    try:
        run.policy = Policy(run.cfg.model, run.cfg.lora, run.cfg.trainer)  # loads while vLLM starts
        if server:
            server.wait_ready()
            url = server.url
        run.sampler = Sampler(url, run.cfg.model, run.cfg.name)
        advance(run, slots)
        log.info("all members finished")
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
    p = sub.add_parser("prepare", help="freeze the config and create seeding pools and base-model probe answers")
    p.add_argument("config", type=Path)
    p.add_argument("--attach", help="URL of a running vLLM server of the run's model (development)")
    r = sub.add_parser("run", help="advance member slots on the visible GPU; resumes from checkpoints")
    r.add_argument("run_dir", type=Path)
    r.add_argument("--members", required=True, help="comma-separated member ids of this worker")
    r.add_argument("--attach", help="URL of a running vLLM server of the run's model (development)")
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        prepare(args.config, args.attach)
    else:
        run_worker(args.run_dir, [s for s in args.members.split(",") if s], args.attach)


if __name__ == "__main__":
    main()
