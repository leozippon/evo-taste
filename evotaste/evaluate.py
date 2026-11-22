"""Final evaluation of adapters (or the base model) on a suite, the harvest manifest, and the blind guard.

  python -m evotaste.evaluate run runs/<name> --suite selection --adapters base,m00,m01 --per-family 50
  python -m evotaste.evaluate harvest runs/<name> --tag n50x1

Suites: train (fresh train-family instances), selection (selection families + foundation items), blind1, blind2.
Adapters: `base`, a lineage id (its last checkpoint) or `<lineage>@<step>`. `--shard i/n` splits the
(adapter, instance) jobs across GPUs; every shard writes its own record file and the summary covers all files of
the tag. A blind vault is opened only through `blind_guard`, which needs a committed, unmodified harvest manifest
listing every adapter with its content hash (base-model calibration is exempt) and appends to the blind ledger
before anything is generated.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
from pathlib import Path

from evotaste import external
from evotaste.member import Run, load_lineage, response_record
from evotaste.records import now, read_json, read_jsonl_gz, write_json, write_jsonl_gz
from evotaste.sampler import Request, Sampler
from evotaste.tasks import BlindReceipt, blind_suite, check_gold, fixed_tasks
from evotaste.trainer import adapter_sha256
from evotaste.worker import revision, setup_logging, start_server

log = logging.getLogger("evotaste.evaluate")
SUITES = ("train", "selection", "blind1", "blind2")
VAULTS = ("blind1", "blind2")
LEDGER = Path("docs/blind-ledger.md")
LEDGER_HEADER = ("# Blind ledger\n\nAppended by `evotaste.evaluate` before any generation on a blind vault. Commit it "
                 "together with the results.\n\n| time | vault | revision | manifest sha256 | adapters | per family "
                 "x samples | tasks |\n|---|---|---|---|---|---|---|\n")


def manifest_path(run_name: str, repo: Path = Path(".")) -> Path:
    return repo / "docs" / "harvest" / f"{run_name}.json"


def _git(repo: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def blind_guard(vault: str, run_name: str, adapters: dict[str, Path | None], per_family: int, samples: int,
                repo: Path = Path(".")) -> BlindReceipt:
    """Refuse unless every adapter is in a committed, unmodified harvest manifest; then append to the ledger.

    `adapters` maps labels to adapter directories (None = base model, allowed without a manifest).
    """
    if vault not in VAULTS:
        raise ValueError(f"{vault!r} is not a blind vault")
    trained = {label: path for label, path in adapters.items() if path is not None}
    manifest_hash = "none (base model only)"
    if trained:
        path = manifest_path(run_name, repo)
        rel = str(path.relative_to(repo))
        if not path.exists() or _git(repo, "ls-files", "--error-unmatch", rel).returncode != 0:
            raise PermissionError(f"blind evaluation refused: harvest manifest {rel} is not tracked in git")
        if _git(repo, "diff", "--quiet", "HEAD", "--", rel).returncode != 0:
            raise PermissionError(f"blind evaluation refused: {rel} differs from the committed version")
        manifest = read_json(path)
        listed = {e["sha256"] for e in manifest["ranking"]}
        for label, adapter in trained.items():
            if adapter_sha256(adapter) not in listed:
                raise PermissionError(f"blind evaluation refused: adapter {label} ({adapter}) is not in {rel} "
                                      "with a matching content hash")
        manifest_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    dirty = bool(_git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip())
    line = (f"| {now()} | {vault} | {head}{'-dirty' if dirty else ''} | {manifest_hash} | "
            f"{', '.join(sorted(adapters))} | {per_family} x {samples} | run {run_name} |\n")
    ledger = repo / LEDGER
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if not ledger.exists():
        ledger.write_text(LEDGER_HEADER)
    with open(ledger, "a") as f:
        f.write(line)
    return BlindReceipt(vault, line)


def resolve_adapter(run: Run, spec: str) -> Path | None:
    if spec == "base":
        return None
    lineage_id, _, step = spec.partition("@")
    lin = load_lineage(run, lineage_id)
    if lin is None or lin.status == "new":
        raise SystemExit(f"{spec}: lineage has no checkpoint")
    path = run.dir / "members" / lineage_id / "checkpoints" / f"step_{int(step) if step else lin.step:06d}"
    if not (path / "adapter_model.safetensors").exists():
        raise SystemExit(f"{spec}: no adapter at {path}")
    return path


def suite_tasks(run: Run, suite: str, per_family: int, receipt: BlindReceipt | None):
    if suite == "train":
        return fixed_tasks(run.suites.train, "eval", per_family)
    if suite == "selection":
        return fixed_tasks(run.suites.selection, "eval", per_family) + [
            t for s in run.suites.foundation for t in external.items(s, "foundation")[-per_family:]]
    families, sources = blind_suite(run.cfg.suites, suite, receipt)
    return fixed_tasks(families, "eval", per_family) + [
        t for s in sources for t in external.items(s, suite, receipt)[:per_family]]


def summarize(records: list[dict], external_sources: set[str]) -> dict:
    out = {}
    for label in sorted({r["adapter"] for r in records}):
        rs = [r for r in records if r["adapter"] == label]
        fams: dict[str, list[float]] = {}
        for r in rs:
            fams.setdefault(r["family"], []).append(r["reward"])
        acc = {f: sum(v) / len(v) for f, v in sorted(fams.items())}
        procedural = [a for f, a in acc.items() if f not in external_sources]
        out[label] = {"adapter_sha256": rs[0]["adapter_sha256"], "n": len(rs),
                      "n_by_family": {f: len(v) for f, v in sorted(fams.items())},
                      "macro": sum(procedural) / len(procedural) if procedural else None,
                      "by_family": acc, "parsed": sum(r["parsed"] for r in rs) / len(rs),
                      "truncated": sum(r["truncated"] for r in rs) / len(rs),
                      "completion_tokens": sum(r["completion_tokens"] for r in rs) / len(rs)}
    return out


def evaluate(run_dir: Path, suite: str, adapter_specs: list[str], per_family: int, samples: int, shard: str,
             tag: str | None, attach: str | None) -> None:
    run = Run(run_dir)
    setup_logging(run.cfg.name, f"eval-{suite}")
    if suite not in SUITES:
        raise SystemExit(f"unknown suite {suite!r}; choose from {SUITES}")
    adapters = {spec: resolve_adapter(run, spec) for spec in adapter_specs}
    receipt = blind_guard(suite, run.cfg.name, adapters, per_family, samples) if suite in VAULTS else None
    tasks = suite_tasks(run, suite, per_family, receipt)
    check_gold(tasks)
    index, count = (int(x) for x in shard.split("/"))
    jobs = [(label, t) for label in adapters for t in tasks][index::count]
    out = run.dir / "eval" / suite / (tag or f"n{per_family}x{samples}")
    params = {"suite": suite, "per_family": per_family, "samples": samples}
    if (out / "params.json").exists() and read_json(out / "params.json") != params:
        raise SystemExit(f"{out} holds an evaluation with other parameters; choose another --tag")
    write_json(out / "params.json", params)
    log.info("%s: %d tasks x %d adapters, shard %s: %d jobs", suite, len(tasks), len(adapters), shard, len(jobs))

    server, url = start_server(run, attach, f"eval-{suite}")
    sampler = None
    try:
        if server:
            server.wait_ready()
            url = server.url
        sampler = Sampler(url, run.cfg.model, run.cfg.name)
        names = {label: run.cfg.model if path is None else sampler.load_existing(label, path)
                 for label, path in adapters.items()}
        hashes = {label: None if path is None else adapter_sha256(path) for label, path in adapters.items()}
        results = sampler.generate([Request(names[label], run.prompt(t.question), samples,
                                            run.cfg.probes.max_tokens) for label, t in jobs])
    finally:
        if sampler:
            sampler.close()
        if server:
            server.stop()
    records = [{"adapter": label, "adapter_sha256": hashes[label], "suite": suite, "sample": i,
                **response_record(t, s)} for (label, t), group in zip(jobs, results) for i, s in enumerate(group)]
    key = hashlib.sha256(",".join(adapter_specs).encode()).hexdigest()[:8]
    write_jsonl_gz(out / f"records-{key}-{index}of{count}.jsonl.gz", records)
    every = [r for f in sorted(out.glob("records-*.jsonl.gz")) for r in read_jsonl_gz(f)]
    keys = [(r["adapter"], r["uid"], r["sample"]) for r in every]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"{out}: duplicate records across invocations; evaluate each adapter once per tag")
    sources = {s for s in external.PARTS}
    write_json(out / "summary.json", {"time": now(), **params, "adapters": summarize(every, sources)})
    log.info("summary written to %s", out / "summary.json")


def harvest(run_dir: Path, tag: str) -> None:
    """Rank finished lineages by selection macro accuracy of their final adapters; write the harvest manifest."""
    run = Run(run_dir)
    summary = read_json(run.dir / "eval" / "selection" / tag / "summary.json")
    expected = summary["per_family"] * summary["samples"]
    ranking = []
    for path in sorted((run.dir / "members").glob("*/state.json")):
        lin = load_lineage(run, path.parent.name)
        if lin.status != "done":
            continue
        s = summary["adapters"].get(lin.id)
        adapter = resolve_adapter(run, lin.id)
        if s is None or s["adapter_sha256"] != adapter_sha256(adapter):
            raise SystemExit(f"{lin.id}: final adapter not evaluated on selection under tag {tag}")
        if any(n != expected for f, n in s["n_by_family"].items() if f in run.suites.selection):
            raise SystemExit(f"{lin.id}: incomplete selection evaluation under tag {tag}")
        ranking.append({"lineage": lin.id, "slot": lin.slot, "arm": lin.arm, "scaffold": lin.scaffold,
                        "selection_macro": s["macro"], "adapter": str(adapter), "sha256": s["adapter_sha256"]})
    if not ranking:
        raise SystemExit("no finished lineage to rank")
    ranking.sort(key=lambda e: -e["selection_macro"])
    for i, e in enumerate(ranking):
        e["rank"] = i + 1
    manifest = {"run": run.cfg.name, "time": now(), "revision": revision(), "tag": tag, "outlier": ranking[0]["lineage"],
                "top3": [e["lineage"] for e in ranking[:3]], "ranking": ranking}
    write_json(manifest_path(run.cfg.name), manifest)
    print(f"wrote {manifest_path(run.cfg.name)}; commit it before any blind evaluation")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("run")
    e.add_argument("run_dir", type=Path)
    e.add_argument("--suite", required=True, choices=SUITES)
    e.add_argument("--adapters", required=True, help="comma-separated: base, <lineage>, <lineage>@<step>")
    e.add_argument("--per-family", type=int, required=True)
    e.add_argument("--samples", type=int, default=1)
    e.add_argument("--shard", default="0/1", help="i/n: this process evaluates every n-th job from i")
    e.add_argument("--tag", help="output subdirectory (default n<per-family>x<samples>)")
    e.add_argument("--attach", help="URL of a running vLLM server of the run's model (development)")
    h = sub.add_parser("harvest")
    h.add_argument("run_dir", type=Path)
    h.add_argument("--tag", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        evaluate(args.run_dir, args.suite, [a for a in args.adapters.split(",") if a], args.per_family, args.samples,
                 args.shard, args.tag, args.attach)
    else:
        harvest(args.run_dir, args.tag)


if __name__ == "__main__":
    main()
