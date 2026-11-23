"""Final evaluation of adapters (or the base model) on a suite, the harvest manifest, and the blind guard.

  python -m evotaste.evaluate run runs/<name> --suite selection --adapters base,m00,m01 --per-family 50 [--per-source 500]
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
from evotaste.member import Run, load_lineage, response_records
from evotaste.records import now, read_json, read_jsonl_gz, write_json, write_jsonl_gz
from evotaste.sampler import Request, Sampler
from evotaste.tasks import BlindReceipt, blind_suite, check_gold, fixed_tasks
from evotaste.trainer import adapter_sha256
from evotaste.worker import logged, revision, setup_logging, start_server

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
                repo: Path = Path("."), per_source: int | None = None) -> BlindReceipt:
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
        listed = {e["sha256"] for e in manifest["ranking"] + manifest.get("baselines", [])}
        for label, adapter in trained.items():
            if adapter_sha256(adapter) not in listed:
                raise PermissionError(f"blind evaluation refused: adapter {label} ({adapter}) is not in {rel} "
                                      "with a matching content hash")
        manifest_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    dirty = bool(_git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip())
    line = (f"| {now()} | {vault} | {head}{'-dirty' if dirty else ''} | {manifest_hash} | "
            f"{', '.join(sorted(adapters))} | {per_family} x {samples} ({per_source or per_family} per external source) "
            f"| run {run_name} |\n")
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


def suite_tasks(run: Run, suite: str, per_family: int, receipt: BlindReceipt | None, per_source: int | None = None):
    """Instances of a suite: `per_family` per procedural family, `per_source` (default `per_family`; at most the
    whole part) per external source."""
    per_source = per_family if per_source is None else per_source
    if suite == "train":
        return fixed_tasks(run.suites.train, "eval", per_family)
    if suite == "selection":
        return fixed_tasks(run.suites.selection, "eval", per_family) + [
            t for s in run.suites.foundation for t in external.items(s, "foundation")[-per_source:]]
    families, sources = blind_suite(run.cfg.suites, suite, receipt)
    return fixed_tasks(families, "eval", per_family) + [
        t for s in sources for t in external.items(s, suite, receipt)[:per_source]]


def eval_params(suite: str, per_family: int, samples: int, per_source: int | None) -> tuple[dict, str]:
    """params.json of an evaluation and its default tag; the tag names per_source only when it differs."""
    per_source = per_family if per_source is None else per_source
    if min(per_family, samples, per_source) < 1:
        raise SystemExit("--per-family, --samples and --per-source must be at least 1")
    tag = f"n{per_family}x{samples}" + (f"s{per_source}" if per_source != per_family else "")
    return {"suite": suite, "per_family": per_family, "samples": samples, "per_source": per_source}, tag


def same_params(old: dict, new: dict) -> bool:
    """params.json written before per_source existed meant per_source == per_family."""
    return {"per_source": old.get("per_family"), **old} == new


class IncompleteEvaluation(Exception):
    """Some label lacks records of the requested grid (for example while other shards are still running)."""


def summarize(records: list[dict], tasks: list, samples: int) -> dict:
    """Per-label accuracy, after checking that each label comes from a single adapter (one hash across all record
    files) and covers the requested grid exactly: every (family, instance, sample) once. Raises IncompleteEvaluation
    when records are only missing, ValueError for anything else."""
    expected = {(t.family, t.uid, i) for t in tasks for i in range(samples)}
    external_sources = {t.family for t in tasks if t.entry is None}
    out, incomplete = {}, []
    for label in sorted({r["adapter"] for r in records}):
        rs = [r for r in records if r["adapter"] == label]
        hashes = {r["adapter_sha256"] for r in rs}
        if len(hashes) != 1:
            raise ValueError(f"{label}: records from {len(hashes)} different adapters {sorted(map(str, hashes))}")
        got = [(r["family"], r["uid"], r["sample"]) for r in rs]
        if len(set(got)) != len(got) or not set(got) <= expected:
            raise ValueError(f"{label}: duplicate records, or records outside the requested grid")
        if len(got) < len(expected):
            incomplete.append(f"{label}: {len(got)} of {len(expected)} records")
            continue
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
    if incomplete:
        raise IncompleteEvaluation("; ".join(incomplete))
    return out


def read_records(out: Path) -> list[dict]:
    return [r for f in sorted(out.glob("records-*.jsonl.gz")) for r in read_jsonl_gz(f)]


def evaluate(run_dir: Path, suite: str, adapter_specs: list[str], per_family: int, samples: int, shard: str,
             tag: str | None, attach: str | None, per_source: int | None = None) -> None:
    run = Run(run_dir)
    if suite not in SUITES:
        raise SystemExit(f"unknown suite {suite!r}; choose from {SUITES}")
    adapters = {spec: resolve_adapter(run, spec) for spec in adapter_specs}
    params, default_tag = eval_params(suite, per_family, samples, per_source)
    receipt = (blind_guard(suite, run.cfg.name, adapters, per_family, samples, per_source=params["per_source"])
               if suite in VAULTS else None)
    tasks = suite_tasks(run, suite, per_family, receipt, params["per_source"])
    check_gold(tasks)
    index, count = (int(x) for x in shard.split("/"))
    jobs = [(label, t) for label in adapters for t in tasks][index::count]
    out = run.dir / "eval" / suite / (tag or default_tag)
    if (out / "params.json").exists() and not same_params(read_json(out / "params.json"), params):
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
    pairs = [(label, t, i, s) for (label, t), group in zip(jobs, results) for i, s in enumerate(group)]
    scored = response_records([(t, s) for _, t, _, s in pairs])
    records = [{"adapter": label, "adapter_sha256": hashes[label], "suite": suite, "sample": i, **r}
               for (label, _, i, _), r in zip(pairs, scored)]
    key = hashlib.sha256(",".join(adapter_specs).encode()).hexdigest()[:8]
    write_jsonl_gz(out / f"records-{key}-{index}of{count}.jsonl.gz", records)
    try:
        summary = summarize(read_records(out), tasks, samples)
    except IncompleteEvaluation as e:
        (out / "summary.json").unlink(missing_ok=True)
        log.warning("no summary yet, the evaluation of %s is incomplete (shards still running?): %s", out, e)
        return
    write_json(out / "summary.json", {"time": now(), **params, "adapters": summary})
    log.info("summary written to %s", out / "summary.json")


def harvest(run_dir: Path, tag: str) -> None:
    """Rank finished population lineages by selection macro accuracy of their final adapters and write the harvest
    manifest; finished baselines (role: baseline) are listed with their hashes but never ranked."""
    run = Run(run_dir)
    out = run.dir / "eval" / "selection" / tag
    params = read_json(out / "params.json")
    # Recomputed from the records with the same checks as the summary: one adapter per label, complete grid.
    tasks = suite_tasks(run, "selection", params["per_family"], None, params.get("per_source"))
    summary = summarize(read_records(out), tasks, params["samples"])
    ranking, baselines = [], []
    for path in sorted((run.dir / "members").glob("*/state.json")):
        lin = load_lineage(run, path.parent.name)
        if lin.status != "done":
            continue
        s = summary.get(lin.id)
        adapter = resolve_adapter(run, lin.id)
        if s is None or s["adapter_sha256"] != adapter_sha256(adapter):
            raise SystemExit(f"{lin.id}: final adapter not evaluated on selection under tag {tag}")
        entry = {"lineage": lin.id, "slot": lin.slot, "arm": lin.arm, "scaffold": lin.scaffold,
                 "selection_macro": s["macro"], "adapter": str(adapter), "sha256": s["adapter_sha256"]}
        (baselines if run.cfg.member(lin.slot).role == "baseline" else ranking).append(entry)
    if not ranking:
        raise SystemExit("no finished lineage to rank")
    ranking.sort(key=lambda e: -e["selection_macro"])
    for i, e in enumerate(ranking):
        e["rank"] = i + 1
    manifest = {"run": run.cfg.name, "time": now(), "revision": revision(), "tag": tag, "outlier": ranking[0]["lineage"],
                "top3": [e["lineage"] for e in ranking[:3]], "ranking": ranking, "baselines": baselines}
    write_json(manifest_path(run.cfg.name), manifest)
    print(f"wrote {manifest_path(run.cfg.name)}; commit it before any blind evaluation")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("run")
    e.add_argument("run_dir", type=Path)
    e.add_argument("--suite", required=True, choices=SUITES)
    e.add_argument("--adapters", required=True, help="comma-separated: base, <lineage>, <lineage>@<step>")
    e.add_argument("--per-family", type=int, required=True, help="instances per procedural family")
    e.add_argument("--per-source", type=int, help="items per external source (default --per-family; at most the "
                                                   "whole part)")
    e.add_argument("--samples", type=int, default=1)
    e.add_argument("--shard", default="0/1", help="i/n: this process evaluates every n-th job from i")
    e.add_argument("--tag", help="output subdirectory (default n<per-family>x<samples>, plus s<per-source> if set "
                                 "to another value)")
    e.add_argument("--attach", help="URL of a running vLLM server of the run's model (development)")
    h = sub.add_parser("harvest")
    h.add_argument("run_dir", type=Path)
    h.add_argument("--tag", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        setup_logging(args.run_dir.name, f"eval-{args.suite}")
        logged(evaluate, args.run_dir, args.suite, [a for a in args.adapters.split(",") if a], args.per_family,
               args.samples, args.shard, args.tag, args.attach, args.per_source)
    else:
        harvest(args.run_dir, args.tag)


if __name__ == "__main__":
    main()
