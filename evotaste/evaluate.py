"""Final evaluation of adapters (or the base model) on a suite, the harvest manifest, the blind guard, and the
context score.

  python -m evotaste.evaluate run runs/<name> --suite selection --adapters base,m00,m01 --per-family 50 [--per-source 500]
  python -m evotaste.evaluate harvest runs/<name> --tag n50x1
  python -m evotaste.evaluate context-score runs/<name> [--members m00,m01]

Suites: the task layer's evaluation sets, selection and the vault blind2 (dataset items and procedural instances).
Adapters: `base`, a lineage id (its last checkpoint), `<lineage>@<step>` or `population@<step>`. `--shard i/n`
splits the (adapter, item) jobs across GPUs; every shard writes its own record file and the summary covers all files
of the tag. A blind vault is opened only through `blind_guard`, which needs a committed, unmodified harvest manifest
listing every adapter with its content hash (base-model calibration is exempt) and appends to the blind ledger
before anything is generated. The context score needs no vLLM: it runs the frozen base model on the visible GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import logging
import subprocess
from pathlib import Path

from evotaste.member import Run, load_lineage, response_records
from evotaste.records import atomic_write, now, read_json, read_jsonl_gz, write_json, write_jsonl_gz
from evotaste.sampler import Request, Sampler
from evotaste.tasks import VAULT, BlindReceipt, check_gold, fixed_tasks, stable_seed
from evotaste.trainer import Policy, Sequence, adapter_sha256
from evotaste.worker import (apply_server, logged, revision, server_options, server_overrides, setup_logging,
                             start_server)

log = logging.getLogger("evotaste.evaluate")
SUITES = ("selection", VAULT)  # the evaluation sets of the task layer; blind1 was spent in Round 1
VAULTS = ("blind1", "blind2")  # names the guard and the ledger know (Round 1's ledger records blind1)
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
    ledger = repo / LEDGER
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if not ledger.exists():
        ledger.write_text(LEDGER_HEADER)
    earlier = [row.split("|") for row in ledger.read_text().splitlines()
               if row.startswith("| ") and row.split("|")[2].strip() == vault]
    reuse = (f"; reuses {vault}, first opened {earlier[0][1].strip()} ({earlier[0][7].strip()})" if earlier else "")
    line = (f"| {now()} | {vault} | {head}{'-dirty' if dirty else ''} | {manifest_hash} | "
            f"{', '.join(sorted(adapters))} | {per_family} x {samples} ({per_source or per_family} per external source) "
            f"| run {run_name}{reuse} |\n")
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
    """Items of a suite: `per_family` per procedural family, `per_source` (default `per_family`; at most what the part
    holds besides the probe set) per dataset source. The vault needs the guard's receipt."""
    return fixed_tasks(run.suites, suite, per_family if per_source is None else per_source, per_family, receipt)


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
    external_sources = {t.family for t in tasks if t.kind != "procedural"}
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
                      "by_family": acc, "complied": sum(r["complied"] for r in rs) / len(rs),
                      "correct_any": sum(r["correct_any"] for r in rs) / len(rs),
                      "truncated": sum(r["truncated"] for r in rs) / len(rs),
                      "completion_tokens": sum(r["completion_tokens"] for r in rs) / len(rs)}
    if incomplete:
        raise IncompleteEvaluation("; ".join(incomplete))
    return out


def read_records(out: Path) -> list[dict]:
    return [r for f in sorted(out.glob("records-*.jsonl.gz")) for r in read_jsonl_gz(f)]


def evaluate(run_dir: Path, suite: str, adapter_specs: list[str], per_family: int, samples: int, shard: str,
             tag: str | None, attach: str | None, per_source: int | None = None, server: dict | None = None) -> None:
    run = Run(run_dir)
    apply_server(run, server)
    if suite not in SUITES:
        raise SystemExit(f"unknown suite {suite!r}; choose from {SUITES}")
    adapter_specs = expand_adapters(run, adapter_specs)
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
        # the seed is a function of the adapter's content and the item: the same in every shard and under any label
        results = sampler.generate([Request(names[label], run.prompt(t.question), samples, run.cfg.probes.max_tokens,
                                            stable_seed("eval", hashes[label] or "base", t.uid)) for label, t in jobs])
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


def slot_chain(run: Run, slot: str) -> list:
    """The slot's lineages in order (read only): the first, then each replacement after a cull."""
    chain, lin = [], load_lineage(run, slot)
    while lin is not None:
        chain.append(lin)
        lin = load_lineage(run, lin.replaced_by) if lin.status == "culled" and lin.replaced_by else None
    return chain


def slot_at_step(run: Run, slot: str, step: int) -> dict:
    """How a population slot enters a harvest at `step` (docs/design/protocol.md, "Harvest rule"): `rank` its
    current lineage's step checkpoint; `young` a replacement that has not reached the step yet; `exhausted` no
    replacement left; `missing` a never-culled slot without that checkpoint, which must stop the harvest."""
    chain = slot_chain(run, slot)
    culls = [{"lineage": lin.id, "step": lin.step, "reason": lin.cull_reason} for lin in chain
             if lin.status == "culled"]
    current = chain[-1] if chain and chain[-1].status != "culled" else None
    path = current and run.dir / "members" / current.id / "checkpoints" / f"step_{step:06d}"
    if current is not None and current.status != "new" and (path / "adapter_model.safetensors").exists():
        return {"kind": "rank", "lineage": current, "label": f"{current.id}@{step}", "path": path}
    if culls:
        return {"kind": "exhausted" if current is None else "young",
                "record": {"slot": slot, "culls": culls, "current": current and current.id,
                           "current_step": current and current.step,
                           "status": "exhausted" if current is None else "young"}}
    where = f"is at step {current.step}" if current is not None else "has not started"
    return {"kind": "missing", "problem": f"{slot}: never culled, but has no checkpoint at step {step} ({where})"}


def population_at(run: Run, step: int) -> list[str]:
    """Labels `<lineage>@<step>` of every population slot that takes part in a harvest at `step`."""
    slots = [slot_at_step(run, m.id, step) for m in run.cfg.members if m.role == "member"]
    missing = [s["problem"] for s in slots if s["kind"] == "missing"]
    if missing:
        raise SystemExit("population@{}: ".format(step) + "; ".join(missing))
    return [s["label"] for s in slots if s["kind"] == "rank"]


def expand_adapters(run: Run, specs: list[str]) -> list[str]:
    """`population@S` stands for population_at(run, S); other labels pass through, order kept, no duplicates."""
    out = []
    for spec in specs:
        name, _, at = spec.partition("@")
        for label in population_at(run, int(at)) if name == "population" and at else [spec]:
            if label not in out:
                out.append(label)
    return out


def step_harvest(run: Run, summary: dict, step: int) -> tuple[list, list, list, list]:
    """(ranking entries, baseline entries, culled-slot records, problems) for a harvest at `step` of every
    population slot as slot_at_step decides; baselines are listed at every step at which they were evaluated."""
    ranking, baselines, culled, problems = [], [], [], []
    for spec in run.cfg.members:
        if spec.role == "baseline":
            chain = slot_chain(run, spec.id)
            lin = chain[-1] if chain else None
            for label in sorted(summary):
                lineage_id, _, at = label.partition("@")
                if lin is None or lineage_id != lin.id:
                    continue
                path = resolve_adapter(run, label)
                if summary[label]["adapter_sha256"] != adapter_sha256(path):
                    problems.append(f"{label}: the evaluated adapter differs from {path}")
                    continue
                baselines.append({"lineage": lin.id, "slot": spec.id, "context": lin.context,
                                  "step": int(at) if at else lin.step, "label": label,
                                  "selection_macro": summary[label]["macro"], "adapter": str(path),
                                  "sha256": summary[label]["adapter_sha256"]})
            continue
        s = slot_at_step(run, spec.id, step)
        if s["kind"] in ("young", "exhausted"):
            culled.append(s["record"])
        elif s["kind"] == "missing":
            problems.append(s["problem"])
        elif s["label"] not in summary:
            problems.append(f"{spec.id}: {s['label']} is not evaluated under this tag")
        elif summary[s["label"]]["adapter_sha256"] != adapter_sha256(s["path"]):
            problems.append(f"{spec.id}: the evaluated {s['label']} differs from {s['path']}")
        else:
            lin = s["lineage"]
            ranking.append({"lineage": lin.id, "slot": spec.id, "context": lin.context,
                            "step": step, "label": s["label"], "selection_macro": summary[s["label"]]["macro"],
                            "adapter": str(s["path"]), "sha256": summary[s["label"]]["adapter_sha256"]})
    return ranking, baselines, culled, problems


def harvest(run_dir: Path, tag: str, step: int | None = None) -> None:
    """Rank population lineages by the selection macro accuracy of their adapters and write the harvest manifest:
    the final adapters of finished lineages, or with `step` every population slot's step-`step` checkpoint
    (evaluated as `<lineage>@<step>`). Baselines (role: baseline) are listed with their hashes but never ranked."""
    run = Run(run_dir)
    out = run.dir / "eval" / "selection" / tag
    params = read_json(out / "params.json")
    # Recomputed from the records with the same checks as the summary: one adapter per label, complete grid.
    tasks = suite_tasks(run, "selection", params["per_family"], None, params.get("per_source"))
    summary = summarize(read_records(out), tasks, params["samples"])
    if step is not None:
        ranking, baselines, culled, problems = step_harvest(run, summary, step)
        if problems:
            raise SystemExit(f"cannot harvest at step {step}:\n  " + "\n  ".join(problems))
        population = any(m.role == "member" for m in run.cfg.members)
        if population and not ranking:
            raise SystemExit(f"no population lineage has reached step {step}")
        if not population and not baselines:
            raise SystemExit(f"a baseline-only run, but no baseline adapter is evaluated under tag {tag}")
        ranking.sort(key=lambda e: -e["selection_macro"])
        for i, e in enumerate(ranking):
            e["rank"] = i + 1
        # A baseline-only run ranks nothing: no outlier, and the manifest only lists the baselines' hashes.
        manifest = {"run": run.cfg.name, "time": now(), "revision": revision(), "tag": tag, "step": step,
                    "outlier": ranking[0]["label"] if ranking else None,
                    "top3": [e["label"] for e in ranking[:3]] if ranking else None, "ranking": ranking,
                    "baselines": baselines, "culled_slots": culled}
        write_json(manifest_path(run.cfg.name), manifest)
        young = sum(c["status"] == "young" for c in culled)
        print(f"step {step}: ranked {len(ranking)}, culled with a young replacement {young}, exhausted "
              f"{len(culled) - young}; baselines {len(baselines)}")
        print(f"wrote {manifest_path(run.cfg.name)}; commit it before any blind evaluation")
        return
    ranking, baselines = [], []
    for path in sorted((run.dir / "members").glob("*/state.json")):
        lin = load_lineage(run, path.parent.name)
        if lin.status != "done":
            continue
        s = summary.get(lin.id)
        adapter = resolve_adapter(run, lin.id)
        if s is None or s["adapter_sha256"] != adapter_sha256(adapter):
            raise SystemExit(f"{lin.id}: final adapter not evaluated on selection under tag {tag}")
        entry = {"lineage": lin.id, "slot": lin.slot, "context": lin.context,
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


# -- context score -------------------------------------------------------------------------------------------------

SCORE_COLUMNS = ("subject", "lineage", "step", "context", "domain", "responses", "tokens", "delta_sum", "delta_mean")


def score_subjects(run: Run, lineages: list[str] | None) -> list[tuple[str, str | None, int | None, Path]]:
    """(subject, lineage, step, probe responses) to score: the base model's plain answers ("base") and its answers
    under each context ("base:<context>"), then every committed probe of every lineage (or of `lineages`)."""
    out = [("base", None, None, run.dir / "base" / "probe_responses.jsonl.gz")]
    out += [(f"base:{c}", None, None, run.dir / "base" / "contexts" / f"{c}.jsonl.gz") for c in sorted(run.contexts)]
    for state in sorted((run.dir / "members").glob("*/state.json")):
        lin = load_lineage(run, state.parent.name)
        if lin.status == "new" or (lineages is not None and lin.id not in lineages):
            continue
        for path in sorted((state.parent / "probe_responses").glob("step_*.jsonl.gz")):
            step = int(path.name[5:11])
            if step <= lin.step:  # later files are not committed (a kill between probe and checkpoint)
                out.append((f"{lin.id}@{step}", lin.id, step, path))
    return out


def context_deltas(run: Run, policy: Policy, records: list[dict]) -> dict[str, list[tuple[str, float, int]]]:
    """Per context, per response: (domain, sum over completion tokens of log p_base(token | prompt with the context)
    - log p_base(token | plain prompt), completion tokens). The completion is the stored sampled token ids."""
    plain = policy.base_logprobs([Sequence(run.prompt(r["question"]), r["token_ids"]) for r in records])
    out = {}
    for context in sorted(run.contexts):
        lps = policy.base_logprobs([Sequence(run.prompt(r["question"], context), r["token_ids"]) for r in records])
        out[context] = [(r["domain"], float((a - b).sum()), len(a)) for r, a, b in zip(records, lps, plain)]
    return out


def score_rows(subject: str, lineage: str | None, step: int | None, deltas: dict) -> list[dict]:
    """The table rows of one subject: per context and domain, the token-mean delta and its parts."""
    rows = []
    for context, per_response in deltas.items():
        for domain in sorted({d for d, _, _ in per_response}):
            mine = [(x, n) for d, x, n in per_response if d == domain]
            total, tokens = sum(x for x, _ in mine), sum(n for _, n in mine)
            rows.append({"subject": subject, "lineage": lineage, "step": step, "context": context, "domain": domain,
                         "responses": len(mine), "tokens": tokens, "delta_sum": total, "delta_mean": total / tokens})
    return rows


def context_score(run_dir: Path, lineages: list[str] | None = None) -> None:
    """For each subject and context, the mean over completion tokens of log p_base(token | context prompt) -
    log p_base(token | plain prompt), by domain, written to runs/<name>/context_score/table.csv. Each subject's rows
    are cached next to it with the SHA-256 of its responses file, so a rerun scores only new or rewritten probes."""
    run = Run(run_dir)
    out = run.dir / "context_score"
    policy = None
    rows = []
    for subject, lineage, step, path in score_subjects(run, lineages):
        cache = out / f"{subject.replace(':', '--')}.json"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        cached = read_json(cache) if cache.exists() else None
        if cached and cached["sha256"] == digest and {r["context"] for r in cached["rows"]} == set(run.contexts):
            rows += cached["rows"]
            continue
        if policy is None:
            policy = Policy(run.cfg.model, run.cfg.lora, run.cfg.trainer)
        records = read_jsonl_gz(path)
        subject_rows = score_rows(subject, lineage, step, context_deltas(run, policy, records))
        write_json(cache, {"time": now(), "responses": str(path), "sha256": digest, "rows": subject_rows})
        rows += subject_rows
        log.info("context score of %s: %d responses", subject, len(records))
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=SCORE_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(out / "table.csv", text.getvalue().encode())
    log.info("wrote %s (%d rows)", out / "table.csv", len(rows))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("run")
    e.add_argument("run_dir", type=Path)
    e.add_argument("--suite", required=True, choices=SUITES)
    e.add_argument("--adapters", required=True, help="comma-separated: base, <lineage>, <lineage>@<step>, and "
                                                     "population@<step> for every population slot at that step")
    e.add_argument("--per-family", type=int, required=True, help="instances per procedural family")
    e.add_argument("--per-source", type=int, help="items per external source (default --per-family; at most the "
                                                   "whole part)")
    e.add_argument("--samples", type=int, default=1)
    e.add_argument("--shard", default="0/1", help="i/n: this process evaluates every n-th job from i")
    e.add_argument("--tag", help="output subdirectory (default n<per-family>x<samples>, plus s<per-source> if set "
                                 "to another value)")
    e.add_argument("--attach", help="URL of a running vLLM server of the run's model (development)")
    server_options(e)
    h = sub.add_parser("harvest")
    h.add_argument("run_dir", type=Path)
    h.add_argument("--tag", required=True)
    h.add_argument("--step", type=int, help="harvest every population slot's checkpoint at this step, evaluated as "
                                            "<lineage>@<step> (default: final adapters of finished lineages)")
    c = sub.add_parser("context-score", help="how much better each context lets the base model predict the stored "
                                             "probe answers (a table; the index is computed in analysis)")
    c.add_argument("run_dir", type=Path)
    c.add_argument("--members", help="comma-separated lineage ids to score (default: all; the base references always)")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        setup_logging(args.run_dir.name, f"eval-{args.suite}")
        logged(evaluate, args.run_dir, args.suite, [a for a in args.adapters.split(",") if a], args.per_family,
               args.samples, args.shard, args.tag, args.attach, args.per_source, server_overrides(args))
    elif args.cmd == "context-score":
        setup_logging(args.run_dir.name, "context-score")
        logged(context_score, args.run_dir, [m for m in args.members.split(",") if m] if args.members else None)
    else:
        harvest(args.run_dir, args.tag, args.step)


if __name__ == "__main__":
    main()
