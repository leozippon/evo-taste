"""Final evaluation of adapters (or the base model) on a suite, the harvest manifest, the blind guard, the context
score, probes of scaled adapters, and the likelihood measures of deviation and distinctness.

  python -m evotaste.evaluate run runs/<name> --suite selection --adapters base,m00,m01 --per-family 50 [--per-source 500]
  python -m evotaste.evaluate harvest runs/<name> --tag n50x1
  python -m evotaste.evaluate context-score runs/<name> [--items 120] [--contexts a,b] [--members m00] [--steps 0,45]
  python -m evotaste.evaluate scaled runs/<name> --adapters m00@45,m01@45 --factors 1.5,2
  python -m evotaste.evaluate deviation runs/<name> [--items 120] [--members m00,m01@45x2] [--steps 45] [--cross]

Suites: the task layer's evaluation sets, selection and the vault blind2 (dataset items and procedural instances).
Adapters: `base`, a lineage id (its last checkpoint), `<lineage>@<step>` or `population@<step>`. `--shard i/n`
splits the (adapter, item) jobs across GPUs; every shard writes its own record file and the summary covers all files
of the tag. A blind vault is opened only through `blind_guard`, which needs a committed, unmodified harvest manifest
listing every trained adapter with its content hash (also for the base model alone) and appends to the blind ledger
before anything is generated. The context score and the deviation need no vLLM: they run the trainer's model on the
visible GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import random
import shutil
import subprocess
import time
from pathlib import Path

from evotaste.member import Run, load_lineage, probe_requests, response_records, summarize_probe
from evotaste.records import atomic_write, now, read_json, read_jsonl_gz, write_json, write_jsonl_gz
from evotaste.sampler import Request, Sampler
from evotaste.tasks import VAULT, BlindReceipt, check_gold, fixed_tasks, stable_seed
from evotaste.trainer import MemberWeights, Policy, Sequence, adapter_sha256, scale_adapter
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
    """Refuse unless the run's harvest manifest is committed and unmodified and lists every trained adapter; then
    append to the ledger. Round 2's rule: no model runs on a blind item before the manifest is committed, the base
    model (`None` in `adapters`) included.
    """
    if vault not in VAULTS:
        raise ValueError(f"{vault!r} is not a blind vault")
    path = manifest_path(run_name, repo)
    rel = str(path.relative_to(repo))
    if not path.exists() or _git(repo, "ls-files", "--error-unmatch", rel).returncode != 0:
        raise PermissionError(f"blind evaluation refused: harvest manifest {rel} is not tracked in git")
    if _git(repo, "diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise PermissionError(f"blind evaluation refused: {rel} differs from the committed version")
    manifest = read_json(path)
    listed = {e["sha256"] for e in manifest["ranking"] + manifest.get("baselines", [])}
    for label, adapter in adapters.items():
        if adapter is not None and adapter_sha256(adapter) not in listed:
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
RESPONSE_COLUMNS = ("subject", "lineage", "step", "uid", "domain", "context", "tokens", "delta_sum")
# Forward-only passes keep no activations, so they take micro-batches of this many times the trainer's token budget.
SCORE_BUDGET_FACTOR = 4


def score_subjects(run: Run, contexts: list[str]) -> list[tuple[str, str | None, int | None, Path]]:
    """(subject, lineage, step, probe responses): the base model's plain answers ("base") and its answers under each
    of `contexts` ("base:<context>"), then every committed probe of every lineage."""
    out = [("base", None, None, run.dir / "base" / "probe_responses.jsonl.gz")]
    out += [(f"base:{c}", None, None, run.dir / "base" / "contexts" / f"{c}.jsonl.gz") for c in contexts]
    for state in sorted((run.dir / "members").glob("*/state.json")):
        lin = load_lineage(run, state.parent.name)
        if lin.status == "new":
            continue
        for path in sorted((state.parent / "probe_responses").glob("step_*.jsonl.gz")):
            step = int(path.name[5:11])
            if step <= lin.step:  # later files are not committed (a kill between probe and checkpoint)
                out.append((f"{lin.id}@{step}", lin.id, step, path))
    return out


def probe_positions(base_records: list[dict], items: int | None) -> tuple[list[int], list[str]]:
    """The probe positions (and their uids) that the context score and the deviation score. With `items`, a
    domain-stratified sample of the probe set (proportional shares, largest remainders, a seeded draw within each
    domain), the same for every subject and both commands; without, or with at least the probe set's size, all."""
    n = len(base_records)
    if items is None or items >= n:
        positions = list(range(n))
    else:
        if items < 1:
            raise SystemExit("--items must be at least 1")
        by_domain: dict[str, list[int]] = {}
        for i, r in enumerate(base_records):
            by_domain.setdefault(r["domain"], []).append(i)
        quota = {d: items * len(v) / n for d, v in sorted(by_domain.items())}
        take = {d: int(q) for d, q in quota.items()}
        for d in sorted(quota, key=lambda d: (take[d] - quota[d], d))[:items - sum(take.values())]:
            take[d] += 1
        positions = sorted(i for d, v in by_domain.items()
                           for i in random.Random(stable_seed("context-score", items, d)).sample(v, take[d]))
    uids = [base_records[i]["uid"] for i in positions]
    if len(set(uids)) != len(uids):
        raise RuntimeError("the probe set repeats a uid; per-response scores would be ambiguous")
    return positions, uids


def score_selection(run: Run, base_records: list[dict], items: int | None, contexts: list[str] | None) -> dict:
    """The probe positions and contexts a context score uses, and the selection's name."""
    contexts = sorted(run.contexts if contexts is None else contexts)
    unknown = sorted(set(contexts) - set(run.contexts))
    if unknown or not contexts:
        raise SystemExit(f"--contexts: unknown or none {unknown} (contexts of the run: {sorted(run.contexts)})")
    positions, uids = probe_positions(base_records, items)
    key = hashlib.sha256(json.dumps({"positions": positions, "contexts": contexts}).encode()).hexdigest()[:10]
    return {"name": f"{len(positions)}items-{len(contexts)}contexts-{key}", "positions": positions,
            "uids": uids, "contexts": contexts}


def context_deltas(run: Run, policy: Policy, records: list[dict], contexts: list[str],
                   token_budget: int | None = None) -> dict[str, list[tuple[str, float, int]]]:
    """Per context, per response: (domain, sum over completion tokens of log p_base(token | prompt with the context)
    - log p_base(token | plain prompt), completion tokens). The completion is the stored sampled token ids; the plain
    and every context's sequences go through the base model in one batched pass."""
    seqs = [Sequence(run.prompt(r["question"], c), r["token_ids"]) for c in (None, *contexts) for r in records]
    lps = policy.base_logprobs(seqs, token_budget)
    n = len(records)
    return {c: [(r["domain"], float((a - b).sum()), len(a))
                for r, a, b in zip(records, lps[k * n:(k + 1) * n], lps[:n])] for k, c in enumerate(contexts, 1)}


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


def context_score(run_dir: Path, items: int | None = None, contexts: list[str] | None = None,
                  lineages: list[str] | None = None, steps: list[int] | None = None) -> None:
    """For each subject and selected context, the mean over completion tokens of log p_base(token | context prompt) -
    log p_base(token | plain prompt), by domain, on the selected probe items. Each selection has its own directory
    runs/<name>/context_score/<selection>/ with selection.json, one cache file per subject (keyed by the SHA-256 of
    its responses file), table.csv, which holds every subject scored under that selection by context and domain, and
    responses.csv with the same values per scored response (uid, context). `lineages` and `steps`
    choose which lineages' probes are scored now; the base references are always scored."""
    run = Run(run_dir)
    base_records = read_jsonl_gz(run.dir / "base" / "probe_responses.jsonl.gz")
    selection = score_selection(run, base_records, items, contexts)
    out = run.dir / "context_score" / selection["name"]
    if (out / "selection.json").exists() and read_json(out / "selection.json") != selection:
        raise SystemExit(f"{out}: selection.json describes another selection")
    write_json(out / "selection.json", selection)
    budget = SCORE_BUDGET_FACTOR * run.cfg.trainer.token_budget
    policy, rows, responses = None, [], []
    for subject, lineage, step, path in score_subjects(run, selection["contexts"]):
        cache = out / f"{subject.replace(':', '--')}.json"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        cached = read_json(cache) if cache.exists() else None
        if cached and cached["sha256"] == digest and isinstance(cached.get("responses"), list):  # older: no values
            rows += cached["rows"]
            responses += cached["responses"]
            continue
        if lineage is not None and ((lineages is not None and lineage not in lineages)
                                    or (steps is not None and step not in steps)):
            continue
        picked = _picked(path, selection["positions"], selection["uids"])
        if policy is None:
            policy = Policy(run.cfg.model, run.cfg.lora, run.cfg.trainer)
        t0 = time.perf_counter()
        deltas = context_deltas(run, policy, picked, selection["contexts"], budget)
        subject_rows = score_rows(subject, lineage, step, deltas)
        subject_responses = [{"subject": subject, "lineage": lineage, "step": step, "uid": r["uid"], "domain": d,
                              "context": c, "tokens": n, "delta_sum": x}
                             for c, values in deltas.items() for r, (d, x, n) in zip(picked, values)]
        seconds = round(time.perf_counter() - t0, 1)
        write_json(cache, {"time": now(), "file": str(path), "sha256": digest, "seconds": seconds,
                           "rows": subject_rows, "responses": subject_responses})
        rows += subject_rows
        responses += subject_responses
        log.info("context score of %s: %d responses x %d prompts in %.0f s", subject, len(picked),
                 1 + len(selection["contexts"]), seconds)
    _csv(out / "table.csv", SCORE_COLUMNS, rows)
    _csv(out / "responses.csv", RESPONSE_COLUMNS, responses)


def _csv(path: Path, columns, rows) -> None:
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, text.getvalue().encode())
    log.info("wrote %s (%d rows)", path, len(rows))


# -- scaled adapters -----------------------------------------------------------------------------------------------

def scaled_label(lineage: str, step: int, factor: float) -> str:
    """`<lineage>@<step>x<factor>`: the probe of a lineage's step-`step` adapter with its update scaled by factor."""
    return f"{lineage}@{step}x{factor:g}"


def scaled(run_dir: Path, specs: list[str], factors: list[float], attach: str | None,
           server: dict | None = None) -> None:
    """For each adapter `<lineage>@<step>` and factor, the adapter whose update to the base model is multiplied by the
    factor (trainer.scale_adapter) answers the run's probe set once, with the seeds of that lineage's own probe at that
    step; the answers are scored and stored like a probe in runs/<name>/scaled/<label>/ (probe_responses.jsonl.gz,
    the scaled adapter, and probe_summary.json, written last). Nothing of the run's members changes; labels done
    before are skipped."""
    run = Run(run_dir)
    apply_server(run, server)
    if not factors or min(factors) <= 0 or any(float(f"{f:g}") != f for f in factors):
        raise SystemExit("--factors must be positive numbers with at most 6 significant digits")
    jobs = []
    for spec in specs:
        lineage, sep, step = spec.partition("@")
        lin = load_lineage(run, lineage) if sep and step.isdigit() else None
        if lin is None or lin.status == "new" or int(step) > lin.step:
            raise SystemExit(f"{spec}: name a committed checkpoint as <lineage>@<step>")
        src = resolve_adapter(run, spec)
        for factor in factors:
            label = scaled_label(lineage, int(step), factor)
            out = run.dir / "scaled" / label
            if (out / "probe_summary.json").exists():
                log.info("%s was probed before; skipped", label)
            elif label not in [j[0] for j in jobs]:
                jobs.append((label, lin, int(step), factor, src, out))
    if not jobs:
        return
    run.check_probe_set()
    for label, _, _, factor, src, out in jobs:
        shutil.rmtree(out, ignore_errors=True)  # an unfinished earlier attempt
        scale_adapter(src, out / "adapter", factor)
    server, url = start_server(run, attach, "scaled")
    sampler = None
    try:
        if server:
            server.wait_ready()
            url = server.url
        sampler = Sampler(url, run.cfg.model, run.cfg.name)
        requests, spans = [], []
        for label, lin, step, _, _, out in jobs:
            reqs = probe_requests(sampler.load_existing(label, out / "adapter"), run.probe_tasks, run.prompt,
                                  run.cfg.probes.max_tokens, lin.seed, step)
            spans.append((len(requests), len(requests) + len(reqs)))
            requests += reqs
        t0 = time.perf_counter()
        samples = sampler.generate(requests)
        log.info("scaled probes of %d adapters: %.0f s of generation", len(jobs), time.perf_counter() - t0)
    finally:
        if sampler:
            sampler.close()
        if server:
            server.stop()
    for (label, lin, step, factor, src, out), (a, b) in zip(jobs, spans):
        records = [{"step": step, "factor": factor, **r} for r in response_records(
            [(t, g[0]) for t, g in zip(run.probe_tasks, samples[a:b])], token_ids=True)]
        write_jsonl_gz(out / "probe_responses.jsonl.gz", records)
        summary = {"label": label, "lineage": lin.id, "step": step, "factor": factor, "time": now(),
                   "source": str(src), "source_sha256": adapter_sha256(src),
                   "adapter_sha256": adapter_sha256(out / "adapter"), **summarize_probe(records)}
        write_json(out / "probe_summary.json", summary)
        log.info("%s: accuracy %.3f, complied %.3f, mean length %.0f", label, summary["accuracy"],
                 summary["complied"], summary["completion_tokens"])


# -- deviation and cross-likelihood --------------------------------------------------------------------------------

DEVIATION_COLUMNS = ("subject", "lineage", "step", "factor", "uid", "domain", "tokens", "delta_sum")
CROSS_COLUMNS = ("scorer", "author", "step", "uid", "domain", "tokens", "logprob_sum")


def probe_authors(run: Run) -> list[dict]:
    """Stored probes and the adapters that wrote them: every committed probe of every lineage (`<lineage>@<step>`,
    its checkpoint at that step) and every finished scaled probe (`<lineage>@<step>x<factor>`, its scaled adapter)."""
    out = [{"label": subject, "lineage": lineage, "step": step, "factor": None, "responses": path,
            "adapter": path.parents[1] / "checkpoints" / f"step_{step:06d}"}
           for subject, lineage, step, path in score_subjects(run, []) if lineage is not None]
    for summary in sorted((run.dir / "scaled").glob("*/probe_summary.json")):
        s = read_json(summary)
        out.append({"label": s["label"], "lineage": s["lineage"], "step": s["step"], "factor": s["factor"],
                    "responses": summary.parent / "probe_responses.jsonl.gz", "adapter": summary.parent / "adapter"})
    return out


def _picked(path: Path, positions: list[int], uids: list[str]) -> list[dict]:
    records = read_jsonl_gz(path)
    picked = [records[i] for i in positions]
    if [r["uid"] for r in picked] != uids:
        raise RuntimeError(f"{path}: the answers are not in the order of the base model's probe answers")
    return picked


def deviation(run_dir: Path, items: int | None = None, names: list[str] | None = None, steps: list[int] | None = None,
              cross: bool = False) -> None:
    """Likelihood measures on stored probe answers (the selection of `probe_positions`, the same items as the context
    score for the same --items). Without `cross`: per response of every selected author, its completion tokens and
    the sum over them of log p_author - log p_base, where the author is the adapter that wrote the response (a
    lineage's checkpoint at the probe's step, or a scaled adapter); written to responses.csv. With `cross`, for one
    step: every selected adapter at that step and the base model score every one's responses (the base model's plain
    probe answers included), giving the summed log-probability per (scorer, author, item) in cross-<step>.csv. Authors
    are selected by lineage id or label (`names`) and by step; scaled probes count as their lineage's at their step.
    Results are cached per subject (or per scorer and author), keyed by the SHA-256 of the responses file and the
    scoring adapter's content hash, in runs/<name>/deviation/<N>items-<hash>/."""
    run = Run(run_dir)
    base_path = run.dir / "base" / "probe_responses.jsonl.gz"
    positions, uids = probe_positions(read_jsonl_gz(base_path), items)
    key = hashlib.sha256(json.dumps({"positions": positions}).encode()).hexdigest()[:10]
    out = run.dir / "deviation" / f"{len(positions)}items-{key}"
    selection = {"positions": positions, "uids": uids}
    if (out / "selection.json").exists() and read_json(out / "selection.json") != selection:
        raise SystemExit(f"{out}: selection.json describes another selection")
    write_json(out / "selection.json", selection)
    budget = SCORE_BUDGET_FACTOR * run.cfg.trainer.token_budget
    authors = probe_authors(run)

    def chosen(a):
        return ((names is None or a["lineage"] in names or a["label"] in names)
                and (steps is None or a["step"] in steps))
    policy = None

    def scores(adapter: Path | None, seqs: list[Sequence]) -> list:
        nonlocal policy
        if policy is None:
            policy = Policy(run.cfg.model, run.cfg.lora, run.cfg.trainer)
        if adapter is None:
            return policy.base_logprobs(seqs, budget)
        return policy.logprobs(MemberWeights(policy.load_adapter(adapter)), seqs, budget)

    def sequences(records):
        return [Sequence(run.prompt(r["question"]), r["token_ids"]) for r in records]

    if not cross:
        rows = []
        for a in authors:
            cache = out / f"{a['label']}.json"
            digest = hashlib.sha256(a["responses"].read_bytes()).hexdigest()
            cached = read_json(cache) if cache.exists() else None
            if cached is None or cached["sha256"] != digest:
                cached = None
                if not chosen(a):
                    continue
            ident = {"sha256": digest, "adapter_sha256": adapter_sha256(a["adapter"])}
            if cached and cached["adapter_sha256"] == ident["adapter_sha256"]:
                rows += cached["responses"]
                continue
            if not chosen(a):
                continue
            t0 = time.perf_counter()
            picked = _picked(a["responses"], positions, uids)
            seqs = sequences(picked)
            mine, base = scores(a["adapter"], seqs), scores(None, seqs)
            responses = [{"subject": a["label"], "lineage": a["lineage"], "step": a["step"], "factor": a["factor"],
                          "uid": r["uid"], "domain": r["domain"], "tokens": len(m), "delta_sum": float((m - b).sum())}
                         for r, m, b in zip(picked, mine, base)]
            seconds = round(time.perf_counter() - t0, 1)
            write_json(cache, {"time": now(), "file": str(a["responses"]), **ident, "seconds": seconds,
                               "responses": responses})
            rows += responses
            tokens = sum(x["tokens"] for x in responses)
            log.info("deviation of %s: %.4f nats per token over %d responses (%d tokens) in %.0f s", a["label"],
                     sum(x["delta_sum"] for x in responses) / tokens, len(responses), tokens, seconds)
        _csv(out / "responses.csv", DEVIATION_COLUMNS, rows)
        return

    if steps is None or len(steps) != 1:
        raise SystemExit("--cross scores the adapters of one step: give exactly one --steps value")
    step = steps[0]
    subjects = [{"label": "base", "responses": base_path, "adapter": None}] + [
        a for a in authors if a["step"] == step and chosen(a)]
    digests = {s["label"]: hashlib.sha256(s["responses"].read_bytes()).hexdigest() for s in subjects}
    picked = {s["label"]: _picked(s["responses"], positions, uids) for s in subjects}
    rows = []
    for scorer in subjects:
        scorer_hash = "base" if scorer["adapter"] is None else adapter_sha256(scorer["adapter"])
        pending = []
        for author in subjects:
            cache = out / f"cross-{step}" / f"{scorer['label']}--{author['label']}.json"
            cached = read_json(cache) if cache.exists() else None
            if cached and (cached["sha256"], cached["adapter_sha256"]) == (digests[author["label"]], scorer_hash):
                rows += cached["responses"]
            else:
                pending.append((author, cache))
        if not pending:
            continue
        t0 = time.perf_counter()
        lps = iter(scores(scorer["adapter"], [s for author, _ in pending for s in sequences(picked[author["label"]])]))
        for author, cache in pending:
            responses = [{"scorer": scorer["label"], "author": author["label"], "step": step, "uid": r["uid"],
                          "domain": r["domain"], "tokens": len(lp), "logprob_sum": float(lp.sum())}
                         for r, lp in zip(picked[author["label"]], lps)]
            write_json(cache, {"time": now(), "file": str(author["responses"]), "sha256": digests[author["label"]],
                               "adapter_sha256": scorer_hash, "responses": responses})
            rows += responses
        log.info("cross-likelihood: %s scored %d authors in %.0f s", scorer["label"], len(pending),
                 time.perf_counter() - t0)
    _csv(out / f"cross-{step}.csv", CROSS_COLUMNS, rows)


def _names(text: str | None) -> list[str] | None:
    """A comma-separated command-line list; None when the option is absent."""
    return [x for x in text.split(",") if x] if text else None


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
    c.add_argument("--items", type=int, help="a domain-stratified sample of this many probe items, the same for every "
                                              "subject (default: all)")
    c.add_argument("--contexts", help="comma-separated contexts (default: every context of the run)")
    c.add_argument("--members", help="comma-separated lineage ids to score (default: all; the base references always)")
    c.add_argument("--steps", help="comma-separated probe steps to score (default: all)")
    s = sub.add_parser("scaled", help="probe adapters whose update to the base model is scaled by factors")
    s.add_argument("run_dir", type=Path)
    s.add_argument("--adapters", required=True, help="comma-separated <lineage>@<step>")
    s.add_argument("--factors", required=True, help="comma-separated factors, e.g. 1.5,2,3")
    s.add_argument("--attach", help="URL of a running vLLM server of the run's model (development)")
    server_options(s)
    d = sub.add_parser("deviation", help="log p_author - log p_base of stored probe answers, or with --cross every "
                                         "adapter's log-probability of every adapter's answers at one step")
    d.add_argument("run_dir", type=Path)
    d.add_argument("--items", type=int, help="the context score's domain-stratified sample of this many probe items "
                                              "(default: all)")
    d.add_argument("--members", help="comma-separated lineage ids or labels (default: all, scaled probes included)")
    d.add_argument("--steps", help="comma-separated probe steps (default: all; --cross needs exactly one)")
    d.add_argument("--cross", action="store_true", help="every selected adapter and the base model score every one's "
                                                        "answers")
    args = ap.parse_args(argv)
    steps = [int(x) for x in _names(args.steps)] if getattr(args, "steps", None) else None
    if args.cmd == "run":
        setup_logging(args.run_dir.name, f"eval-{args.suite}")
        logged(evaluate, args.run_dir, args.suite, _names(args.adapters), args.per_family,
               args.samples, args.shard, args.tag, args.attach, args.per_source, server_overrides(args))
    elif args.cmd == "context-score":
        setup_logging(args.run_dir.name, "context-score")
        logged(context_score, args.run_dir, args.items, _names(args.contexts), _names(args.members), steps)
    elif args.cmd == "scaled":
        setup_logging(args.run_dir.name, "scaled")
        logged(scaled, args.run_dir, _names(args.adapters), [float(x) for x in _names(args.factors)], args.attach,
               server_overrides(args))
    elif args.cmd == "deviation":
        setup_logging(args.run_dir.name, "deviation")
        logged(deviation, args.run_dir, args.items, _names(args.members), steps, args.cross)
    else:
        harvest(args.run_dir, args.tag, args.step)


if __name__ == "__main__":
    main()
