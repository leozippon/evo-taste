"""The seeding pilot: per seeded member, its distillation trajectory and what it carries straight after seeding.

    python -m evotaste.analysis.seeding runs/<name> [--selection <context-score selection>] [--out DIR]

For every member that is not a clone: the KL of each opcd update (correct teacher answers for imitation); the specific
context index with its two parts, gain and effect in nats per token, overall and by domain, identification of its own
context and the jargon share at step 0 (from evotaste.analysis.washout); and its step-0 probe accuracy by domain
against the reference, the mean of every probe of the base weights (the base model's prepare-time answers and the
step-0 probes of unseeded members and their clones). A member's own value averages every probe of its step-0
checkpoint (its own and its clones' first probes). Intervals resample items within units, the same draws on both
sides, so they carry the sampling noise of both. Writes results/<name>/seeding/: members.csv, domains.csv, kl.csv,
accuracy.csv and report.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evotaste.analysis import data, health, stats, washout
from evotaste.analysis.report import num, pci, pct, render, write

METRICS = ("accuracy", "domain_code", "domain_logic", "domain_math", "tokens", "truncated", "heldout_complied")


def seeded(run: data.Run) -> list[data.Lineage]:
    """Members that are not clones and have a committed step-0 probe, the unseeded reference members excepted."""
    return [lin for lin in run.lineages.values()
            if lin.parent is None and lin.seeding != "none" and 0 in run.probe_steps(lin)]


def kl_table(run: data.Run) -> pd.DataFrame:
    rows = []
    for lin in seeded(run):
        for r in run.history(lin)[0]:
            rows.append({"lineage": lin.id, "method": r["method"], "update": r["step"], "kl": r.get("kl"),
                         "loss": r["loss"], "correct": r.get("correct"), "tokens": r["completion_tokens"]})
    return pd.DataFrame(rows, columns=["lineage", "method", "update", "kl", "loss", "correct", "tokens"])


def accuracy_table(run: data.Run, rng) -> pd.DataFrame:
    """Step-0 probe metrics of every seeded member (averaging the probes of its step-0 checkpoint) and of the
    reference (averaging every probe of the base weights), with the member - reference difference."""
    subjects = {"base": (None, None)}
    for lin in run.lineages.values():
        if 0 in run.probe_steps(lin):
            subjects[f"{lin.id}@0"] = (lin, 0)
    grid = data.probe_grid(run, subjects)
    point, draws = health.evaluate(grid, rng)
    ref = health.reference_rows(run, grid.labels, subjects.get)
    rp, rd = health.pooled(point, ref), health.pooled(draws, ref)
    keys = [k for k in METRICS if k in point]
    rows = [{"subject": health.REFERENCE, "metric": k, "draws": len(ref), "value": rp[k][0],
             **dict(zip(("lo", "hi"), stats.ci(rd[k])))} for k in keys]
    for lin in seeded(run):
        own = [i for i, a in enumerate(grid.labels) if subjects[a][0] is not None
               and run.state(*subjects[a]) == (lin.id, 0)]
        mp, md = health.pooled(point, own), health.pooled(draws, own)
        rows += [{"subject": lin.id, "metric": k, "draws": len(own), "value": mp[k][0],
                  **dict(zip(("lo", "hi"), stats.ci(md[k]))), "diff": mp[k][0] - rp[k][0],
                  **dict(zip(("diff_lo", "diff_hi"), stats.ci(md[k] - rd[k])))} for k in keys]
    return pd.DataFrame(rows)


def report_lines(run, members, domains, kl, acc, wo) -> list[str]:
    ref = acc[acc["subject"] == health.REFERENCE].set_index("metric")
    lines = [f"seeding {run.name}: {len(members)} seeded members at step 0 (clones excluded); [95% intervals resample "
             "items]; specific = gain / effect, gain and effect in nats per token"]
    lines += (washout.selection_lines(wo["_index"]) + [washout.noise_line(wo["noise"])] if wo["_index"] else
              ["no context-score selection with per-response scores yet (python -m evotaste.evaluate context-score)"])
    lines.append("")
    rows = []
    for r in members.itertuples():
        k = kl[kl["lineage"] == r.lineage]
        traj = ("-" if k.empty else f"{k['kl'].iloc[0]:.4f} -> {k['kl'].iloc[-1]:.4f}" if k["kl"].notna().all()
                else f"correct {k['correct'].iloc[0]:.0f} -> {k['correct'].iloc[-1]:.0f}")
        rows.append([r.lineage, r.level, r.context if isinstance(r.context, str) else "-", r.seeding, traj,
                     num(r.specific, r.specific_lo, r.specific_hi, 3), num(r.gain, r.gain_lo, r.gain_hi, 4),
                     num(r.effect, r.effect_lo, r.effect_hi, 4),
                     "-" if pd.isna(r.identified) else ("yes" if r.identified else f"no ({r.top})"), pct(r.jargon)])
    lines += render(["member", "level", "context", "seeding", "kl first -> last", "specific [95%]", "gain [95%]",
                     "effect [95%]", "own top", "jargon"], rows)
    if len(domains):
        lines += ["", "by domain: specific index, gain and effect (nats per token)"]
        lines += render(["member", "domain", "specific [95%]", "gain [95%]", "effect [95%]"],
                        [[r.lineage, r.domain, num(r.specific, r.specific_lo, r.specific_hi, 3),
                          num(r.gain, r.gain_lo, r.gain_hi, 4), num(r.effect, r.effect_lo, r.effect_hi, 4)]
                         for r in domains.itertuples()])
    if len(ref):
        cols = [k for k in METRICS if k.startswith("domain_") or k == "accuracy"]
        lines += ["", "step-0 probe accuracy against the reference, the mean of "
                  f"{int(ref.loc['accuracy', 'draws'])} probes of the base weights: "
                  + ", ".join(f"{k.removeprefix('domain_')} {pct(ref.loc[k, 'value'])}" for k in cols)]
        cell = {(r.subject, r.metric): r for r in acc.itertuples()}
        lines += render(["member", "draws"] + [f"{k.removeprefix('domain_')} - ref" for k in cols] + ["tokens"],
                        [[lin, int(cell[(lin, "accuracy")].draws)]
                         + [pci(cell[(lin, k)].diff, cell[(lin, k)].diff_lo, cell[(lin, k)].diff_hi, sign=True)
                            for k in cols] + [f"{cell[(lin, 'tokens')].value:.0f}"]
                         for lin in dict.fromkeys(acc.loc[acc["subject"] != health.REFERENCE, "subject"])])
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--selection", type=Path, help="a context-score selection directory (default: the run's only one)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    run = data.Run(args.run_dir)
    rng = np.random.default_rng(0)
    wo = washout.analyse(run, rng, args.selection) if args.selection or len(data.selections(run)) == 1 else None
    if wo is None:  # no context scores yet: accuracy only
        wo = {"specific": pd.DataFrame(), "hits": pd.DataFrame(columns=["step", "lineage", "hit", "top"]),
              "jargon": washout.jargon_table(run), "noise": pd.DataFrame(), "_index": None}
    spec = wo["specific"]
    spec = spec[(spec["step"] == 0) & spec["lineage"].isin([lin.id for lin in seeded(run)])] if len(spec) else spec
    hits = wo["hits"][wo["hits"]["step"] == 0].set_index("lineage")
    jargon = wo["jargon"][wo["jargon"]["step"] == 0].set_index("lineage")["share"]
    cols = ["specific", "gain", "effect"]
    keep = [f"{c}{s}" for c in cols for s in ("", "_lo", "_hi")]
    rows = []
    for lin in seeded(run):
        sp = spec[(spec["lineage"] == lin.id) & (spec["domain"] == washout.DOMAINS_ALL)] if len(spec) else spec
        rows.append({"lineage": lin.id, "level": lin.level, "context": lin.context, "seeding": lin.seeding,
                     **({k: sp[k].iloc[0] for k in keep} if len(sp) else dict.fromkeys(keep, np.nan)),
                     "identified": hits["hit"].get(lin.id), "top": hits["top"].get(lin.id),
                     "jargon": jargon.get(lin.id, np.nan)})
    domains = (spec[spec["domain"] != washout.DOMAINS_ALL][["lineage", "domain"] + keep] if len(spec)
               else pd.DataFrame(columns=["lineage", "domain"] + keep))
    members, kl, acc = pd.DataFrame(rows), kl_table(run), accuracy_table(run, rng)
    write(args.out or Path("results") / run.name / "seeding",
          {"members": members, "domains": domains, "kl": kl, "accuracy": acc},
          report_lines(run, members, domains, kl, acc, wo))


if __name__ == "__main__":
    with threadpool_limits(stats.THREADS):
        main()
