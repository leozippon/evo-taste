"""The seeding pilot: per seeded member, its distillation trajectory and what it carries straight after seeding.

    python -m evotaste.analysis.seeding runs/<name> [--table <context-score table>] [--out DIR]

For every member that is not a clone: the KL of each opcd update (correct teacher answers for imitation), the raw and
specific context index overall and by domain, identification of its own context and the jargon share at step 0 (from
evotaste.analysis.washout), and its step-0 probe accuracy by domain against the unseeded member (the base model's
probe answers when the run has none), length and share of open answers given by program. Writes
results/<name>/seeding/: members.csv, kl.csv, accuracy.csv and report.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evotaste.analysis import data, health, stats, washout
from evotaste.analysis.report import num, pci, pct, render, write


def kl_table(run: data.Run) -> pd.DataFrame:
    rows = []
    for lin in run.lineages.values():
        if lin.parent is None:
            for r in run.history(lin)[0]:
                rows.append({"lineage": lin.id, "method": r["method"], "update": r["step"], "kl": r.get("kl"),
                             "loss": r["loss"], "correct": r.get("correct"), "tokens": r["completion_tokens"]})
    return pd.DataFrame(rows, columns=["lineage", "method", "update", "kl", "loss", "correct", "tokens"])


def accuracy_table(run: data.Run, rng) -> tuple[pd.DataFrame, str]:
    """Step-0 probe metrics of every non-clone member, each against the reference (the unseeded member, else the base
    model's plain probe answers), with paired item-bootstrap intervals."""
    seeded = [lin for lin in run.lineages.values() if lin.parent is None and 0 in run.probe_steps(lin)]
    unseeded = [lin for lin in seeded if lin.level == "unseeded"]
    reference = f"{unseeded[0].id}@0" if unseeded else "base"
    subjects = {"base": (None, None)} | {f"{lin.id}@0": (lin, 0) for lin in seeded}
    grid = data.probe_grid(run, subjects)
    point, draws = health.evaluate(grid, rng)
    keys = [k for k in point if k.startswith("domain_")] + ["accuracy", "tokens", "program_share", "truncated",
                                                             "heldout_complied"]
    ri = grid.labels.index(reference)
    rows = []
    for s, label in enumerate(grid.labels):
        for k in keys:
            d = draws[k][s] - draws[k][ri]
            lo, hi = np.nanquantile(d, [0.025, 0.975]) if s != ri else (np.nan, np.nan)
            rows.append({"subject": label, "metric": k, "value": point[k][s, 0], "diff": point[k][s, 0] - point[k][ri, 0]
                         if s != ri else np.nan, "diff_lo": lo, "diff_hi": hi})
    return pd.DataFrame(rows), reference


def report_lines(run, members, kl, acc, reference, wo) -> list[str]:
    lines = [f"seeding {run.name}: step-0 probes of {len(members)} non-clone members; accuracy differences against "
             f"{reference} on the same probe items [95% bootstrap over items]"]
    cell = {(r.subject, r.metric): r for r in acc.itertuples()}
    domains = sorted(k for k in {m for _, m in cell} if k.startswith("domain_"))
    rows = []
    for r in members.itertuples():
        label = f"{r.lineage}@0"
        k = kl[kl["lineage"] == r.lineage]
        traj = ("-" if k.empty else f"{k['kl'].iloc[0]:.4f} -> {k['kl'].iloc[-1]:.4f}" if k["kl"].notna().all()
                else f"correct {k['correct'].iloc[0]:.0f} -> {k['correct'].iloc[-1]:.0f}")
        rows.append([r.lineage, r.level, r.context if isinstance(r.context, str) else "-", r.seeding, traj, num(r.raw, digits=3),
                     num(r.specific, digits=3), "-" if r.identified is None else ("yes" if r.identified else "no"),
                     pct(r.jargon)]
                    + [pci(cell[(label, d)].diff, cell[(label, d)].diff_lo, cell[(label, d)].diff_hi, sign=True)
                       if label != reference else pct(cell[(label, d)].value) for d in domains]
                    + [f"{cell[(label, 'tokens')].value:.0f}", pct(cell[(label, 'program_share')].value)])
    lines += render(["member", "level", "context", "seeding", "kl first -> last", "raw", "specific", "own top",
                     "jargon"] + [f"{d[7:]} - ref" for d in domains] + ["tokens", "program"], rows)
    lines += ["", "specific index by domain (seeds distilled on some domains only show cross-domain coherence)"]
    s = wo["specific"]
    s = s[(s["step"] == 0) & s["lineage"].isin(members["lineage"])] if len(s) else s
    if len(s):
        wide = s.pivot_table(index="lineage", columns="domain", values="specific")
        lines += render(["member"] + list(wide.columns),
                        [[lin] + [num(v, digits=3) for v in r] for lin, r in wide.iterrows()])
    else:
        lines.append("  no context scores at step 0")
    return lines + [washout.NO_ITEM_INTERVALS, washout.noise_line(wo["noise"])]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--table", type=Path, help="a context-score table (default: the run's only selection)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    run = data.Run(args.run_dir)
    rng = np.random.default_rng(0)
    wo = washout.analyse(run, rng, args.table)
    acc, reference = accuracy_table(run, rng)
    spec0 = wo["specific"][(wo["specific"]["step"] == 0) & (wo["specific"]["domain"] == washout.DOMAINS_ALL)] \
        if len(wo["specific"]) else pd.DataFrame(columns=["lineage", "raw", "specific"])
    jargon = wo["jargon"][wo["jargon"]["step"] == 0].set_index("lineage")["share"]
    hits = wo["hits"][wo["hits"]["step"] == 0]
    ident = dict(zip(hits["lineage"], hits["hit"]))
    rows = []
    for lin in run.lineages.values():
        if lin.parent is not None or 0 not in run.probe_steps(lin):
            continue
        sp = spec0[spec0["lineage"] == lin.id]
        rows.append({"lineage": lin.id, "level": lin.level, "context": lin.context, "seeding": lin.seeding,
                     "arm": lin.arm, "raw": sp["raw"].iloc[0] if len(sp) else np.nan,
                     "specific": sp["specific"].iloc[0] if len(sp) else np.nan, "identified": ident.get(lin.id),
                     "jargon": jargon.get(lin.id, np.nan)})
    members = pd.DataFrame(rows)
    kl = kl_table(run)
    write(args.out or Path("results") / run.name / "seeding", {"members": members, "kl": kl, "accuracy": acc},
          report_lines(run, members, kl, acc, reference, wo))


if __name__ == "__main__":
    with threadpool_limits(stats.THREADS):
        main()
