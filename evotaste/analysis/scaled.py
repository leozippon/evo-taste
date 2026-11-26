"""Scaled probes: accuracy, compliance, length and deviation of an adapter whose update to the base model is
multiplied by a factor (python -m evotaste.evaluate scaled), against the factor, per adapter.

    python -m evotaste.analysis.scaled runs/<name> [--selection DIR] [--partial] [--boot 1000] [--out DIR]

For every scaled adapter <lineage>@<step>: factor 0 is the reference (the mean of the probes of the base model's
weights, health's reference), factor 1 the lineage's own probe at that step, whose seeds the scaled probes share, so
that the differences from factor 1 are paired draws. Metrics are health's (item means over the whole probe set);
the deviation is read from a deviation selection that holds the source and its scaled probes (the only such one,
or --selection). Intervals resample items within unit kinds and domains, the same draws for every subject. Writes
results/<name>/scaled/: scaled.csv and report.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evotaste.analysis import data, deviation, health, stats
from evotaste.analysis.report import num, pci, render, write

METRICS = ("accuracy", "default_complied", "heldout_complied", "format_loss", "truncated", "tokens")


def deviation_of(run: data.Run, labels: list[str], selection, partial: bool, b: int, rng) -> tuple[dict, str | None]:
    """label -> (deviation, lo, hi) from the deviation selection that holds every label (the only such one unless
    named), and its name; empty when none does."""
    if selection is None:
        found = []
        for sel in sorted(p.parent for p in (run.dir / "deviation").glob("*/selection.json")):
            try:
                df = data.deviation_table(run, sel, partial)[0]
            except SystemExit:
                continue
            if set(labels) <= set(df["subject"]):
                found.append(sel)
        if not found:
            return {}, None
        if len(found) > 1:
            raise SystemExit(f"several deviation selections hold the scaled probes ({', '.join(p.name for p in found)}): "
                             "name one with --selection")
        selection = found[0]
    t = deviation.Table(run, data.deviation_table(run, selection, partial)[0])
    missing = sorted(set(labels) - set(t.labels))
    if missing:
        raise SystemExit(f"{selection}: no deviation of {missing}")
    point, draws = t.ratio(stats.ones(len(t.items)))[:, 0], t.ratio(t.draws(b, rng))
    return {s: (point[t.labels.index(s)], *stats.ci(draws[t.labels.index(s)])) for s in labels}, Path(selection).name


def analyse(run: data.Run, selection=None, partial: bool = False, b: int = 1000, rng=None) -> tuple[pd.DataFrame, str]:
    rng = rng or np.random.default_rng(0)
    if not run.scaled:
        raise SystemExit(f"{run.dir}/scaled: no finished scaled probe (python -m evotaste.evaluate scaled)")
    sources = {}
    for label, s in run.scaled.items():
        sources.setdefault(f"{s['lineage']}@{s['step']}", []).append((s["factor"], label))
    subjects = {label: (lin, step) for label, (lin, step) in health.probe_subjects(run).items()
                if run.base_weights(lin, step)}
    ref = list(subjects)
    for src, scaled in sources.items():
        subjects[src] = run.resolve(src)
        subjects |= {label: run.resolve(src) for _, label in scaled}
    grid = data.probe_grid(run, subjects)
    point, draws = health.evaluate(grid, rng, b)
    rp, rd = health.pooled(point, [grid.labels.index(a) for a in ref]), health.pooled(draws, [grid.labels.index(a) for a in ref])
    dev, sel = deviation_of(run, [s for src, sc in sources.items() for s in [src] + [lab for _, lab in sc]],
                            selection, partial, b, rng)
    rows = []
    for src, scaled in sorted(sources.items()):
        one = grid.labels.index(src)
        for factor, label in [(0.0, health.REFERENCE), (1.0, src)] + sorted(scaled):
            row = {"adapter": src, "factor": factor, "label": label}
            for m in METRICS:
                if label == health.REFERENCE:
                    v, d = rp[m][0], rd[m]
                else:
                    v, d = point[m][grid.labels.index(label), 0], draws[m][grid.labels.index(label)]
                row[m], (row[f"{m}_lo"], row[f"{m}_hi"]) = v, stats.ci(d)
                row[f"{m}_diff"] = v - point[m][one, 0]
                row[f"{m}_diff_lo"], row[f"{m}_diff_hi"] = stats.ci(d - draws[m][one])
            dv = (0.0, 0.0, 0.0) if label == health.REFERENCE else dev.get(label, (np.nan, np.nan, np.nan))
            row["deviation"], row["deviation_lo"], row["deviation_hi"] = dv
            rows.append(row)
    return pd.DataFrame(rows), sel


def report_lines(run: data.Run, table: pd.DataFrame, sel: str | None) -> list[str]:
    lines = [f"scaled probes {run.name}: factor 0 = the reference (probes of the base weights), 1 = the adapter's own "
             "probe, whose seeds the scaled probes share; percentages and tokens [95% bootstrap over items]; "
             f"deviation in nats per token from {'selection ' + sel if sel else 'no deviation selection'}"]
    for src, g in table.groupby("adapter", sort=False):
        lines += ["", src]
        lines += render(["factor", "accuracy", "complied", "held-out compl", "truncated", "tokens", "deviation",
                         "accuracy - x1"],
                        [[f"{r.factor:g}", pci(r.accuracy, r.accuracy_lo, r.accuracy_hi),
                          pci(r.default_complied, r.default_complied_lo, r.default_complied_hi),
                          pci(r.heldout_complied, r.heldout_complied_lo, r.heldout_complied_hi),
                          pci(r.truncated, r.truncated_lo, r.truncated_hi), f"{r.tokens:.0f} [{r.tokens_lo:.0f}, {r.tokens_hi:.0f}]",
                          num(r.deviation, r.deviation_lo, r.deviation_hi, 3),
                          "-" if r.factor == 1 else pci(r.accuracy_diff, r.accuracy_diff_lo, r.accuracy_diff_hi, sign=True)]
                         for r in g.itertuples()], 1)
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--selection", type=Path, help="the deviation selection holding the scaled probes")
    ap.add_argument("--partial", action="store_true", help="read a deviation selection still being scored")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap draws over items")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    run = data.Run(args.run_dir)
    table, sel = analyse(run, args.selection, args.partial, args.boot)
    write(args.out or Path("results") / run.name / "scaled", {"scaled": table}, report_lines(run, table, sel))


if __name__ == "__main__":
    with threadpool_limits(stats.THREADS):
        main()
