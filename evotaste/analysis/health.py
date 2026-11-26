"""Health of a run: accuracy overall, by domain, near against far units and by the base model's stratum; held-out
wrappers; format loss; truncation and length; answer channels; training reward and informative groups.

    python -m evotaste.analysis.health runs/<name> [--tag T] [--census results/census/<census>] [--out DIR]

Reads every committed probe of every lineage and the base model's probe answers, the training history, and with
--tag a selection evaluation. --census gives the base model's strata on selection items (its selection part).
Writes results/<name>/health/: probes.csv, groups.csv, history.csv, eval.csv (with --tag) and report.txt.
Definitions: docs/dev/ANALYSIS_NOTES.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evotaste.analysis import data, stats
from evotaste.analysis.report import pci, pct, render, write
from evotaste.verify.wrappers import HELDOUT_WRAPPERS

# Metrics whose difference from the base model is reported with an interval (the health criteria of the design).
VERSUS_BASE = ("accuracy", "near", "far", "mastered", "frontier", "heldout_complied", "format_loss", "truncated")


def _rate(x: np.ndarray, mask: np.ndarray, w: np.ndarray) -> np.ndarray:
    """sum over items and samples of w_i x where mask, divided by the same sum of w_i over mask: subject x draw."""
    num = np.where(mask, x, 0.0).sum(-1)
    den = mask.sum(-1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (num @ w.T) / (den @ w.T)


def metrics(grid: data.Grid, w: np.ndarray) -> dict[str, np.ndarray]:
    """Every health metric of every subject of the grid under item weights w: name -> subject x draw. Accuracies are
    means over items (not the engine's mean over units, which in the probe set gives each single-instance procedural
    family as much weight as a source's dozen items)."""
    items, x = grid.items, grid.x
    every = np.ones(x["reward"].shape, bool)

    def item(mask):
        return np.broadcast_to(np.asarray(mask)[None, :, None], x["reward"].shape)

    out = {"accuracy": _rate(x["reward"], every, w)}
    for d in sorted(items["domain"].unique()):
        out[f"domain_{d}"] = _rate(x["reward"], item(items["domain"] == d), w)
    for kind in data.UNIT_KINDS:
        if (items["unit"] == kind).any():
            out[kind] = _rate(x["reward"], item(items["unit"] == kind), w)
    if "stratum" in items and items["stratum"].notna().any():
        for s in ("mastered", "sweet", "frontier"):
            out[s] = _rate(x["reward"], item(items["stratum"] == s), w)
    heldout = item(items["wrapper"].isin(HELDOUT_WRAPPERS))
    out["heldout_complied"] = _rate(x["complied"], heldout, w)
    out["heldout_accuracy"] = _rate(x["reward"], heldout, w)
    out["default_complied"] = _rate(x["complied"], ~heldout, w)
    out["format_loss"] = _rate(x["correct_any"], every, w) - _rate(x["reward"], every, w)
    out["truncated"] = _rate(x["truncated"], every, w)
    out["tokens"] = _rate(x["tokens"], every, w)
    has_channel = np.isfinite(x["program"])
    out["program_share"] = _rate(np.nan_to_num(x["program"]), has_channel, w)
    out["direct_accuracy"] = _rate(x["reward"], has_channel & (x["program"] == 0), w)
    out["program_accuracy"] = _rate(x["reward"], has_channel & (x["program"] == 1), w)
    return out


def evaluate(grid: data.Grid, rng, b: int = stats.B) -> tuple[dict, dict]:
    """Metrics of every subject: point values and b bootstrap draws (items resampled within data.strata, the same
    draws for every subject, so that differences between subjects are paired)."""
    return metrics(grid, stats.ones(len(grid.items))), metrics(grid, stats.item_draws(data.strata(grid.items), b, rng))


REFERENCE = "reference"


def reference_rows(run: data.Run, labels: list[str], subject) -> list[int]:
    """Rows that sample the base model's weights (data.Run.base_weights); `subject` maps a label to (lineage, step)."""
    return [i for i, a in enumerate(labels) if run.base_weights(*subject(a))]


def pooled(values: dict, rows: list[int]) -> dict:
    """The mean over `rows` of every metric: one subject averaging several draws of the same weights."""
    return {k: v[rows].mean(0) for k, v in values.items()}


def table(grid: data.Grid, point: dict, draws: dict, ref: list[int]) -> pd.DataFrame:
    """Long table: subject, metric, value and interval; for every subject outside the reference, its difference from
    the reference (the mean of the rows `ref`, draws of the base weights) with a paired interval, which carries the
    noise of both sides; length percentiles; and a last subject REFERENCE with the number of draws it averages."""
    rows = []
    rp, rd = pooled(point, ref), pooled(draws, ref)
    for s, label in enumerate(grid.labels):
        for name in point:
            row = {"subject": label, "metric": name, "value": point[name][s, 0], "reference": s in ref,
                   **dict(zip(("lo", "hi"), stats.ci(draws[name][s])))}
            if ref and s not in ref and name in VERSUS_BASE:
                d = draws[name][s] - rd[name]
                row |= {"diff": point[name][s, 0] - rp[name][0], **dict(zip(("diff_lo", "diff_hi"), stats.ci(d)))}
            rows.append(row)
        tokens = grid.x["tokens"][s].ravel()
        rows += [{"subject": label, "metric": f"tokens_p{q}", "value": float(np.percentile(tokens, q)),
                  "reference": s in ref} for q in (50, 90, 99)]
    if ref:
        rows += [{"subject": REFERENCE, "metric": name, "value": rp[name][0], "draws": len(ref), "reference": False,
                  **dict(zip(("lo", "hi"), stats.ci(rd[name])))} for name in point]
    return pd.DataFrame(rows)


def group_table(run: data.Run, grid: data.Grid, subjects: dict, point: dict, draws: dict,
                ref: list[int]) -> pd.DataFrame:
    """Per probe step, update rule and seed level: the mean over members of (member - reference) for the VERSUS_BASE
    metrics, with intervals from the same item draws for every member. Reference draws and clones' first probes (a
    second draw of their parent's state) are left out."""
    rd, rp = pooled(draws, ref), pooled(point, ref)
    rows = []
    members = [(label, lin, step) for i, (label, (lin, step)) in enumerate(subjects.items())
               if lin is not None and i not in ref and not run.restart(lin, step)]
    for step in sorted({s for _, _, s in members}):
        for by in ("arm", "level"):
            for g in sorted({getattr(lin, by) for _, lin, s in members if s == step}):
                idx = [grid.labels.index(label) for label, lin, s in members if s == step and getattr(lin, by) == g]
                for name in VERSUS_BASE:
                    if name not in point:
                        continue
                    d = (draws[name][idx] - rd[name]).mean(0)
                    rows.append({"step": step, "by": by, "group": g, "members": len(idx), "metric": name,
                                 "diff": float((point[name][idx, 0] - rp[name][0]).mean()),
                                 **dict(zip(("lo", "hi"), stats.ci(d)))})
    return pd.DataFrame(rows)


def history_table(run: data.Run) -> pd.DataFrame:
    rows = []
    for lin in run.lineages.values():
        seed, amp = run.history(lin)
        for r in seed:
            rows.append({"lineage": lin.id, "phase": "seed", "step": r["step"], "method": r["method"],
                         "kl": r.get("kl"), "loss": r["loss"], "tokens": r["completion_tokens"],
                         "truncated": r["truncated"]})
        for r in amp:
            rows.append({"lineage": lin.id, "phase": "amp", "step": r["step"], "reward": r["reward"],
                         "return": r["return"], "groups": r["groups"], "prompts": r["prompts"],
                         "tokens": r["completion_tokens"], "complied": r["complied"], "truncated": r["truncated"],
                         "si_sequences": r.get("si_sequences")})
    return pd.DataFrame(rows)


def probe_subjects(run: data.Run) -> dict:
    subjects = {"base": (None, None)}
    for lin in run.lineages.values():
        for step in run.probe_steps(lin):
            subjects[f"{lin.id}@{step}"] = (lin, step)
    return subjects


def report_lines(run, subjects, probes, groups, hist, ev, tag, census) -> list[str]:
    lines = [f"health {run.name}: {len(subjects) - 1} lineage probes; percentages, [95% bootstrap interval over items]; "
             "differences are member - reference on the same items, the reference being the mean of the probes of the "
             "base weights: " + ", ".join(probes.loc[probes["reference"].fillna(False).astype(bool), "subject"].unique()),
             f"strata: {'census ' + census if census else 'not available (no --census): no retention or new-ground rows'}"]
    cell = {(r.subject, r.metric): r for r in probes.itertuples()}
    last = {}
    for label, (lin, step) in subjects.items():
        if lin is not None and step >= last.get(lin.id, (None, -1))[1]:
            last[lin.id] = (label, step)
    cols = [m for m in ("far", "near", "mastered", "frontier", "heldout_complied") if (REFERENCE, m) in cell]
    rows = []
    for lineage, (label, step) in last.items():
        lin = subjects[label][0]
        acc = cell[(label, "accuracy")]
        rows.append([label, lin.arm, lin.level, pct(acc.value)] + [pci(cell[(label, m)].diff, cell[(label, m)].diff_lo,
                                                                     cell[(label, m)].diff_hi, sign=True) for m in cols]
                    + [pct(cell[(label, "truncated")].value), pct(cell[(label, "program_share")].value)])
    lines += ["", "latest probe of every lineage (reference accuracy " + pct(cell[(REFERENCE, "accuracy")].value) + ")"]
    lines += render(["probe", "rule", "level", "acc"] + [f"{m} - ref" for m in cols] + ["trunc", "program"], rows)

    if len(groups):
        lines += ["", "mean over members of (member - reference), by update rule and by seed level"]
        g = groups[groups["metric"].isin(["far", "mastered", "frontier", "heldout_complied", "truncated"])]
        rows = []
        for (step, by, group), sub in g.groupby(["step", "by", "group"], sort=True):
            m = {r.metric: r for r in sub.itertuples()}
            rows.append([step, f"{by} {group}", sub["members"].iloc[0]]
                        + [pci(m[k].diff, m[k].lo, m[k].hi, sign=True) if k in m else "-"
                           for k in ("far", "mastered", "frontier", "heldout_complied", "truncated")])
        lines += render(["step", "group", "n", "far", "mastered", "frontier", "heldout compl", "truncated"], rows, 2)

    if len(hist) and (hist["phase"] == "amp").any():
        lines += ["", "training: mean reward and informative groups over the first and last ten committed steps"]
        rows = []
        for lineage, h in hist[hist["phase"] == "amp"].groupby("lineage", sort=False):
            first, end = h.head(10), h.tail(10)
            rows.append([lineage, len(h), pct(first["reward"].mean()), pct(end["reward"].mean()),
                         f"{first['groups'].mean():.1f}/{first['prompts'].mean():.0f}",
                         f"{end['groups'].mean():.1f}/{end['prompts'].mean():.0f}", f"{end['tokens'].mean():.0f}",
                         pct(end["truncated"].mean())])
        lines += render(["lineage", "steps", "reward first", "reward last", "groups first", "groups last",
                         "tokens last", "trunc last"], rows)

    if ev is not None:
        cell = {(r.subject, r.metric): r for r in ev.itertuples()}
        has_base = (REFERENCE, "accuracy") in cell
        refs = ev.loc[ev["reference"].fillna(False).astype(bool), "subject"].unique()
        lines += ["", f"selection evaluation {tag}" + (f" (reference: {', '.join(refs)})" if has_base else
                                                       " (the base weights are not evaluated: no differences)")]
        rows = []
        for label in dict.fromkeys(ev["subject"]):
            r = cell[(label, "accuracy")]
            rows.append([label, pci(r.value, r.lo, r.hi)] + [
                pci(cell[(label, m)].diff, cell[(label, m)].diff_lo, cell[(label, m)].diff_hi, sign=True)
                if has_base and (label, m) in cell and pd.notna(cell[(label, m)].diff) else "-" for m in cols]
                + [pct(cell[(label, "truncated")].value), pct(cell[(label, "program_share")].value)])
        lines += render(["adapter", "accuracy"] + [f"{m} - ref" for m in cols] + ["trunc", "program"], rows)
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--tag", help="a selection evaluation tag")
    ap.add_argument("--census", help="census output directory (results/census/<name>) for the base model's strata")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    run = data.Run(args.run_dir)
    strata = data.census_strata(args.census) if args.census else None
    subjects = probe_subjects(run)
    grid = data.probe_grid(run, subjects)
    data.add_strata(grid, strata)
    rng = np.random.default_rng(0)
    point, draws = evaluate(grid, rng)
    ref = reference_rows(run, grid.labels, subjects.get)
    probes = table(grid, point, draws, ref)
    probes.insert(1, "lineage", [s if s == REFERENCE else subjects[s][0].id if subjects[s][0] else "base"
                                 for s in probes["subject"]])
    probes.insert(2, "step", [subjects[s][1] if s in subjects else None for s in probes["subject"]])
    groups = group_table(run, grid, subjects, point, draws, ref)
    hist = history_table(run)
    tables = {"probes": probes, "groups": groups, "history": hist}
    ev = None
    if args.tag:
        egrid = data.load_eval(run, args.tag)
        data.add_strata(egrid, strata)
        ev = tables["eval"] = table(egrid, *evaluate(egrid, rng), reference_rows(run, egrid.labels, run.resolve))
    write(args.out or Path("results") / run.name / "health", tables,
          report_lines(run, subjects, probes, groups, hist, ev, args.tag, args.census))


if __name__ == "__main__":
    with threadpool_limits(stats.THREADS):
        main()
