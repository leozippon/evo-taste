"""Harvest simulation inside the selection suite: would selecting the best member on some selection units pay off on
others? No blind item is involved.

    python -m evotaste.analysis.harvest runs/<name> --tag T [--step S] [--splits 500] [--out DIR]

On the dataset units of a selection evaluation (the procedural format probe left out), the members evaluated at one
step (default: the step with most labels) are repeatedly split into a select half and a test half, by whole units
within each domain (selecting on some sources, testing on others) and by items within each unit (the same sources:
an upper bound). Per split: members ranked by accuracy on the select half; for the top 1 and the top 3, the lead over
the population mean on the select half and on the test half, and the retained fraction test lead / select lead; the
Spearman correlation of members' scores between halves; and the expected test-half lead of the best of n random
members (exact over subsets). All members and each update rule. Also the between-member variance of accuracy beyond
sampling noise and its share of the observed variance (reliability). Intervals: over splits (items fixed) and over
splits with an item bootstrap draw per split. Writes results/<name>/harvest/: splits.csv, best_of_n.csv,
reliability.csv and report.txt.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from threadpoolctl import threadpool_limits

from evotaste.analysis import data, divergence, stats
from evotaste.analysis.report import num, render, write

SIZES = (1, 2, 4, 8, 14, 28)
LIMITS = ("A held-out half of the selection units is closer to the selected half than the blind vault is: the "
          "vault's unseen units were drawn from the other half of each domain's units, its held-out training-source "
          "items are new items of training sources, and nobody, members or analyst, has looked at either. The "
          "retained fraction here is therefore an optimistic bound for the vault, and the item split is more "
          "optimistic still. The simulation also takes the evaluated members as the population: it says nothing "
          "about members other seeds would have produced.")


def halves(items: pd.DataFrame, mode: str, rng) -> np.ndarray:
    """A boolean mask of the select half: half of the units of each domain (mode units), or half of the items of
    each unit (mode items)."""
    sel = np.zeros(len(items), bool)
    if mode == "units":
        for d in items["domain"].unique():
            units = sorted(items.loc[items["domain"] == d, "family"].unique())
            chosen = set(rng.choice(units, len(units) // 2, replace=False)) if len(units) > 1 else set()
            sel |= (items["domain"] == d).to_numpy() & items["family"].isin(chosen).to_numpy()
    else:
        for f in items["family"].unique():
            idx = np.flatnonzero((items["family"] == f).to_numpy())
            sel[rng.choice(idx, len(idx) // 2, replace=False)] = True
    return sel


def best_of(x: np.ndarray, y: np.ndarray, n: int) -> float:
    """Expected test score of the member ranked first on x among n members drawn at random without replacement,
    exact: the member of rank r is the best of the draw when it is drawn and no better one is."""
    m = len(x)
    order = np.argsort(-x, kind="stable")
    total = math.comb(m, n)
    return float(sum(math.comb(m - r - 1, n - 1) / total * y[order[r]] for r in range(m - n + 1)))


def split_stats(x: np.ndarray, y: np.ndarray, rng) -> dict:
    x = x + rng.uniform(0, 1e-9, len(x))  # random order among exact ties
    order = np.argsort(-x)
    out = {"spearman": float(sps.spearmanr(x, y)[0])}
    for k in (1, 3):
        top = order[:k]
        sel_lead, test_lead = x[top].mean() - x.mean(), y[top].mean() - y.mean()
        out |= {f"select_lead_top{k}": sel_lead, f"test_lead_top{k}": test_lead,
                f"retained_top{k}": test_lead / sel_lead if sel_lead > 0 else np.nan}
    for n in SIZES:
        if n <= len(x):
            out[f"best_of_{n}"] = best_of(x, y, n) - y.mean()
    return out


def simulate(y: np.ndarray, items: pd.DataFrame, sets: dict, splits: int, rng) -> pd.DataFrame:
    """Per split mode, split, member set and item weighting (fixed items, or an item bootstrap draw): the statistics
    of split_stats."""
    rows = []
    strata = data.strata(items)
    for mode in ("units", "items"):
        for s in range(splits):
            sel = halves(items, mode, rng)
            w_boot = stats.item_draws(strata, 1, rng)[0]
            for weighting, w in (("fixed", np.ones(len(items))), ("bootstrap", w_boot)):
                ws, wt = w * sel, w * ~sel
                xs, ys = y @ ws / ws.sum(), y @ wt / wt.sum()
                for g, idx in sets.items():
                    rows.append({"mode": mode, "split": s, "weighting": weighting, "members": g,
                                 **split_stats(xs[idx], ys[idx], rng)})
    return pd.DataFrame(rows)


def reliability(y: np.ndarray, v: np.ndarray, items: pd.DataFrame, sets: dict, rng) -> pd.DataFrame:
    """Between-member variance of accuracy beyond sampling noise, the observed variance, and their ratio."""
    w = stats.item_draws(data.strata(items), stats.B, rng)
    rows = []
    for g, idx in sets.items():
        t0, n0 = stats.member_means(y[idx], v[idx], stats.ones(y.shape[1]))
        tb, nb = stats.member_means(y[idx], v[idx], w)
        obs, obs_b = t0.var(0, ddof=1)[0], tb.var(0, ddof=1)
        between, between_b = stats.between(t0, n0)[0], stats.between(tb, nb)
        rows.append({"members": g, "m": len(idx), "mean": float(t0.mean()), "observed_var": obs,
                     "noise_var": float(n0.mean()), "between_var": between,
                     **dict(zip(("between_lo", "between_hi"), stats.ci(between_b))),
                     "between_sd": stats.sd(between), "reliability": between / obs if obs > 0 else np.nan,
                     **dict(zip(("reliability_lo", "reliability_hi"), stats.ci(between_b / obs_b)))})
    return pd.DataFrame(rows)


def summarise(sim: pd.DataFrame) -> pd.DataFrame:
    """Per mode and member set: the mean over splits with fixed items, and 95% intervals over splits (fixed items)
    and over splits with an item draw each."""
    cols = [c for c in sim.columns if c not in ("mode", "split", "weighting", "members")]
    rows = []
    for (mode, g), sub in sim.groupby(["mode", "members"], sort=False):
        fixed, boot = sub[sub["weighting"] == "fixed"], sub[sub["weighting"] == "bootstrap"]
        for c in cols:
            if fixed[c].isna().all():
                continue
            rows.append({"mode": mode, "members": g, "statistic": c,
                         "value": float(fixed[c].median() if c.startswith("retained") else fixed[c].mean()),
                         **dict(zip(("splits_lo", "splits_hi"), stats.ci(fixed[c]))),
                         **dict(zip(("lo", "hi"), stats.ci(boot[c])))})
    return pd.DataFrame(rows)


def report_lines(run, tag, step, members, grid, summary, rel, splits) -> list[str]:
    cell = {(r.mode, r.members, r.statistic): r for r in summary.itertuples()}
    lines = [f"harvest simulation {run.name}, selection {tag}: {len(members)} members at step {step}, "
             f"{len(grid.items)} items of {grid.items['family'].nunique()} dataset units x {grid.samples} samples "
             f"(procedural format probe left out); {splits} splits per mode",
             "leads in accuracy points over the population mean; [95% over splits and items], (95% over splits with "
             "the items fixed); retained = test lead / select lead (median over splits)"]

    def show(mode, g, stat, scale=100, digits=1):
        r = cell.get((mode, g, stat))
        if r is None:
            return "-"
        return f"{scale * r.value:+.{digits}f} [{scale * r.lo:+.{digits}f}, {scale * r.hi:+.{digits}f}]"

    groups = list(dict.fromkeys(summary["members"]))
    for mode, title in (("units", "split by whole units within domains (select on some sources, test on others)"),
                        ("items", "split by items within units (the same sources: an upper bound)")):
        lines += ["", title]
        lines += render(["members", "top-1 select", "top-1 test", "retained", "top-3 select", "top-3 test",
                         "retained", "Spearman"],
                        [[g, show(mode, g, "select_lead_top1"), show(mode, g, "test_lead_top1"),
                          show(mode, g, "retained_top1", 1, 2), show(mode, g, "select_lead_top3"),
                          show(mode, g, "test_lead_top3"), show(mode, g, "retained_top3", 1, 2),
                          show(mode, g, "spearman", 1, 2)] for g in groups])
        lines += ["  expected test lead of the best of n random members (points): " + "; ".join(
            f"{g}: " + ", ".join(f"n={n} {100 * cell[(mode, g, f'best_of_{n}')].value:+.2f}" for n in SIZES
                                 if (mode, g, f"best_of_{n}") in cell) for g in groups)]
    lines += ["", "between-member variance of accuracy beyond sampling noise (all dataset items)"]
    lines += render(["members", "m", "accuracy", "observed SD", "noise SD", "between SD [95%]", "reliability [95%]"],
                    [[r.members, r.m, f"{100 * r.mean:.1f}", f"{100 * math.sqrt(r.observed_var):.2f}",
                      f"{100 * math.sqrt(r.noise_var):.2f}",
                      num(100 * r.between_sd, 100 * stats.sd(r.between_lo), 100 * stats.sd(r.between_hi)),
                      num(r.reliability, r.reliability_lo, r.reliability_hi)] for r in rel.itertuples()])
    return lines + ["", "what this cannot show: " + LIMITS]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--tag", required=True, help="selection evaluation tag")
    ap.add_argument("--step", type=int, help="members evaluated at this step (default: the step with most labels)")
    ap.add_argument("--splits", type=int, default=500)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    run = data.Run(args.run_dir)
    grid = data.load_eval(run, args.tag).real()
    members = divergence.pick_members(run, grid, args.step, None)
    step = run.resolve(members[0])[1]
    y, v = stats.cells(grid.take(members).x["reward"])
    arm = [run.resolve(m)[0].arm for m in members]
    sets = {"all": list(range(len(members)))} | {a: [i for i, x in enumerate(arm) if x == a] for a in sorted(set(arm))}
    sets = {g: idx for g, idx in sets.items() if len(idx) >= 3 and (g == "all" or len(sets) > 2)}
    rng = np.random.default_rng(0)
    sim = simulate(y, grid.items, sets, args.splits, rng)
    summary, rel = summarise(sim), reliability(y, v, grid.items, sets, rng)
    best = summary[summary["statistic"].str.startswith("best_of")]
    write(args.out or Path("results") / run.name / "harvest",
          {"splits": sim, "summary": summary, "best_of_n": best, "reliability": rel},
          report_lines(run, args.tag, step, members, grid, summary, rel, args.splits))


if __name__ == "__main__":
    with threadpool_limits(stats.THREADS):
        main()
