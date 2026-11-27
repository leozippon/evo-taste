"""Arms against their reference on held-out selection items, paired by parent, when the clones of one parent were
evaluated under different tags.

    python -m evotaste.analysis.paired runs/<name> --tags e90,f90,r90 --reference neu,pln --arms rev,own
                                        [--census results/census/<c>] [--deviation DIR] [--boot 1000] [--out DIR]

The tags are merged on the items every label answered (data.load_eval) and read on the dataset units. A clone is a
label <parent>-<suffix>@<step> of a lineage that continues <parent>, at the latest step the tags hold for it. Each
parent needs exactly one reference clone (any suffix of --reference: the neutral arm is `neu` in pilot E and `pln`
in pilot F); each arm of --arms is compared with it where the parent has that arm. Per arm and metric (accuracy
overall, near, far, by the base model's stratum with --census, by domain; completion tokens and the share cut off;
with --deviation the deviation from the base model): arm - reference per parent with an interval over items; their
mean over parents with an interval over items alone and one that resamples parents as well as items; the parents
agreeing in sign and the exact two-sided sign-flip p over parents; and the mean among unseeded parents (the fragile
lineages trained from the bare base model) and among distilled ones apart. Finally, whether the size of an arm's
accuracy difference goes with the size of its change in deviation across parents: Spearman's rho with an exact (or,
beyond 100,000 orderings, Monte Carlo) permutation p. Writes results/<name>/paired/: pairs.csv, summary.csv,
association.csv and report.txt.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from threadpoolctl import threadpool_limits

from evotaste.analysis import data, deviation, health, stats
from evotaste.analysis.report import num, pci, render, write

METRICS = ("accuracy", "near", "far", "frontier", "sweet", "mastered", "domain_math", "domain_code", "domain_logic",
           "tokens", "truncated")


def clones(run: data.Run, labels: list[str], suffixes: set[str]) -> dict[str, dict[str, str]]:
    """parent -> suffix -> label of the clone <parent>-<suffix> at the latest step the labels hold for it."""
    out, at = {}, {}
    for label in labels:
        if label == "base":
            continue
        lin, step = run.resolve(label)
        if lin.parent is None or not lin.id.startswith(lin.parent[0] + "-"):
            continue
        p, s = lin.parent[0], lin.id[len(lin.parent[0]) + 1:]
        if s in suffixes and step > at.get(lin.id, -1):
            out.setdefault(p, {})[s], at[lin.id] = label, step
    return out


def sign_flip(d: np.ndarray) -> float:
    return stats.sign_flip_p(d) if len(d) else np.nan


def rank_association(x: np.ndarray, y: np.ndarray, rng, limit: int = 100_000) -> tuple[float, float, int]:
    """Spearman's rho of x and y and its two-sided permutation p: exact over every ordering of y when there are at
    most `limit`, else over `limit` random ones; and the number of orderings used."""
    rx, ry = sps.rankdata(x), sps.rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    perms = (np.array(list(itertools.permutations(range(n)))) if math.factorial(n) <= limit
             else np.array([rng.permutation(n) for _ in range(limit)]))
    rp = ry[perms]
    rp = rp - rp.mean(1, keepdims=True)
    null = (rp @ (rx - rx.mean())) / (np.linalg.norm(rp, axis=1) * np.linalg.norm(rx - rx.mean()))
    return rho, float(np.mean(np.abs(null) >= abs(rho) - 1e-12)), len(perms)


def analyse(run: data.Run, tags: str, references: list[str], arms: list[str], census=None, selection=None,
            b: int = 1000, rng=None) -> dict:
    rng = rng or np.random.default_rng(0)
    grid = data.load_eval(run, tags)
    data.add_strata(grid, data.census_strata(census) if census else None)
    real = grid.real()
    by = clones(run, real.labels, set(references) | set(arms))
    for p, d in by.items():
        refs = [s for s in references if s in d]
        if len(refs) != 1:
            raise SystemExit(f"{p}: {len(refs)} reference clones among {references} in tags {tags}")
        d["_ref"] = d[refs[0]]
    parents = sorted(p for p in by if any(a in by[p] for a in arms))
    if not parents:
        raise SystemExit(f"no parent with a clone of the arms {arms} and a reference in tags {tags}")
    point, draws = health.evaluate(real, rng, b)
    point = {m: point[m][:, 0] for m in METRICS if m in point}
    draws = {m: draws[m] for m in point}
    index = {label: i for i, label in enumerate(real.labels)}
    if selection is not None:  # deviation of each clone's answers at the step evaluated, from the probe of that step
        t = deviation.Table(run, data.deviation_table(run, selection)[0])
        dp, dd = t.ratio(stats.ones(len(t.items)))[:, 0], t.ratio(t.draws(b, rng))
        missing = sorted({lab for p in parents for lab in by[p].values() if lab not in t.labels})
        if missing:
            raise SystemExit(f"{selection}: no deviation of {missing}")
        point["deviation"] = np.array([dp[t.labels.index(a)] if a in t.labels else np.nan for a in real.labels])
        draws["deviation"] = np.array([dd[t.labels.index(a)] if a in t.labels else np.full(b, np.nan)
                                       for a in real.labels])
    kind = {p: "unseeded" if run.root(run.lineages[p]).seeding == "none" else "distilled" for p in parents}
    pairs, summary, assoc = [], [], []
    for arm in arms:
        ps = [p for p in parents if arm in by[p]]
        if not ps:
            continue
        ia, ir = [index[by[p][arm]] for p in ps], [index[by[p]["_ref"]] for p in ps]
        diff = {}
        for m in point:
            d, dd_ = point[m][ia] - point[m][ir], draws[m][ia] - draws[m][ir]  # parents, parents x draws
            diff[m] = d
            for j, p in enumerate(ps):
                pairs.append({"arm": arm, "parent": p, "kind": kind[p], "metric": m, "arm_label": by[p][arm],
                              "reference": by[p]["_ref"], "diff": d[j], **dict(zip(("lo", "hi"), stats.ci(dd_[j])))})
            two = np.array([dd_[rng.integers(0, len(ps), len(ps)), k].mean() for k in range(dd_.shape[1])])
            row = {"arm": arm, "metric": m, "parents": len(ps), "mean": float(d.mean()),
                   **dict(zip(("items_lo", "items_hi"), stats.ci(dd_.mean(0)))),
                   **dict(zip(("lo", "hi"), stats.ci(two))),
                   "agree": int(max((d > 0).sum(), (d < 0).sum())), "p_sign_flip": sign_flip(d)}
            for k in ("unseeded", "distilled"):
                sel = [j for j, p in enumerate(ps) if kind[p] == k]
                row[f"mean_{k}"], row[f"n_{k}"] = (float(d[sel].mean()) if sel else np.nan), len(sel)
            summary.append(row)
        if "deviation" in diff and len(ps) >= 3:
            rho, p_, n_ = rank_association(diff["accuracy"], diff["deviation"], rng)
            assoc.append({"arm": arm, "parents": len(ps), "rho": rho, "p_perm": p_, "orderings": n_})
    return {"pairs": pd.DataFrame(pairs), "summary": pd.DataFrame(summary), "association": pd.DataFrame(assoc),
            "_items": len(real.items), "_dropped": grid.dropped, "_kind": kind, "_by": by}


def report_lines(run, tags, res) -> list[str]:
    s, pr = res["summary"], res["pairs"]
    lines = [f"paired {run.name}, tags {tags}: {res['_items']} dataset items every label answered"
             + (f" ({res['_dropped']} items of some labels only left out)" if res["_dropped"] else "")
             + "; arm - reference of the same parent; accuracy in points, tokens in tokens, deviation in nats per "
               "token; [95% over parents and items] (over items alone); agree: parents of the majority sign; p: exact "
               "two-sided sign flip over parents"]
    for arm, g in s.groupby("arm", sort=False):
        refs = sorted({r for r in pr.loc[pr["arm"] == arm, "reference"]})
        lines += ["", f"{arm} against {', '.join(sorted({x.split('@')[0].rsplit('-', 1)[1] for x in refs}))} "
                      f"({g['parents'].iloc[0]} parents)"]
        rows = []
        for r in g.itertuples():
            f = (lambda v: num(v, digits=0)) if r.metric == "tokens" else (
                (lambda v: num(v, digits=3)) if r.metric == "deviation" else (lambda v: num(100 * v, digits=1)))
            rows.append([r.metric, f(r.mean) + f" [{f(r.lo)}, {f(r.hi)}]", f"({f(r.items_lo)}, {f(r.items_hi)})",
                         f"{r.agree}/{r.parents}", f"{r.p_sign_flip:.3f}", f(r.mean_unseeded) + f" ({r.n_unseeded})",
                         f(r.mean_distilled) + f" ({r.n_distilled})"])
        lines += render(["metric", "mean [95% parents+items]", "(items)", "agree", "p", "unseeded (n)",
                         "distilled (n)"], rows)
        per = pr[pr["arm"] == arm]
        mets = [m for m in ("accuracy", "near", "far", "frontier", "sweet", "mastered", "tokens", "truncated",
                            "deviation") if (per["metric"] == m).any()]
        cell = {(r.parent, r.metric): r for r in per.itertuples()}
        lines += ["  per parent:"]
        lines += ["  " + x for x in render(["parent", "kind"] + mets, [
            [p, res["_kind"][p]] + [(pci(cell[(p, m)].diff, cell[(p, m)].lo, cell[(p, m)].hi, sign=True)
                                     if m == "accuracy" else num(cell[(p, m)].diff, digits=0) if m == "tokens"
                                     else num(cell[(p, m)].diff, digits=3) if m == "deviation"
                                     else num(100 * cell[(p, m)].diff, digits=1)) for m in mets]
            for p in dict.fromkeys(per["parent"])], 2)]
    for r in res["association"].itertuples():
        lines += ["", f"{r.arm}: accuracy difference against change in deviation across {r.parents} parents: Spearman "
                      f"rho {r.rho:+.2f}, permutation p {r.p_perm:.3f} ({r.orderings} orderings)"]
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--tags", required=True, help="comma-separated selection evaluation tags, merged on common items")
    ap.add_argument("--reference", required=True, help="comma-separated suffixes of the reference arm (one per parent)")
    ap.add_argument("--arms", required=True, help="comma-separated suffixes of the arms compared with it")
    ap.add_argument("--census", help="census output directory for the base model's strata")
    ap.add_argument("--deviation", type=Path, help="a deviation selection holding the clones at the step evaluated")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap draws over items (and parents)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    run = data.Run(args.run_dir)
    res = analyse(run, args.tags, args.reference.split(","), args.arms.split(","), args.census, args.deviation,
                  args.boot)
    write(args.out or Path("results") / run.name / "paired",
          {k: v for k, v in res.items() if isinstance(v, pd.DataFrame)}, report_lines(run, args.tags, res))


if __name__ == "__main__":
    with threadpool_limits(stats.THREADS):
        main()
