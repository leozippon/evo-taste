"""Heritability: whether members that share a trained state (clones of one parent) stay more alike than unrelated
members, in scalar traits and in which problems they solve.

    python -m evotaste.analysis.heritability runs/<name> --tag T [--null-tag T0] [--table <context-score table>]
                                              [--out DIR]

A family is the lineages of one root under one update rule: a member and its clones (members that continue a
checkpoint of it). Unrelated pairs are members of different roots under the same rule at the same step. For every
step at which a family has two or more evaluated labels: the intraclass correlation of each scalar trait (accuracy,
by domain, counted and layout traits, and the specific context index when scored) with the update rule as a fixed
effect, the regression of the family mean on the parent at the clone point, and the functional distance of clone
pairs against unrelated pairs. Intervals resample families within each rule and items within units; p-values
relabel members across family slots within each rule. Writes results/<name>/heritability/: families.csv, icc.csv,
regression.csv, distance.csv, pairs.csv and report.txt.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evotaste.analysis import data, divergence, health, stats, traits, washout
from evotaste.analysis.report import num, render, write

PP2 = 1e4


def members_at(run: data.Run, grid: data.Grid) -> dict[int, list[tuple[str, data.Lineage]]]:
    at = collections.defaultdict(list)
    for label in grid.labels:
        lin, step = run.resolve(label)
        if lin is not None:
            at[step].append((label, lin))
    return at


def family_of(run: data.Run, lin: data.Lineage) -> str:
    return f"{washout.root(run, lin).id}/{lin.arm}"


def scalar_traits(grid: data.Grid, w: np.ndarray) -> dict[str, np.ndarray]:
    """Per label (row) and item draw: accuracy and accuracy by domain (means over units, as the engine's), program
    share, mean length, and the mean of every other counted and layout trait over responses that have it."""
    m = health.metrics(grid, w)
    out = {k: m[k] for k in m if k == "accuracy" or k.startswith("domain_")}
    out["program"], out["tokens"] = m["program_share"], m["tokens"]
    for t in traits.COUNTED + traits.LAYOUT:
        if t in ("tokens", "program"):
            continue
        x = grid.x[t]
        out[t] = health._rate(np.nan_to_num(x), np.isfinite(x), w)
    return out


def icc_rows(step, labels, fam, arm, values: dict, draws: dict, specific: dict, rng, b: int) -> list[dict]:
    """ICC of every trait over the families with two or more labels; arm-pooled and per arm."""
    rows = []
    labels = np.array(labels)
    keep = np.array([np.sum(fam == f) >= 2 for f in fam])
    sets = {"all": keep} | {a: keep & (arm == a) for a in np.unique(arm)}
    for g, sel in sets.items():
        f, a = fam[sel], arm[sel]
        if len(np.unique(f)) < 2 or (g == "all" and len(sets) == 2):
            continue
        names = [k for k in values if np.isfinite(values[k][sel, 0]).all()]
        y, yd = np.stack([values[k][sel, 0] for k in names], 1), np.stack([draws[k][sel] for k in names], 1)
        if all(lab in specific for lab in labels[sel]):  # no item draws: only families are resampled
            names.append("specific")
            spec = np.array([specific[lab] for lab in labels[sel]])
            y, yd = np.c_[y, spec], np.concatenate([yd, np.tile(spec[:, None, None], (1, 1, b))], 1)
        res = stats.icc(y, f, a)
        boot = np.full((b, len(names)), np.nan)
        for j, drawn in enumerate(stats.family_draws(f, a, b, rng)):
            members, slot, _ = stats.drawn_members(f, drawn)
            boot[j] = stats.icc(yd[members, :, j], slot, a[members])["icc"]
        perm, distinct = stats.relabellings(f, a, rng)
        for t, name in enumerate(names):
            rows.append({"step": step, "rule": g, "trait": name, "families": len(np.unique(f)), "members": len(f),
                         **{k: (v[t] if isinstance(v, np.ndarray) else v) for k, v in res.items()},
                         **dict(zip(("icc_lo", "icc_hi"), stats.ci(boot[:, t]))),
                         "p_perm": stats.perm_p(stats.between_ss(y[:, t], f)[0], stats.between_ss(y[:, t], perm)),
                         "relabellings": distinct})
    return rows


def distance_rows(step, labels, fam, arm, d_point, d_draws, null_mean, rng, b, estimator) -> tuple[list, list]:
    """Mean distance of clone pairs and of unrelated pairs (same rule), their difference with intervals resampling
    families and items, and the permutation p-value of the difference over family relabellings."""
    rows, pairs = [], []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if arm[i] == arm[j]:
                pairs.append({"step": step, "estimator": estimator, "a": labels[i], "b": labels[j], "rule": arm[i],
                              "clones": fam[i] == fam[j], "d2": d_point[i, j]})
    sets = {"all": np.ones(n, bool)} | {a: arm == a for a in np.unique(arm)}
    for g, sel in sets.items():
        if g == "all" and len(sets) == 2:
            continue
        f, a, idx = fam[sel], arm[sel], np.flatnonzero(sel)
        dp = d_point[np.ix_(idx, idx)]
        clone, unrelated = means(dp, f, a)
        if not (np.isfinite(clone) and np.isfinite(unrelated)):
            continue
        perm, distinct = stats.relabellings(f, a, rng)
        pc = np.array([np.subtract(*means(dp, p, a)[::-1]) for p in perm])
        draws = stats.family_draws(f, a, b, rng)
        boot = np.array([boot_means(d_draws[k][np.ix_(idx, idx)], f, a, draws[k]) for k in range(b)])
        diff = boot[:, 1] - boot[:, 0]
        rows.append({"step": step, "estimator": estimator, "rule": g, "families": len(np.unique(f)), "members": len(f),
                     "clone": clone, "clone_lo": stats.ci(boot[:, 0])[0], "clone_hi": stats.ci(boot[:, 0])[1],
                     "unrelated": unrelated, "unrelated_lo": stats.ci(boot[:, 1])[0],
                     "unrelated_hi": stats.ci(boot[:, 1])[1], "difference": unrelated - clone,
                     **dict(zip(("diff_lo", "diff_hi"), stats.ci(diff))), "ratio": clone / unrelated,
                     **dict(zip(("ratio_lo", "ratio_hi"), stats.ci(boot[:, 0] / boot[:, 1]))),
                     "p_perm": stats.perm_p(unrelated - clone, pc), "relabellings": distinct, "null": null_mean})
    return rows, pairs


def means(d: np.ndarray, fam: np.ndarray, arm: np.ndarray) -> tuple[float, float]:
    """Mean distance over clone pairs (same family) and over unrelated pairs (different families, same rule)."""
    same = (fam[:, None] == fam[None, :]) & ~np.eye(len(fam), dtype=bool)
    other = (fam[:, None] != fam[None, :]) & (arm[:, None] == arm[None, :])
    return (float(d[same].mean()) if same.any() else np.nan, float(d[other].mean()) if other.any() else np.nan)


def boot_means(d: np.ndarray, fam: np.ndarray, arm: np.ndarray, drawn: np.ndarray) -> tuple[float, float]:
    """Clone and unrelated means on a draw of families; pairs between two copies of one family are left out."""
    sel, slot, orig = stats.drawn_members(fam, drawn)
    dd, a = d[np.ix_(sel, sel)], arm[sel]
    same = (slot[:, None] == slot[None, :]) & ~np.eye(len(sel), dtype=bool)
    other = (orig[:, None] != orig[None, :]) & (a[:, None] == a[None, :])
    return (float(dd[same].mean()) if same.any() else np.nan, float(dd[other].mean()) if other.any() else np.nan)


def slope(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    sxx = ((x - x.mean()) ** 2).sum()
    return float(((x - x.mean()) * (y - y.mean())).sum() / sxx) if sxx > 0 else np.nan


def analyse(run: data.Run, grid: data.Grid, null: data.Grid | None, rng, table: Path | None = None,
            b: int = stats.B) -> dict:
    spec = {}
    if table is not None or len(list((run.dir / "context_score").glob("*/table.csv"))) == 1:
        s = washout.analyse(run, rng, table)["specific"]
        if len(s):
            s = s[s["domain"] == washout.DOMAINS_ALL]
            spec = dict(zip(s["subject"], s["specific"]))
    null_mean = {}
    if null is not None:
        nl = divergence.null_pairs(null, lambda it: stats.item_draws(it["family"], b, rng))
        null_mean = dict(zip(nl.loc[nl["adapter"] == "mean", "estimator"], nl.loc[nl["adapter"] == "mean", "d2"]))
    icc, reg, dist, pairs, families = [], [], [], [], []
    w = stats.item_draws(grid.items["family"], b, rng)
    values_all, draws_all = scalar_traits(grid, stats.ones(len(grid.items))), scalar_traits(grid, w)
    for step, group in sorted(members_at(run, grid).items()):
        labels = [lab for lab, _ in group]
        fam = np.array([family_of(run, lin) for _, lin in group])
        arm = np.array([lin.arm for _, lin in group])
        families += [{"step": step, "family": f, "members": " ".join(np.array(labels)[fam == f])}
                     for f in dict.fromkeys(fam)]
        if not any(np.sum(fam == f) >= 2 for f in fam):
            continue
        rows_of = [grid.labels.index(lab) for lab in labels]
        values = {k: v[rows_of] for k, v in values_all.items()}
        draws = {k: v[rows_of] for k, v in draws_all.items()}
        icc += icc_rows(step, labels, fam, arm, values, draws, spec, rng, b)
        reg += family_regression(run, grid, labels, fam, values, draws, values_all, draws_all, step, rng, b)
        sub = grid.take(labels)
        if grid.samples >= 2:
            y, v = stats.cells(sub.x["reward"])
            for estimator, classes in (("adjusted", stats.difficulty_classes(y)), ("plain", stats.one_class(y))):
                dp = stats.distances(y, v, classes, stats.ones(y.shape[1]))[0]
                dd = stats.distances(y, v, classes, w)
                r, p = distance_rows(step, labels, fam, arm, dp, dd, null_mean.get(estimator, np.nan), rng, b,
                                     estimator)
                dist += r
                pairs += p
    return {"families": pd.DataFrame(families), "icc": pd.DataFrame(icc), "regression": pd.DataFrame(reg),
            "distance": pd.DataFrame(dist), "pairs": pd.DataFrame(pairs)}


def family_regression(run, grid, labels, fam, values, draws, values_all, draws_all, step, rng, b) -> list[dict]:
    """Slope of the family mean at `step` on the parent's value at the clone point, over families whose parent
    checkpoint at that point is evaluated under the tag (label <parent>@<clone step>); the interval resamples
    families and items."""
    points = {}
    for lab, f in zip(labels, fam):
        lin = run.resolve(lab)[0]
        if lin.parent:
            points.setdefault(f, set()).add(f"{lin.parent[0]}@{lin.parent[1]}")
    aliases = grid.aliases or {}
    usable = {f: aliases.get(next(iter(p)), next(iter(p))) for f, p in points.items() if len(p) == 1}
    usable = {f: p for f, p in usable.items() if p in grid.labels}
    if len(usable) < 3:
        return []
    fams = sorted(usable)
    pi = [grid.labels.index(usable[f]) for f in fams]
    rows = []
    for name in values:
        x, xd = values_all[name][pi, 0], draws_all[name][pi]
        y = np.array([values[name][fam == f, 0].mean() for f in fams])
        yd = np.array([draws[name][fam == f].mean(0) for f in fams])
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            continue
        boot = [slope(xd[j, k], yd[j, k]) for k in range(b) for j in [rng.integers(0, len(fams), len(fams))]]
        rows.append({"step": step, "trait": name, "families": len(fams), "slope": slope(x, y),
                     **dict(zip(("lo", "hi"), stats.ci(boot)))})
    return rows


def report_lines(run, tag, grid, res, null_tag) -> list[str]:
    lines = [f"heritability {run.name}, selection {tag} ({grid.samples} samples per item): families are a root "
             "lineage's members under one update rule; [95% interval resampling families within rules and items "
             "within units]; p: permutation over family labels (relabellings = distinct partitions)"]
    fams = res["families"]
    multi = fams[fams["members"].str.contains(" ")] if len(fams) else fams
    lines += ["", "families with two or more evaluated members: " + ("; ".join(
        f"step {r.step} {r.family}: {r.members}" for r in multi.itertuples()) or "none")]
    icc = res["icc"]
    lines += [""] + (["intraclass correlation (one-way, rule as fixed effect)"] if len(icc) else [
        "intraclass correlation: needs two families with two or more members at one step"])
    lines += render(["step", "rule", "trait", "fam", "n", "ICC [95%]", "p", "relab"],
                    [[r.step, r.rule, r.trait, r.families, r.members, num(r.icc, r.icc_lo, r.icc_hi), f"{r.p_perm:.3f}",
                      r.relabellings] for r in icc.itertuples()], 3) if len(icc) else []
    reg = res["regression"]
    lines += [""] + (["regression of the family mean on the parent at the clone point"]
                     + render(["step", "trait", "families", "slope [95%]"],
                              [[r.step, r.trait, r.families, num(r.slope, r.lo, r.hi)] for r in reg.itertuples()], 2)
                     if len(reg) else ["regression on the parent: needs three families whose parent's clone point is "
                                       "evaluated under this tag (label <parent>@<clone step>)"])
    dist = res["distance"]
    if len(dist):
        lines += ["", f"functional distance, pp^2: clone pairs against unrelated pairs (null = two halves of one "
                      f"adapter, {null_tag or 'not available: --null-tag'})"]
        lines += render(["step", "estimator", "rule", "fam", "clone [95%]", "unrelated [95%]", "unrel - clone [95%]",
                         "ratio [95%]", "p", "relab", "null"],
                        [[r.step, r.estimator, r.rule, r.families, num(PP2 * r.clone, PP2 * r.clone_lo, PP2 * r.clone_hi, 1),
                          num(PP2 * r.unrelated, PP2 * r.unrelated_lo, PP2 * r.unrelated_hi, 1),
                          num(PP2 * r.difference, PP2 * r.diff_lo, PP2 * r.diff_hi, 1),
                          num(r.ratio, r.ratio_lo, r.ratio_hi), f"{r.p_perm:.3f}", r.relabellings,
                          num(PP2 * r.null, digits=2)] for r in dist.itertuples()], 3)
    elif grid.samples < 2:
        lines += ["", "functional distance: needs an evaluation with two or more samples per item"]
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--null-tag", help="evaluation with >= 4 samples per item whose adapters give the noise floor")
    ap.add_argument("--table", type=Path, help="a context-score table for the specific index (default: the run's "
                                                "only selection, if any)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    run = data.Run(args.run_dir)
    grid = data.load_eval(run, args.tag, traits=traits.response_traits)
    null_tag = args.null_tag or (args.tag if grid.samples >= 4 else None)
    null = (grid if null_tag == args.tag else data.load_eval(run, null_tag)) if null_tag else None
    res = analyse(run, grid, null, np.random.default_rng(0), args.table)
    write(args.out or Path("results") / run.name / "heritability", res, report_lines(run, args.tag, grid, res, null_tag))


if __name__ == "__main__":
    with threadpool_limits(stats.THREADS):
        main()
