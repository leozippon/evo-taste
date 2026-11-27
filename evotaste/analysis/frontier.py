"""Pilot F: does consolidating one's own frontier make a member better and more itself?

    python -m evotaste.analysis.frontier runs/<name> --parents p1,p2,p3 [--tag ef90 --parent-tag s45]
                                          [--census results/census/<c>] [--cross-step 90 --selection DIR]
                                          [--boot 1000] [--out DIR]

The arms of a parent p are its clones p-fro (revisits its own frontier), p-yok (a donor's), p-pln (no revisits) and
p-mat (difficulty-matched revisits), where they exist. Three questions:

(a) consolidation.csv: on the training items of each parent's frontier at the clone point (rebuilt from the parent's
    rollouts, checked against the item record a revisiting clone took over; solved at least once in fewer than 3/4 of
    at least min_seen attempts), each clone's attempts after the clone point: how many of the items it met again, its
    success rate there with an interval over those items, at its first and at later visits, against the parent's
    success rate on the same items before the clone point (a selected baseline: frontier items were chosen for a
    middling past rate). Items an arm did not meet again have no estimate and are counted, not imputed.
(b) heldout.csv: on held-out selection items, fro - pln and fro - yok (and mat - pln) of the same parent, overall,
    near and far, and by the base model's stratum, averaged over parents with intervals over items; from the evaluation
    --tag when it is complete, otherwise from the clones' latest common probe.
(c) individuality.csv: noise-corrected adjusted functional distances on the dataset items of --tag (with
    --parent-tag for the parents at the clone point, merged on common items): the mean distance among the fro clones
    against among the pln (and yok) clones, and each clone's distance from its own parent at the clone point, fro
    against yok and pln of the same parent; with --cross-step, the same among-clone comparison by likelihood
    distance. With three parents nothing reaches conventional significance: the report gives effect sizes with item
    intervals, per-parent values, and the number of parents a paired test would need at the observed spread.

Writes results/<name>/frontier/ and report.txt. Definitions: docs/dev/ANALYSIS_NOTES.md.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from threadpoolctl import threadpool_limits

from evotaste.analysis import data, distinct, health, stats
from evotaste.analysis.report import num, pci, render, write
from evotaste.records import read_jsonl_gz

ARMS = ("fro", "yok", "pln", "mat")
PP2 = 1e4


def clones_of(run: data.Run, parents: list[str]) -> dict[str, dict[str, data.Lineage]]:
    """parent -> arm -> committed clone lineage, for the arms that exist."""
    out = {}
    for p in parents:
        if p not in run.lineages:
            raise SystemExit(f"{p}: no lineage in run {run.name}")
        out[p] = {a: run.lineages[f"{p}-{a}"] for a in ARMS
                  if f"{p}-{a}" in run.lineages and run.committed(run.lineages[f"{p}-{a}"])}
        if "fro" not in out[p] or "pln" not in out[p]:
            raise SystemExit(f"{p}: needs committed clones {p}-fro and {p}-pln")
    return out


def rollouts(run: data.Run, lin: data.Lineage, start: int, stop: int) -> pd.DataFrame:
    """uid, step, reward and revisit flag of a lineage's committed rollouts of steps start..stop-1."""
    rows = []
    for t in range(start, min(stop, lin.step)):
        path = lin.dir / "rollouts" / f"step_{t:06d}.jsonl.gz"
        if not path.exists():
            raise FileNotFoundError(f"{path}: a committed step without its rollouts")
        rows += [(r["uid"], t, r["reward"], bool(r.get("revisit", False))) for r in read_jsonl_gz(path)]
    return pd.DataFrame(rows, columns=["uid", "step", "reward", "revisit"])


def history_until(run: data.Run, lin: data.Lineage, upto: int) -> pd.DataFrame:
    """Every rollout behind a lineage's state at step `upto`: its own from its start, then its parents' up to each
    clone point, as the engine rebuilds an item record."""
    start = lin.parent[1] if lin.parent else 0
    own = rollouts(run, lin, start, upto)
    return pd.concat([history_until(run, run.lineages[lin.parent[0]], start), own]) if lin.parent else own


def record(df: pd.DataFrame) -> pd.DataFrame:
    """Per item: attempts and successes."""
    g = df.groupby("uid")["reward"]
    return pd.DataFrame({"attempts": g.size(), "successes": g.apply(lambda r: int((r == 1).sum()))})


def frontier_of(rec: pd.DataFrame, min_seen: int) -> pd.Index:
    """Items solved at least once in fewer than 3/4 of at least min_seen attempts (the engine's frontier)."""
    ok = (rec["attempts"] >= min_seen) & (rec["successes"] >= 1) & (rec["successes"] < 0.75 * rec["attempts"])
    return rec.index[ok]


def check_record(run: data.Run, lin: data.Lineage, rec: pd.DataFrame) -> None:
    """The rebuilt record must equal the one a revisiting clone took over at its clone point."""
    path = lin.dir / "checkpoints" / f"step_{lin.parent[1]:06d}" / "items.jsonl.gz"
    if not path.exists():
        return
    theirs = pd.DataFrame(read_jsonl_gz(path)).set_index("uid")[["attempts", "successes"]]
    if not theirs.sort_index().equals(rec.sort_index().astype(theirs.dtypes.to_dict())):
        raise ValueError(f"{path}: the item record differs from the one rebuilt from the parent's rollouts")


def consolidation(run: data.Run, clones: dict, b: int, rng) -> pd.DataFrame:
    rows = []
    for p, arms in clones.items():
        fro = arms["fro"]
        at = fro.parent[1]
        past = history_until(run, run.lineages[p], at)
        rec = record(past)
        check_record(run, fro, rec)
        min_seen = run.cfg.settings(run.cfg.member(fro.id))[1].revisit.min_seen
        front = frontier_of(rec, min_seen)
        prior = rec.loc[front, "successes"] / rec.loc[front, "attempts"]
        for arm, lin in arms.items():
            later = rollouts(run, lin, at, lin.step)
            later = later[later["uid"].isin(front)]
            visits = later.groupby(["uid", "step"]).agg(n=("reward", "size"), s=("reward", lambda r: (r == 1).sum()),
                                                        revisit=("revisit", "max")).reset_index()
            visits["order"] = visits.groupby("uid")["step"].rank(method="first")
            items = visits.groupby("uid").agg(n=("n", "sum"), s=("s", "sum"))
            row = {"parent": p, "arm": arm, "clone": lin.id,
                   "steps": f"{at}-{lin.step - 1}" if lin.step > at else "none yet", "frontier": len(front),
                   "met_again": len(items), "not_met": len(front) - len(items), "visits": len(visits),
                   "revisit_visits": int(visits["revisit"].sum()) if len(visits) else 0}
            if len(items):
                idx = rng.integers(0, len(items), (b, len(items)))
                s, n = items["s"].to_numpy(), items["n"].to_numpy()
                first, rest = visits[visits["order"] == 1], visits[visits["order"] > 1]
                row |= {"success": s.sum() / n.sum(), **dict(zip(("lo", "hi"), stats.ci(s[idx].sum(1) / n[idx].sum(1)))),
                        "prior_success": float((rec.loc[items.index, "successes"].sum()
                                                / rec.loc[items.index, "attempts"].sum())),
                        "first_visit": first["s"].sum() / first["n"].sum(),
                        "later_visits": rest["s"].sum() / rest["n"].sum() if len(rest) else np.nan,
                        "items_twice": int((visits.groupby("uid").size() >= 2).sum())}
            row["frontier_prior"] = float(prior.mean())
            rows.append(row)
    return pd.DataFrame(rows)


def labels_at(run: data.Run, grid: data.Grid, lineages: list[data.Lineage]) -> dict[str, str]:
    """lineage id -> its label at the latest step the grid holds for it."""
    out = {}
    for label in grid.labels:
        lin, step = run.resolve(label) if label != "base" else (None, None)
        if lin is not None and lin.id in {x.id for x in lineages}:
            if lin.id not in out or step > run.resolve(out[lin.id])[1]:
                out[lin.id] = label
    return out


CONTRASTS = (("fro", "pln"), ("fro", "yok"), ("yok", "pln"), ("mat", "pln"))
HELDOUT = ("accuracy", "near", "far", "frontier", "sweet", "mastered")


def heldout(run: data.Run, clones: dict, grid: data.Grid, b: int, rng) -> pd.DataFrame:
    """Per contrast (a, b) and metric: a - b of the same parent averaged over the parents that have both, with an
    interval over items, and the per-parent values."""
    lab = labels_at(run, grid, [lin for arms in clones.values() for lin in arms.values()])
    point, draws = health.evaluate(grid, rng, b)
    rows = []
    for a, c in CONTRASTS:
        ps = [p for p, arms in clones.items() if a in arms and c in arms and arms[a].id in lab and arms[c].id in lab]
        if not ps:
            continue
        ia = [grid.labels.index(lab[clones[p][a].id]) for p in ps]
        ic = [grid.labels.index(lab[clones[p][c].id]) for p in ps]
        for m in HELDOUT:
            if m not in point:
                continue
            d = point[m][ia, 0] - point[m][ic, 0]
            rows.append({"contrast": f"{a} - {c}", "metric": m, "parents": len(ps), "diff": float(d.mean()),
                         **dict(zip(("lo", "hi"), stats.ci((draws[m][ia] - draws[m][ic]).mean(0)))),
                         "per_parent": " ".join(f"{p}:{x:+.4f}" for p, x in zip(ps, d)),
                         "labels": " ".join(lab[clones[p][a].id] for p in ps)})
    return pd.DataFrame(rows)


def parents_needed(per_parent: np.ndarray) -> float:
    """Parents a paired two-sided 5% test with 80% power would need to detect the observed mean of per-parent effects
    at their observed SD (normal approximation; NaN when the SD is not estimable)."""
    if len(per_parent) < 2 or per_parent.std(ddof=1) == 0:
        return np.nan
    z = sps.norm.ppf(0.975) + sps.norm.ppf(0.8)
    return float(math.ceil((z * per_parent.std(ddof=1) / abs(per_parent.mean())) ** 2)) if per_parent.mean() else np.inf


def individuality(run: data.Run, clones: dict, grid: data.Grid, b: int, rng) -> pd.DataFrame:
    """On the dataset items of the grid (clones at their latest step, parents at the clone point when present):
    adjusted noise-corrected distances among the clones of each arm, and of each clone from its own parent."""
    real = grid.real()
    lins = [lin for arms in clones.values() for lin in arms.values()]
    lab = labels_at(run, real, lins)
    par = {p: f"{p}@{clones[p]['fro'].parent[1]}" for p in clones}
    members = [lab[lin.id] for lin in lins if lin.id in lab] + [x for x in par.values() if x in real.labels]
    sub = real.take(members)
    y, v = stats.cells(sub.x["reward"])
    classes = stats.difficulty_classes(y)
    w = stats.item_draws(data.strata(sub.items), b, rng)
    dp = stats.distances(y, v, classes, stats.ones(y.shape[1]))[0]
    dd = stats.distances(y, v, classes, w)
    pos = {m: i for i, m in enumerate(members)}
    rows = []
    among = {}
    for arm in ARMS:
        idx = [pos[lab[clones[p][arm].id]] for p in clones if arm in clones[p] and clones[p][arm].id in lab]
        if len(idx) >= 2:
            pairs = [(i, j) for k, i in enumerate(idx) for j in idx[k + 1:]]
            among[arm] = (np.mean([dp[i, j] for i, j in pairs]), np.mean([dd[:, i, j] for i, j in pairs], 0), len(idx))
            rows.append({"measure": "among clones", "arm": arm, "n": len(idx), "d2": among[arm][0],
                         **dict(zip(("lo", "hi"), stats.ci(among[arm][1])))})
    for a, c in (("fro", "pln"), ("fro", "yok")):
        if a in among and c in among:
            rows.append({"measure": "among clones, difference", "arm": f"{a} - {c}", "n": among[a][2],
                         "d2": among[a][0] - among[c][0], **dict(zip(("lo", "hi"), stats.ci(among[a][1] - among[c][1])))})
    if all(x in pos for x in par.values()):
        for arm in ARMS:
            ps = [p for p in clones if arm in clones[p] and clones[p][arm].id in lab]
            for p in ps:
                i, j = pos[lab[clones[p][arm].id]], pos[par[p]]
                rows.append({"measure": "from own parent", "arm": arm, "parent": p, "n": 1, "d2": dp[i, j],
                             **dict(zip(("lo", "hi"), stats.ci(dd[:, i, j])))})
        for a, c in (("yok", "fro"), ("pln", "fro")):
            ps = [p for p in clones if a in clones[p] and c in clones[p] and clones[p][a].id in lab and clones[p][c].id in lab]
            if not ps:
                continue
            d = np.array([dp[pos[lab[clones[p][a].id]], pos[par[p]]] - dp[pos[lab[clones[p][c].id]], pos[par[p]]] for p in ps])
            ddr = np.mean([dd[:, pos[lab[clones[p][a].id]], pos[par[p]]] - dd[:, pos[lab[clones[p][c].id]], pos[par[p]]]
                           for p in ps], 0)
            rows.append({"measure": "from own parent, difference", "arm": f"{a} - {c}", "n": len(ps), "d2": float(d.mean()),
                         **dict(zip(("lo", "hi"), stats.ci(ddr))), "per_parent": " ".join(f"{x * PP2:+.1f}" for x in d),
                         "parents_needed": parents_needed(d)})
    return pd.DataFrame(rows)


def likelihood_individuality(run: data.Run, clones: dict, step: int, selection, b: int, rng) -> pd.DataFrame:
    """Mean likelihood distance (nats per token) among the clones of each arm in a cross table at `step`, and the
    fro - pln and fro - yok differences, with intervals over items."""
    df, _ = data.cross_table(run, step, selection)
    c = distinct.Cross(df)
    w = stats.item_draws(data.strata(c.items), b, rng)
    mp, md = distinct.margins(c, stats.ones(len(c.items)))[..., 0], distinct.margins(c, w)
    pos = {s: i for i, s in enumerate(c.subjects)}
    rows, among = [], {}
    for arm in ARMS:
        idx = [pos[f"{clones[p][arm].id}@{step}"] for p in clones if arm in clones[p] and f"{clones[p][arm].id}@{step}" in pos]
        if len(idx) >= 2:
            pairs = [(i, j) for k, i in enumerate(idx) for j in idx[k + 1:]]
            among[arm] = (np.mean([(mp[i, j] + mp[j, i]) / 2 for i, j in pairs]),
                          np.mean([(md[i, j] + md[j, i]) / 2 for i, j in pairs], 0))
            rows.append({"measure": "likelihood among clones", "arm": arm, "n": len(idx), "distance": among[arm][0],
                         **dict(zip(("lo", "hi"), stats.ci(among[arm][1])))})
    for a, c2 in (("fro", "pln"), ("fro", "yok")):
        if a in among and c2 in among:
            rows.append({"measure": "likelihood among clones, difference", "arm": f"{a} - {c2}",
                         "distance": among[a][0] - among[c2][0], **dict(zip(("lo", "hi"), stats.ci(among[a][1] - among[c2][1])))})
    return pd.DataFrame(rows)


def analyse(run, parents, tag=None, parent_tag=None, census=None, cross_step=None, selection=None, b=1000, rng=None):
    rng = rng or np.random.default_rng(0)
    clones = clones_of(run, parents)
    strata = data.census_strata(census) if census else None
    res = {"clones": clones, "consolidation": consolidation(run, clones, b, rng)}
    source = None
    if tag:
        try:
            grid = data.load_eval(run, ",".join(t for t in (tag, parent_tag) if t))
            source = f"evaluation {tag}" + (f" with {parent_tag} for the parents" if parent_tag else "")
        except SystemExit as e:
            res["note"] = f"evaluation {tag} not usable yet ({e}); held-out gains from probes, no functional distances"
    if source is None:  # the clones' latest common probe, single samples: means only
        steps = set.intersection(*[set(run.probe_steps(lin)) for arms in clones.values() for lin in arms.values()])
        step = max(steps)
        grid = data.probe_grid(run, {f"{lin.id}@{step}": (lin, step) for arms in clones.values() for lin in arms.values()})
        source = f"probes at step {step}"
    data.add_strata(grid, strata)
    res["source"] = source
    res["heldout"] = heldout(run, clones, grid.real(), b, rng)
    res["individuality"] = individuality(run, clones, grid, b, rng) if source.startswith("evaluation") and \
        grid.samples >= 2 else pd.DataFrame()
    res["likelihood"] = likelihood_individuality(run, clones, cross_step, selection, b, rng) if cross_step else pd.DataFrame()
    return res


def report_lines(run, res) -> list[str]:
    clones = res["clones"]
    lines = [f"frontier {run.name}: parents " + "; ".join(f"{p} ({', '.join(f'{a}@{lin.step}' for a, lin in arms.items())})"
                                                         for p, arms in clones.items())]
    co = res["consolidation"]
    lines += ["", "(a) consolidation on each parent's frontier at the clone point (training items): items met again "
                  "after it, success rate there [95% over those items], at the first and later visits, against the "
                  "parent's own rate on the same items before (selected: frontier items had a middling rate)"]
    lines += render(["parent", "arm", "steps", "frontier", "met", "not met", "revisit visits", "success [95%]",
                     "first", "later", "parent before"],
                    [[r.parent, r.arm, r.steps, r.frontier, r.met_again, r.not_met, r.revisit_visits,
                      pci(getattr(r, "success", np.nan), getattr(r, "lo", np.nan), getattr(r, "hi", np.nan)),
                      num(getattr(r, "first_visit", np.nan) * 100, digits=1), num(getattr(r, "later_visits", np.nan) * 100, digits=1),
                      num(getattr(r, "prior_success", np.nan) * 100, digits=1)] for r in co.itertuples()], 2)
    lines += ["  arms that met few or none of these items have no estimate on them; nothing is imputed"]
    if "note" in res:
        lines += ["", res["note"]]
    ho = res["heldout"]
    lines += ["", f"(b) held-out selection items ({res['source']}): differences of the same parent's clones, points "
                  "[95% over items], averaged over parents; per parent"]
    lines += [f"  {r.contrast} {r.metric}: {pci(r.diff, r.lo, r.hi, sign=True)} ({r.parents} parents: {r.per_parent})"
              for r in ho.itertuples()] if len(ho) else ["  none"]
    ind = res["individuality"]
    if len(ind):
        lines += ["", "(c) individuality: adjusted noise-corrected functional distance (pp^2) [95% over items]"]
        for r in ind.itertuples():
            extra = f" per parent {r.per_parent}; parents a paired test would need: {r.parents_needed:g}" \
                if isinstance(getattr(r, "per_parent", None), str) else ""
            lines.append(f"  {r.measure} {r.arm}" + (f" ({r.parent})" if isinstance(getattr(r, "parent", None), str) else "")
                         + f": {num(PP2 * r.d2, PP2 * r.lo, PP2 * r.hi, 1)}" + extra)
    else:
        lines += ["", "(c) individuality by solve profile: needs the evaluation (--tag, two samples per item)"]
    lk = res["likelihood"]
    if len(lk):
        lines += ["  by likelihood (nats per token): " + "; ".join(
            f"{r.measure.replace('likelihood ', '')} {r.arm} {num(r.distance, r.lo, r.hi, 3)}" for r in lk.itertuples())]
    lines += ["", "With three parents a paired comparison cannot reach a two-sided sign-flip p below 1/4: read the "
                  "item intervals and whether the parents agree."]
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--parents", required=True, help="comma-separated parents of the clones <parent>-fro|yok|pln|mat")
    ap.add_argument("--tag", help="selection evaluation of the clones (e.g. ef90)")
    ap.add_argument("--parent-tag", help="evaluation holding the parents at the clone point (e.g. s45)")
    ap.add_argument("--census", help="census output directory for the base model's strata")
    ap.add_argument("--cross-step", type=int, help="step of a cross-likelihood table holding the clones")
    ap.add_argument("--selection", type=Path, help="the deviation selection of that table")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    run = data.Run(args.run_dir)
    res = analyse(run, args.parents.split(","), args.tag, args.parent_tag, args.census, args.cross_step, args.selection,
                  args.boot)
    write(args.out or Path("results") / run.name / "frontier",
          {k: v for k, v in res.items() if isinstance(v, pd.DataFrame)}, report_lines(run, res))


if __name__ == "__main__":
    with threadpool_limits(stats.THREADS):
        main()
