"""Divergence of members on a selection evaluation with at least two samples per item: the member-by-item interaction,
functional distances, domain and source profiles, counted traits, the real-data null, and authorship.

    python -m evotaste.analysis.divergence runs/<name> --tag T [--null-tag T0] [--step S | --members a@60,b@60]
                                            [--classifier-items 400] [--out DIR]

Members are the evaluated labels at one step (default: the step with the most labels), base excluded. The null is
every adapter of --null-tag (default: the main tag when it has four or more samples) with its samples split into two
halves: two copies of one adapter, whose true distance is zero. Writes results/<name>/divergence/: interaction.csv,
distances.csv, profile.csv, traits.csv, trait_spread.csv, null.csv, authorship.csv and report.txt. Definitions and
validation: docs/dev/ANALYSIS_NOTES.md.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evotaste.analysis import data, stats, traits
from evotaste.analysis.report import num, pci, pct, render, write

PP2 = 1e4  # variance in percentage points squared


def pick_members(run: data.Run, grid: data.Grid, step: int | None, members: list[str] | None) -> list[str]:
    if members:
        missing = [m for m in members if m not in grid.labels]
        if missing:
            raise SystemExit(f"not evaluated under this tag: {missing}")
        return members
    at = collections.defaultdict(list)
    for label in grid.labels:
        lin, s = run.resolve(label)
        if lin is not None:
            at[s].append(label)
    if not at:
        raise SystemExit("no member is evaluated under this tag")
    if step is None:
        step = max(at, key=lambda s: (len(at[s]), s))
    if len(at.get(step, [])) < 2:
        raise SystemExit(f"fewer than two members evaluated at step {step}")
    return at[step]


def interaction_rows(y, v, items, w, label: str) -> list[dict]:
    """Interaction estimates of one member set: plain (all items, and items neither never nor always solved) and
    adjusted for difficulty, overall and within each domain."""
    rows = []
    scopes = [("all", np.ones(len(items), bool))] + [(d, (items["domain"] == d).to_numpy())
                                                     for d in sorted(items["domain"].unique())]
    for scope, keep in scopes:
        ys, vs, ws = y[:, keep], v[:, keep], w[:, keep]
        for estimator, classes in (("plain", stats.one_class(ys)), ("plain_nonconstant", stats.one_class(ys, True)),
                                   ("adjusted", stats.difficulty_classes(ys))):
            point = stats.interaction(ys, vs, classes, stats.ones(keep.sum()))[0]
            draws = stats.interaction(ys, vs, classes, ws)
            lo, hi = stats.ci(draws)
            rows.append({"members": label, "m": len(y), "scope": scope, "estimator": estimator,
                         "items": int((classes >= 0).sum()), "var": point, "var_lo": lo, "var_hi": hi,
                         "sd": stats.sd(point), "sd_lo": stats.sd(lo), "sd_hi": stats.sd(hi), "_draws": draws})
    return rows


def null_pairs(grid: data.Grid, w_of) -> pd.DataFrame:
    """Every adapter of an evaluation with K >= 4 samples split into its first K // 2 samples and the rest: plain
    and adjusted distance between the halves, with item-bootstrap intervals; the last rows average over adapters."""
    k = grid.samples
    if k < 4:
        raise SystemExit(f"the null needs an evaluation with at least four samples per item; this one has {k}")
    w = w_of(grid.items)
    rows, draws = [], collections.defaultdict(list)
    for s, label in enumerate(grid.labels):
        halves = [grid.x["reward"][s:s + 1, :, :k // 2], grid.x["reward"][s:s + 1, :, k // 2:]]
        cy, cv = zip(*(stats.cells(h) for h in halves))
        y, v = np.vstack(cy), np.vstack(cv)
        for estimator, classes in (("plain", stats.one_class(y)), ("adjusted", stats.difficulty_classes(y))):
            point = stats.distances(y, v, classes, stats.ones(y.shape[1]))[0, 0, 1]
            d = stats.distances(y, v, classes, w)[:, 0, 1]
            draws[estimator].append(d)
            rows.append({"adapter": label, "estimator": estimator, "d2": point, **dict(zip(("lo", "hi"), stats.ci(d)))})
    for estimator, d in draws.items():
        mean = np.mean(d, axis=0)
        rows.append({"adapter": "mean", "estimator": estimator,
                     "d2": float(np.mean([r["d2"] for r in rows if r["estimator"] == estimator and r["adapter"] != "mean"])),
                     **dict(zip(("lo", "hi"), stats.ci(mean)))})
    return pd.DataFrame(rows)


def trait_tables(grid: data.Grid, members: list[str], arm_of: dict, w) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per member the mean of every counted and layout trait with its interval, and per trait the between-member
    variance of these means beyond their sampling noise (all members and per update rule)."""
    idx = [grid.labels.index(m) for m in members]
    rows, spread = [], []
    for kind, names in (("counted", traits.COUNTED), ("layout", traits.LAYOUT)):
        for t in names:
            y, v = stats.cells(grid.x[t][idx])
            if not np.isfinite(y).any():
                continue
            point, pv = stats.member_means(y, v, stats.ones(y.shape[1]))
            tb, nb = stats.member_means(y, v, w)
            for j, m in enumerate(members):
                rows.append({"member": m, "arm": arm_of[m], "kind": kind, "trait": t, "value": point[j, 0],
                             **dict(zip(("lo", "hi"), stats.ci(tb[j])))})
            groups = {"all": list(range(len(members)))}
            for a in sorted(set(arm_of.values())):
                groups[a] = [j for j, m in enumerate(members) if arm_of[m] == a]
            for g, sel in groups.items():
                if len(sel) < 2:
                    continue
                b0 = stats.between(point[sel], pv[sel])[0]
                lo, hi = stats.ci(stats.between(tb[sel], nb[sel]))
                spread.append({"members": g, "kind": kind, "trait": t, "mean": float(point[sel].mean()),
                               "between_var": b0, "lo": lo, "hi": hi, "between_sd": stats.sd(b0),
                               "noise_sd": float(np.sqrt(pv[sel].mean()))})
    return pd.DataFrame(rows), pd.DataFrame(spread)


def authorship(grid: data.Grid, members: list[str], cap: int, rng) -> pd.DataFrame:
    """Which member wrote a response: trained on sample 0 of the items of half the units and tested on the other half,
    both ways; on raw text, on text stripped of layout and notation, and on layout traits alone. Accuracy with a
    bootstrap interval over test items (all members' answers to an item together); the cosmetic share is
    (raw - stripped) / (raw - chance), with the same draws."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, make_pipeline
    from sklearn.preprocessing import StandardScaler

    idx = [grid.labels.index(m) for m in members]
    units = sorted(grid.items["family"].unique())
    half = np.array([units.index(f) % 2 for f in grid.items["family"]])
    chosen = []
    for h in (0, 1):
        pool = np.flatnonzero(half == h)
        chosen.append(np.sort(rng.choice(pool, min(cap, len(pool)), replace=False)))
    labels = np.array(members)

    def text_model():
        return make_pipeline(FeatureUnion([
            ("w", TfidfVectorizer(token_pattern=r"(?u)\b\w+\b|[^\w\s]+", ngram_range=(1, 2), min_df=2,
                                  sublinear_tf=True)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True))]),
            LogisticRegression(max_iter=2000))

    views = {"raw": (lambda i: [traits.raw_view(t) for t in grid.texts[idx, i]], text_model),
             "stripped": (lambda i: [traits.stripped_view(t) for t in grid.texts[idx, i]], text_model),
             "layout": (lambda i: [[traits.layout_traits(t)[k] for k in traits.LAYOUT] for t in grid.texts[idx, i]],
                        lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))}
    correct = {}
    with threadpool_limits(stats.THREADS):
        for view, (features, model) in views.items():
            hits = np.zeros((len(members), len(grid.items)))
            for train, test in ((chosen[0], chosen[1]), (chosen[1], chosen[0])):
                fx = [f for i in train for f in features(i)]
                fit = model().fit(fx, np.tile(labels, len(train)))
                for i in test:
                    hits[:, i] = fit.predict(features(i)) == labels
            correct[view] = hits
    tested = np.concatenate(chosen)
    draws = rng.choice(tested, (stats.B, len(tested)))
    chance = 1 / len(members)
    acc = {v: (h[:, tested].mean(), h[:, draws].mean((0, 2))) for v, h in correct.items()}
    rows = [{"view": v, "members": len(members), "items": len(tested), "accuracy": a, "chance": chance,
             **dict(zip(("lo", "hi"), stats.ci(d)))} for v, (a, d) in acc.items()]
    with np.errstate(invalid="ignore", divide="ignore"):
        share = (acc["raw"][1] - acc["stripped"][1]) / (acc["raw"][1] - chance)
    rows.append({"view": "cosmetic_share", "members": len(members), "items": len(tested),
                 "accuracy": (acc["raw"][0] - acc["stripped"][0]) / (acc["raw"][0] - chance), "chance": np.nan,
                 **dict(zip(("lo", "hi"), stats.ci(share)))})
    return pd.DataFrame(rows)


def analyse(run: data.Run, grid: data.Grid, members: list[str], null: data.Grid | None, cap: int, rng) -> dict:
    stats.need_samples(grid.x["reward"], "divergence")
    arm_of = {m: run.resolve(m)[0].arm for m in members}
    idx = [grid.labels.index(m) for m in members]
    y, v = stats.cells(grid.x["reward"][idx])
    items = grid.items

    def w_of(it):
        return stats.item_draws(it["family"], stats.B, rng)

    w = w_of(items)
    sets = {"all": list(range(len(members)))}
    for a in sorted(set(arm_of.values())):
        sets[a] = [j for j, m in enumerate(members) if arm_of[m] == a]
    rows = []
    for g, sel in sets.items():
        if len(sel) >= 2 and (g == "all" or len(sets) > 2):
            rows += interaction_rows(y[sel], v[sel], items, w, g)
    arms = [g for g in sets if g != "all" and len(sets[g]) >= 2]
    for i, a in enumerate(arms):  # differences between update rules, paired by item draws
        for b_ in arms[i + 1:]:
            for ra in [r for r in rows if r["members"] == a]:
                rb = next(r for r in rows if r["members"] == b_ and r["scope"] == ra["scope"]
                          and r["estimator"] == ra["estimator"])
                lo, hi = stats.ci(ra["_draws"] - rb["_draws"])
                rows.append({"members": f"{a} - {b_}", "m": ra["m"] + rb["m"], "scope": ra["scope"],
                             "estimator": ra["estimator"], "items": ra["items"], "var": ra["var"] - rb["var"],
                             "var_lo": lo, "var_hi": hi})
    inter = pd.DataFrame(rows).drop(columns="_draws")

    dist_rows = []
    for estimator, classes in (("plain", stats.one_class(y)), ("adjusted", stats.difficulty_classes(y))):
        point = stats.distances(y, v, classes, stats.ones(y.shape[1]))[0]
        draws = stats.distances(y, v, classes, w)
        for a in range(len(members)):
            for b_ in range(a + 1, len(members)):
                dist_rows.append({"a": members[a], "b": members[b_], "same_arm": arm_of[members[a]] == arm_of[members[b_]],
                                  "estimator": estimator, "d2": point[a, b_],
                                  **dict(zip(("lo", "hi"), stats.ci(draws[:, a, b_])))})
    prof = []
    for by in ("domain", "family"):
        point = stats.profile(y, v, items[by], stats.ones(y.shape[1]))[0]
        lo, hi = stats.ci(stats.profile(y, v, items[by], w))
        prof.append({"by": "domain" if by == "domain" else "source", "groups": items[by].nunique(), "var": point,
                     "lo": lo, "hi": hi, "sd": stats.sd(point), "sd_lo": stats.sd(lo), "sd_hi": stats.sd(hi)})
    tr, spread = trait_tables(grid, members, arm_of, w)
    out = {"interaction": inter, "distances": pd.DataFrame(dist_rows), "profile": pd.DataFrame(prof), "traits": tr,
           "trait_spread": spread}
    out["null"] = null_pairs(null, w_of) if null is not None else pd.DataFrame()
    out["authorship"] = authorship(grid, members, cap, rng) if cap > 0 and grid.texts is not None else pd.DataFrame()
    return out


def report_lines(run, tag, members, grid, res, null_tag) -> list[str]:
    lines = [f"divergence {run.name}, selection {tag}: {len(members)} members x {len(grid.items)} items x "
             f"{grid.samples} samples; SDs in percentage points [95% bootstrap over items, within units]"]
    if grid.aliases:
        lines.append("same adapter under several labels (kept the first): "
                     + ", ".join(f"{a} = {b}" for a, b in grid.aliases.items()))
    i = res["interaction"]
    lines += ["", "member-by-item interaction SD (adjusted: within difficulty classes, removing each member's effect per "
                  "class; plain: the two-way interaction, inflated by differences in overall level)"]
    rows = []
    for r in i[~i["members"].str.contains(" - ")].itertuples():
        rows.append([r.members, r.m, r.scope, r.estimator, r.items, num(100 * r.sd, 100 * r.sd_lo, 100 * r.sd_hi, 2)])
    lines += render(["members", "m", "scope", "estimator", "items", "SD [95%]"], rows, 4)
    d = i[i["members"].str.contains(" - ") & (i["scope"] == "all")]
    lines += [f"  {r.members}, {r.estimator}: variance difference {PP2 * r.var:+.2f} pp^2 [{PP2 * r.var_lo:+.2f}, "
              f"{PP2 * r.var_hi:+.2f}]" for r in d.itertuples()]
    p = res["profile"]
    lines += ["", "profile (each member's accuracy by group relative to its own mean; between-member SD beyond noise)"]
    lines += [f"  by {r.by} ({r.groups} groups): SD {num(100 * r.sd, 100 * r.sd_lo, 100 * r.sd_hi, 2)}"
              for r in p.itertuples()]
    dist = res["distances"]
    lines += ["", "functional distance (noise-corrected mean square difference of solve rates, pp^2)"]
    for est, g in dist.groupby("estimator"):
        same, cross = g[g["same_arm"]], g[~g["same_arm"]]
        lines.append(f"  {est}: mean over {len(g)} pairs {PP2 * g['d2'].mean():.1f}; same rule {PP2 * same['d2'].mean():.1f}"
                     + (f", across rules {PP2 * cross['d2'].mean():.1f}" if len(cross) else "")
                     + f"; closest {g.loc[g['d2'].idxmin(), 'a']}/{g.loc[g['d2'].idxmin(), 'b']} "
                       f"{PP2 * g['d2'].min():.1f}, farthest {g.loc[g['d2'].idxmax(), 'a']}/{g.loc[g['d2'].idxmax(), 'b']} "
                       f"{PP2 * g['d2'].max():.1f}")
    nl = res["null"]
    if len(nl):
        lines += [f"  null ({null_tag}, each adapter's halves): " + "; ".join(
            f"{r.estimator} {PP2 * r.d2:+.2f} [{PP2 * r.lo:+.2f}, {PP2 * r.hi:+.2f}]"
            for r in nl[nl["adapter"] == "mean"].itertuples()) + " (true value 0)"]
    else:
        lines += ["  null: not available (needs --null-tag, an evaluation with four or more samples per item)"]
    s = res["trait_spread"]
    if len(s):
        lines += ["", "traits: mean over members and between-member SD beyond sampling noise (all members)"]
        a = s[s["members"] == "all"]
        lines += render(["kind", "trait", "mean", "between SD [95%]", "noise SD"],
                        [[r.kind, r.trait, num(r.mean, digits=3), num(r.between_sd, stats.sd(r.lo), stats.sd(r.hi), 3),
                          num(r.noise_sd, digits=3)] for r in a.itertuples()], 2)
    au = res["authorship"]
    if len(au):
        lines += ["", "authorship (train on half the units, test on the others, both ways)"]
        lines += [f"  {r.view}: {pci(r.accuracy, r.lo, r.hi)}" + (f" (chance {pct(r.chance)})" if r.view != "cosmetic_share"
                                                                   else "") for r in au.itertuples()]
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--tag", required=True, help="selection evaluation tag")
    ap.add_argument("--null-tag", help="evaluation with >= 4 samples per item whose adapters give the null")
    ap.add_argument("--step", type=int, help="members evaluated at this step (default: the step with most labels)")
    ap.add_argument("--members", help="comma-separated labels instead of --step")
    ap.add_argument("--classifier-items", type=int, default=400, help="items per half of the units for authorship "
                                                                      "(0: skip)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    run = data.Run(args.run_dir)
    grid = data.load_eval(run, args.tag, traits=traits.response_traits, keep_texts=args.classifier_items > 0)
    members = pick_members(run, grid, args.step, args.members.split(",") if args.members else None)
    null_tag = args.null_tag or (args.tag if grid.samples >= 4 else None)
    null = (grid if null_tag == args.tag else data.load_eval(run, null_tag)) if null_tag else None
    res = analyse(run, grid, members, null, args.classifier_items, np.random.default_rng(0))
    write(args.out or Path("results") / run.name / "divergence", res,
          report_lines(run, args.tag, members, grid, res, null_tag))


if __name__ == "__main__":
    with threadpool_limits(stats.THREADS):
        main()
