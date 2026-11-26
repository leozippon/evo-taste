"""Divergence of members on a selection evaluation with at least two samples per item: the member-by-item interaction,
functional distances, domain and source profiles, counted traits, the real-data null, and authorship.

    python -m evotaste.analysis.divergence runs/<name> --tag T [--null-tag T0] [--step S | --members a@60,b@60]
                                            [--classifier-items 400] [--out DIR]

Members are the evaluated labels at one step (default: the step with the most labels), base excluded. The null is
every adapter of --null-tag (default: the main tag when it has four or more samples) with its samples split into two
halves: two copies of one adapter, whose true distance is zero. The headline estimators read the dataset units; the
procedural format probe enters only the format table. Writes results/<name>/divergence/: interaction.csv,
distances.csv, profile.csv, traits.csv, trait_spread.csv, null.csv, authorship.csv, format.csv, length.csv and
report.txt. Definitions and validation: docs/dev/ANALYSIS_NOTES.md.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evotaste.analysis import data, health, stats, traits
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


PREFIX = 600  # characters of an answer that the length-free views keep


def _text_model():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, make_pipeline

    return make_pipeline(FeatureUnion([
        ("w", TfidfVectorizer(token_pattern=r"(?u)\b\w+\b|[^\w\s]+", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True))]),
        LogisticRegression(max_iter=2000))


def _feature_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


# What the classifier reads of a response: name -> (features of sample 0 of the subjects' answers to item i, model).
# raw: the text with digits replaced; stripped: without layout and notation; layout: the layout traits alone;
# prefix and stripped_prefix: the first PREFIX characters of those, which hide how long the answer is; length: the
# completion tokens (log) and whether the answer was cut off, alone.
VIEWS = {
    "raw": (lambda g, idx, i: [traits.raw_view(t) for t in g.texts[idx, i]], _text_model),
    "stripped": (lambda g, idx, i: [traits.stripped_view(t) for t in g.texts[idx, i]], _text_model),
    "layout": (lambda g, idx, i: [[traits.layout_traits(t)[k] for k in traits.LAYOUT] for t in g.texts[idx, i]],
               _feature_model),
    "prefix": (lambda g, idx, i: [traits.raw_view(t)[:PREFIX] for t in g.texts[idx, i]], _text_model),
    "stripped_prefix": (lambda g, idx, i: [traits.stripped_view(t)[:PREFIX] for t in g.texts[idx, i]], _text_model),
    "length": (lambda g, idx, i: np.c_[np.log(g.x["tokens"][idx, i, 0]), g.x["truncated"][idx, i, 0]],
               _feature_model),
}


def authorship(grid: data.Grid, members: list[str], cap: int, rng, views=("raw", "stripped", "layout"),
               contrasts=(("cosmetic_share", "raw", "stripped"),), b: int = stats.B) -> pd.DataFrame:
    """Which member wrote a response: trained on sample 0 of the items of half the units and tested on the other half,
    both ways, once per view of the response (VIEWS). Accuracy with a bootstrap interval over test items (all
    members' answers to an item together); each contrast (name, a, b) is the share (a - b) / (a - chance) of what view
    a identifies beyond chance that view b loses, with the same draws (cosmetic_share: what stripping layout and
    notation loses)."""
    idx = [grid.labels.index(m) for m in members]
    units = sorted(grid.items["family"].unique())
    half = np.array([units.index(f) % 2 for f in grid.items["family"]])
    chosen = []
    for h in (0, 1):
        pool = np.flatnonzero(half == h)
        chosen.append(np.sort(rng.choice(pool, min(cap, len(pool)), replace=False)))
    labels = np.array(members)
    correct = {}
    with threadpool_limits(stats.THREADS):
        for view in views:
            features, model = VIEWS[view]
            hits = np.zeros((len(members), len(grid.items)))
            for train, test in ((chosen[0], chosen[1]), (chosen[1], chosen[0])):
                fx = [f for i in train for f in features(grid, idx, i)]
                fit = model().fit(fx, np.tile(labels, len(train)))
                for i in test:
                    hits[:, i] = fit.predict(features(grid, idx, i)) == labels
            correct[view] = hits
    tested = np.concatenate(chosen)
    draws = rng.choice(tested, (b, len(tested)))
    chance = 1 / len(members)
    acc = {v: (h[:, tested].mean(), h[:, draws].mean((0, 2))) for v, h in correct.items()}
    rows = [{"view": v, "members": len(members), "items": len(tested), "accuracy": a, "chance": chance,
             **dict(zip(("lo", "hi"), stats.ci(d)))} for v, (a, d) in acc.items()]
    for name, va, vb in contrasts:
        with np.errstate(invalid="ignore", divide="ignore"):
            share = (acc[va][1] - acc[vb][1]) / (acc[va][1] - chance)
        rows.append({"view": name, "members": len(members), "items": len(tested),
                     "accuracy": (acc[va][0] - acc[vb][0]) / (acc[va][0] - chance), "chance": np.nan,
                     **dict(zip(("lo", "hi"), stats.ci(share)))})
    return pd.DataFrame(rows)


def format_table(grid: data.Grid, members: list[str], rng) -> pd.DataFrame:
    """Per member, on the dataset units and on the procedural format probe: strict accuracy (`reward`), tolerant
    correctness (`correct_any`, any known answer wrapper), compliance with the wrapper asked for, and the gap
    correct_any - reward, each with an item-bootstrap interval."""
    rows = []
    for part, sub in (("dataset", grid.real()), ("procedural", grid.procedural())):
        if not len(sub.items):
            continue
        sub = sub.take(members)
        w = stats.item_draws(data.strata(sub.items), stats.B, rng)
        every = np.ones(sub.x["reward"].shape, bool)
        vals = {k: (health._rate(sub.x[k], every, stats.ones(len(sub.items))), health._rate(sub.x[k], every, w))
                for k in ("reward", "correct_any", "complied")}
        vals["gap"] = tuple(vals["correct_any"][i] - vals["reward"][i] for i in (0, 1))
        for j, m in enumerate(members):
            rows.append({"member": m, "part": part, "items": len(sub.items), **{
                k2: v for k, (p0, pb) in vals.items() for k2, v in
                ((k, p0[j, 0]), (f"{k}_lo", stats.ci(pb[j])[0]), (f"{k}_hi", stats.ci(pb[j])[1]))}})
    return pd.DataFrame(rows)


def length_table(grid: data.Grid, members: list[str], rng) -> pd.DataFrame:
    """Per member and domain of the dataset units: mean and 90th percentile of completion tokens and the share
    truncated, with item-bootstrap intervals."""
    sub = grid.real().take(members)
    w = stats.item_draws(data.strata(sub.items), stats.B, rng)
    point, draws = health.length_metrics(sub, stats.ones(len(sub.items))), health.length_metrics(sub, w)
    rows = []
    for j, m in enumerate(members):
        for d in ["all"] + sorted(sub.items["domain"].unique()):
            suffix = "" if d == "all" else f"_{d}"
            row = {"member": m, "domain": d}
            for k in ("tokens", "tokens_p90", "truncated"):
                row[k] = point[k + suffix][j, 0]
                row[f"{k}_lo"], row[f"{k}_hi"] = stats.ci(draws[k + suffix][j])
            rows.append(row)
    return pd.DataFrame(rows)


def analyse(run: data.Run, grid: data.Grid, members: list[str], null: data.Grid | None, cap: int, rng) -> dict:
    """Every divergence table. The headline estimators read the dataset units only; the procedural format probe,
    which tests answer notation as much as content, enters the format table."""
    whole, grid = grid, grid.real()
    null = null.real() if null is not None else None
    stats.need_samples(grid.x["reward"], "divergence")
    arm_of = {m: run.resolve(m)[0].arm for m in members}
    idx = [grid.labels.index(m) for m in members]
    y, v = stats.cells(grid.x["reward"][idx])
    items = grid.items

    def w_of(it):
        return stats.item_draws(data.strata(it), stats.B, rng)

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
    out["format"], out["length"] = format_table(whole, members, rng), length_table(whole, members, rng)
    return out


def report_lines(run, tag, members, grid, res, null_tag) -> list[str]:
    real = grid.real()
    lines = [f"divergence {run.name}, selection {tag}: {len(members)} members x {len(real.items)} items of "
             f"{real.items['family'].nunique()} dataset units x {grid.samples} samples (the "
             f"{len(grid.items) - len(real.items)} items of the procedural format probe only in the format table); "
             "SDs in percentage points [95% bootstrap over items]"]
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
    f = res["format"]
    if len(f):
        wide = {(r.member, r.part): r for r in f.itertuples()}
        parts = list(dict.fromkeys(f["part"]))
        lines += ["", "answer format: strict accuracy, correct_any (any known wrapper) and their gap, on the dataset "
                      "units and on the procedural format probe; compliance with the wrapper asked for"]
        lines += render(["member"] + [f"{p} {k}" for p in parts for k in ("strict", "gap [95%]", "complied")],
                        [[m] + [x for p in parts for x in (
                            pct(wide[(m, p)].reward), pci(wide[(m, p)].gap, wide[(m, p)].gap_lo, wide[(m, p)].gap_hi),
                            pct(wide[(m, p)].complied))] for m in members])
    ln = res["length"]
    if len(ln):
        doms = list(dict.fromkeys(ln["domain"]))
        cell = {(r.member, r.domain): r for r in ln.itertuples()}
        lines += ["", "length on the dataset units: mean / 90th percentile of completion tokens, share truncated (%)"]
        lines += render(["member"] + doms, [[m] + [f"{cell[(m, d)].tokens:.0f} / {cell[(m, d)].tokens_p90:.0f} / "
                                                   f"{pct(cell[(m, d)].truncated)}" for d in doms] for m in members])
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
