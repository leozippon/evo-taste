"""How much of the member-by-item interaction and of the pairwise functional distance is answer length and truncation,
and how much of the members' distinguishability.

    python -m evotaste.analysis.length runs/<name> --tag T [--null-tag T0] [--step S | --members a@45,b@45]
                                        [--boot 1000] [--perms 1000] [--classifier-items 400] [--out DIR]

Members are chosen as by divergence (the evaluated labels at one step); the dataset units only. A member's length is
its mean completion tokens (log) and its truncation the share of its samples cut off at the limit, both over all
items of the scope (all dataset units, or one domain). Four analyses:

- interaction.csv: the interaction (plain and adjusted, as divergence) on all items, on the items on which no sample
  of any member was cut off ("clean"), on the others ("cut"), and per domain, after removing nested member models from
  every item's column of solve rates: none, the update rule, + length, + truncation (stats.member_projection). The
  share explained is 1 - residual / interaction; the length part is what length and truncation remove beyond the
  rule, tested by permuting members' length and truncation within rules.
- bands.csv: the adjusted interaction among members of similar length (within each rule, bands of about five
  members of adjacent length), against random bands of the same sizes within rules.
- association.csv: the pairwise adjusted functional distance regressed on the pairs' difference in log length and in
  truncation, with same-rule as a covariate; partial correlations and the R^2 gain, p-values from relabelling
  members within rules (a Mantel test), intervals over items.
- authorship.csv: the authorship classifier on raw text, on its first characters only (no length cue), on those
  stripped of layout and notation, and on length alone.

Writes results/<name>/length/: members.csv, interaction.csv, bands.csv, association.csv, authorship.csv, null.csv
(with a four-sample evaluation) and report.txt. Definitions and validation: docs/dev/ANALYSIS_NOTES.md.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evotaste.analysis import data, divergence, stats
from evotaste.analysis.report import num, pci, pct, render, write

PP2 = 1e4
BAND = 5  # target members per band of similar length


def covariates(sub: data.Grid) -> dict[str, np.ndarray]:
    """Per member over the grid's items and samples: mean completion tokens, its log, and the share cut off."""
    tokens = sub.x["tokens"].mean((1, 2))
    return {"tokens": tokens, "log_tokens": np.log(tokens), "truncated": sub.x["truncated"].mean((1, 2))}


def rule_dummies(rules: np.ndarray) -> np.ndarray | None:
    names = list(dict.fromkeys(rules))
    return None if len(names) < 2 else np.stack([(rules == r).astype(float) for r in names[1:]], 1)


def models(rules: np.ndarray, log_tokens: np.ndarray, truncated: np.ndarray) -> dict[str, np.ndarray | None]:
    """Nested member models, name -> covariates (members x k): none, rule (when there are two or more), + length
    (log mean tokens), + truncation (share cut off; left out when no member has any)."""
    r = rule_dummies(rules)
    pre = [] if r is None else [r]
    out = {"none": None} | ({} if r is None else {"rule": r})
    out[("rule+" if r is not None else "") + "length"] = np.column_stack(pre + [log_tokens])
    if np.ptp(truncated) > 0:
        out[("rule+" if r is not None else "") + "length+truncation"] = np.column_stack(pre + [log_tokens, truncated])
    return out


def within_rule_permutations(rules: np.ndarray, n: int, rng) -> np.ndarray:
    """n permutations of member indices that keep every member within its rule."""
    out = np.tile(np.arange(len(rules)), (n, 1))
    for r in dict.fromkeys(rules):
        idx = np.flatnonzero(rules == r)
        out[:, idx] = idx[rng.permuted(np.tile(np.arange(len(idx)), (n, 1)), axis=1)]
    return out


def interaction_rows(y, v, classes, w, rules, log_tokens, truncated, perm) -> list[dict]:
    """Residual interaction of each nested model with its interval, the share of the interaction it explains, and
    for the full model the length part (beyond the rule) with its within-rule permutation null."""
    zs = models(rules, log_tokens, truncated)
    ones = stats.ones(y.shape[1])
    point = {k: stats.interaction(y, v, classes, ones, z)[0] for k, z in zs.items()}
    draws = {k: stats.interaction(y, v, classes, w, z) for k, z in zs.items()}
    ref, full = ("rule" if "rule" in zs else "none"), list(zs)[-1]
    part = (point[ref] - point[full]) / point["none"]
    null = np.array([(point[ref] - stats.interaction(y, v, classes, ones, models(
        rules, log_tokens[p], truncated[p])[full])[0]) / point["none"] for p in perm])
    rows = []
    for k in zs:
        lo, hi = stats.ci(draws[k])
        with np.errstate(invalid="ignore", divide="ignore"):
            share = 1 - draws[k] / draws["none"]
        row = {"model": k, "items": int((classes >= 0).sum()), "var": point[k], "var_lo": lo, "var_hi": hi,
               "sd": stats.sd(point[k]), "sd_lo": stats.sd(lo), "sd_hi": stats.sd(hi),
               "share": 1 - point[k] / point["none"], **dict(zip(("share_lo", "share_hi"), stats.ci(share)))}
        if k == full:
            with np.errstate(invalid="ignore", divide="ignore"):
                pd_ = (draws[ref] - draws[full]) / draws["none"]
            row |= {"length_part": part, **dict(zip(("length_part_lo", "length_part_hi"), stats.ci(pd_))),
                    "length_part_null": float(null.mean()), "p_perm": stats.perm_p(part, null)}
        rows.append(row)
    return rows


def interaction_table(y, v, items, scopes, rules, b, perms, rng) -> pd.DataFrame:
    """interaction_rows for every item set (name, mask, covariates of the scope) and both estimators."""
    rows = []
    perm = within_rule_permutations(rules, perms, rng)
    for name, keep, cov in scopes:
        if keep.sum() < 20:
            continue
        ys, vs = y[:, keep], v[:, keep]
        w = stats.item_draws(data.strata(items[keep]), b, rng)
        for est, classes in (("plain", stats.one_class(ys)), ("adjusted", stats.difficulty_classes(ys))):
            for r in interaction_rows(ys, vs, classes, w, rules, cov["log_tokens"], cov["truncated"], perm):
                rows.append({"items_set": name, "estimator": est, **r})
    return pd.DataFrame(rows)


def bands_of(order: np.ndarray, rules: np.ndarray) -> list[np.ndarray]:
    """Bands of about BAND members of adjacent position in `order` within each rule of six or more members."""
    out = []
    for r in dict.fromkeys(rules):
        idx = [i for i in order if rules[i] == r]
        if len(idx) >= 6:
            out += np.array_split(np.array(idx), max(2, round(len(idx) / BAND)))
    return out


def pooled_bands(y, v, bands, w) -> np.ndarray:
    """The adjusted interaction pooled over member bands (sum of each band's sum of squares beyond noise over the sum
    of its degrees of freedom), difficulty classes formed within each band."""
    num, den = np.zeros(len(w)), np.zeros(len(w))
    for band in bands:
        n_, d_ = stats.interaction_parts(y[band], v[band], stats.difficulty_classes(y[band]), w)
        num, den = num + n_, den + d_
    return num / den


def band_table(y, v, items, rules, log_tokens, b, perms, rng) -> pd.DataFrame:
    """Interaction among members of similar length against random bands of the same sizes within rules: the
    point values, the permutation p-value (one-sided, similar-length bands lower), and the ratio with an interval over
    items (its denominator the mean over 50 of the random partitions, the same in every draw)."""
    bands = bands_of(np.argsort(log_tokens), rules)
    if not bands:
        return pd.DataFrame()
    sizes = [len(x) for x in bands]
    perm = within_rule_permutations(rules, perms, rng)
    ones = stats.ones(y.shape[1])
    observed = pooled_bands(y, v, bands, ones)[0]
    null = np.array([pooled_bands(y, v, bands_of(p, rules), ones)[0] for p in perm])  # p: a random order within rules
    w = stats.item_draws(data.strata(items), b, rng)
    obs_d = pooled_bands(y, v, bands, w)
    rand_d = np.mean([pooled_bands(y, v, bands_of(p, rules), w) for p in perm[:50]], 0)
    whole = stats.interaction(y, v, stats.difficulty_classes(y), ones)[0]
    return pd.DataFrame([{
        "bands": len(bands), "sizes": " ".join(map(str, sizes)), "similar_length": observed,
        **dict(zip(("similar_lo", "similar_hi"), stats.ci(obs_d))), "random_bands": float(null.mean()),
        "ratio": observed / null.mean(), **dict(zip(("ratio_lo", "ratio_hi"), stats.ci(obs_d / rand_d))),
        "p_perm": stats.perm_p(-observed, -null), "all_members": whole,
        "length_sd_within_bands": float(np.mean([log_tokens[x].std() for x in bands])),
        "length_sd": float(log_tokens.std())}])


def partial_stats(d: np.ndarray, x: np.ndarray, c: np.ndarray | None) -> np.ndarray:
    """For pair values d and pair predictors x (pairs x 2), with covariates c (pairs x k or None): the partial
    correlation of d with each predictor given c, the R^2 gain of both beyond c, and the slope of each in the joint
    model: [r1, r2, r2_gain, slope1, slope2]."""
    base = np.ones((len(d), 1)) if c is None else np.column_stack([np.ones(len(d)), c])

    def resid(t):
        return t - base @ np.linalg.lstsq(base, t, rcond=None)[0]

    rd, rx = resid(d), resid(x)
    coef = np.linalg.lstsq(rx, rd, rcond=None)[0]  # the joint slopes (Frisch-Waugh)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = [float(rd @ rx[:, j] / np.sqrt((rd @ rd) * (rx[:, j] @ rx[:, j]))) for j in range(x.shape[1])]
        gain = float((rd @ rd - ((rd - rx @ coef) ** 2).sum()) / ((d - d.mean()) ** 2).sum())
    return np.array(r + [gain] + list(coef))


def relabelling_count(rules: np.ndarray) -> int:
    """The number of distinct within-rule relabellings of the members."""
    return math.prod(math.factorial(int(n)) for n in np.unique(rules, return_counts=True)[1])


def association_stats(d: np.ndarray, d_draws: np.ndarray, rules: np.ndarray, log_tokens: np.ndarray,
                      truncated: np.ndarray, perms: int, rng) -> pd.DataFrame:
    """A pairwise distance (pairs in np.triu_indices order; d_draws: pairs x item draws) against the pair's
    |difference in log length| and |difference in truncation|, same rule as a covariate when there are two rules or
    more: partial_stats with intervals over items and one-sided p-values from relabelling members within rules (a
    Mantel test; NaN when fewer than 20 relabellings exist)."""
    a, c = np.triu_indices(len(rules), 1)
    cov = (rules[a] == rules[c]).astype(float)[:, None] if len(set(rules)) > 1 else None

    def predictors(lt, tr):
        return np.column_stack([np.abs(lt[a] - lt[c]), np.abs(tr[a] - tr[c])])

    x = predictors(log_tokens, truncated)
    with np.errstate(invalid="ignore", divide="ignore"):
        obs = partial_stats(d, x, cov)
        null = np.array([partial_stats(d, predictors(log_tokens[p], truncated[p]), cov)
                         for p in within_rule_permutations(rules, perms, rng)])
        boot = np.array([partial_stats(d_draws[:, k], x, cov) for k in range(d_draws.shape[1])])
    testable = relabelling_count(rules) >= 20
    names = ["r_length", "r_truncation", "r2_gain", "slope_length", "slope_truncation"]
    ok = [testable and np.isfinite(obs[j]) and np.isfinite(null[:, j]).any() for j in range(len(names))]
    return pd.DataFrame([{"statistic": n, "value": obs[j], **dict(zip(("lo", "hi"), stats.ci(boot[:, j]))),
                          "null_mean": float(null[np.isfinite(null[:, j]), j].mean()) if ok[j] else np.nan,
                          "p_perm": stats.perm_p(obs[j], null[:, j]) if ok[j] else np.nan}
                         for j, n in enumerate(names)])


def association_table(y, v, items, rules, log_tokens, truncated, b, perms, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """association_stats of the pairwise adjusted functional distance; returns the statistics and the pairs."""
    classes = stats.difficulty_classes(y)
    dp = stats.distances(y, v, classes, stats.ones(y.shape[1]))[0]
    dd = stats.distances(y, v, classes, stats.item_draws(data.strata(items), b, rng))
    a, c = np.triu_indices(len(rules), 1)
    res = association_stats(dp[a, c], dd[:, a, c].T, rules, log_tokens, truncated, perms, rng)
    pairs = pd.DataFrame({"a": a, "b": c, "d2": dp[a, c], "dlog_tokens": np.abs(log_tokens[a] - log_tokens[c]),
                          "dtruncated": np.abs(truncated[a] - truncated[c]), "same_rule": rules[a] == rules[c]})
    return res, pairs


def analyse(run: data.Run, grid: data.Grid, members: list[str], null: data.Grid | None, cap: int, b: int,
            perms: int, rng) -> dict:
    real = grid.real().take(members)
    stats.need_samples(real.x["reward"], "the length decomposition")
    items = real.items
    y, v = stats.cells(real.x["reward"])
    rules = np.array([run.resolve(m)[0].arm for m in members])
    cov = {"all": covariates(real)}
    clean = ~(real.x["truncated"].sum((0, 2)) > 0)
    scopes = [("all", np.ones(len(items), bool), cov["all"]), ("clean", clean, cov["all"]), ("cut", ~clean, cov["all"])]
    for d in sorted(items["domain"].unique()):
        keep = (items["domain"] == d).to_numpy()
        cov[d] = covariates(real.items_where(keep))
        scopes += [(f"all/{d}", keep, cov[d]), (f"clean/{d}", keep & clean, cov[d])]
    mem = pd.DataFrame([{"member": m, "rule": rules[j], "scope": s, "tokens": c["tokens"][j],
                         "log_tokens": c["log_tokens"][j], "truncated": c["truncated"][j],
                         "accuracy": float(np.nanmean(y[j] if s == "all" else y[j, (items["domain"] == s).to_numpy()]))}
                        for s, c in cov.items() for j, m in enumerate(members)])
    out = {"members": mem,
           "interaction": interaction_table(y, v, items, scopes, rules, b, perms, rng),
           "bands": band_table(y, v, items, rules, cov["all"]["log_tokens"], b, perms, rng)}
    out["association"], pairs = association_table(y, v, items, rules, cov["all"]["log_tokens"],
                                                  cov["all"]["truncated"], b, perms, rng)
    out["pairs"] = pairs.assign(a=[members[i] for i in pairs["a"]], b=[members[i] for i in pairs["b"]])
    out["clean_items"] = int(clean.sum())
    if cap > 0 and grid.texts is not None:
        out["authorship"] = divergence.authorship(
            real, members, cap, rng, views=("raw", "prefix", "stripped_prefix", "length"),
            contrasts=(("length_cue_share", "raw", "prefix"), ("presentation_share", "prefix", "stripped_prefix")),
            b=b)
    else:
        out["authorship"] = pd.DataFrame()
    out["null"] = divergence.null_pairs(null.real(), lambda it: stats.item_draws(data.strata(it), b, rng)) \
        if null is not None else pd.DataFrame()
    return out


def report_lines(run, tag, members, res, null_tag) -> list[str]:
    mem = res["members"]
    a = mem[mem["scope"] == "all"]
    inter = res["interaction"]
    lines = [f"length {run.name}, selection {tag}: {len(members)} members, dataset units only; "
             f"{res['clean_items']} of {inter.loc[(inter['items_set'] == 'all') & (inter['estimator'] == 'plain'), 'items'].iloc[0]} "
             "items with no sample of any member cut off; [95% bootstrap over items]",
             f"members' mean length {a['tokens'].min():.0f} to {a['tokens'].max():.0f} tokens, share cut off "
             f"{pct(a['truncated'].min())} to {pct(a['truncated'].max())}%"]
    lines += ["", "member-by-item interaction SD (pp) after removing nested member models from each item's column; "
                  "share = 1 - residual / interaction; length part = what length and truncation remove beyond the rule, "
                  "p from permuting members' length and truncation within rules"]
    rows = []
    for (s, e), g in inter.groupby(["items_set", "estimator"], sort=False):
        full = g.iloc[-1]
        rows.append([s, e, int(g["items"].iloc[0])] + [num(100 * r.sd, 100 * r.sd_lo, 100 * r.sd_hi, 2)
                                                       for r in g.itertuples()]
                    + [pci(full.length_part, full.length_part_lo, full.length_part_hi), f"{full.p_perm:.3f}",
                       pct(full.length_part_null)])
    names = list(inter.loc[(inter["items_set"] == "all") & (inter["estimator"] == "plain"), "model"])
    lines += render(["items", "estimator", "n"] + names + ["length part %", "p", "null %"], rows, 2)
    bd = res["bands"]
    if len(bd):
        r = bd.iloc[0]
        lines += ["", f"members of similar length: adjusted interaction pooled over {r.bands} bands (sizes {r.sizes}) "
                      f"within rules, whose log length spreads {r.length_sd_within_bands:.2f} against {r.length_sd:.2f} "
                      "over all members",
                  f"  similar length {PP2 * r.similar_length:.1f} pp^2 [{PP2 * r.similar_lo:.1f}, {PP2 * r.similar_hi:.1f}],"
                  f" random bands {PP2 * r.random_bands:.1f}, ratio {num(r.ratio, r.ratio_lo, r.ratio_hi)}, "
                  f"p {r.p_perm:.3f} (difficulty classes formed within a band of five are coarser than over all "
                  "members, so only the ratio to random bands of the same sizes is read)"]
    asc = res["association"]
    lines += ["", "pairwise adjusted distance against the pair's difference in log length and in truncation (same rule "
                  "as covariate): partial correlations, R^2 gain of both, slopes in pp^2 per unit; p: members relabelled "
                  "within rules"]
    lines += render(["statistic", "value [95%]", "null mean", "p"],
                    [[r.statistic, num(r.value * (PP2 if r.statistic.startswith("slope") else 1),
                                       r.lo * (PP2 if r.statistic.startswith("slope") else 1),
                                       r.hi * (PP2 if r.statistic.startswith("slope") else 1), 3),
                      num(r.null_mean * (PP2 if r.statistic.startswith("slope") else 1), digits=3), f"{r.p_perm:.3f}"]
                     for r in asc.itertuples()])
    au = res["authorship"]
    if len(au):
        lines += ["", f"authorship of sample 0 (train on half the units, test on the others; first {divergence.PREFIX} "
                      "characters hide length): accuracy %; shares are what the second view loses of the first's "
                      "accuracy beyond chance"]
        lines += [f"  {r.view}: {pci(r.accuracy, r.lo, r.hi)}" + (f" (chance {pct(r.chance)})" if np.isfinite(r.chance)
                                                                   else "") for r in au.itertuples()]
    nl = res["null"]
    lines += [""] + ([f"null ({null_tag}, each adapter's halves, distance pp^2): " + "; ".join(
        f"{r.estimator} {PP2 * r.d2:+.2f} [{PP2 * r.lo:+.2f}, {PP2 * r.hi:+.2f}]"
        for r in nl[nl["adapter"] == "mean"].itertuples()) + " (true value 0)"] if len(nl) else
        ["null: not available (needs --null-tag, an evaluation with four or more samples per item)"])
    lines += ["", "reading: " + summary_sentence(res)]
    return lines


def summary_sentence(res) -> str:
    inter = res["interaction"]

    def full(s):
        g = inter[(inter["items_set"] == s) & (inter["estimator"] == "adjusted")]
        return (g.iloc[0], g.iloc[-1]) if len(g) else (None, None)

    (n0, f0), (n1, f1) = full("all"), full("clean")
    out = (f"the adjusted interaction SD is {100 * n0.sd:.1f} pp on all items; length and truncation remove "
           f"{pct(f0.length_part)}% [{pct(f0.length_part_lo)}, {pct(f0.length_part_hi)}] of its variance beyond the "
           f"update rule (permutation p {f0.p_perm:.3f}), leaving {100 * f0.sd:.1f} pp [{100 * f0.sd_lo:.1f}, "
           f"{100 * f0.sd_hi:.1f}]")
    if n1 is not None:
        out += (f"; on items no member cut off it is {100 * n1.sd:.1f} pp, {pct(f1.length_part)}% of it length, "
                f"leaving {100 * f1.sd:.1f} pp")
    bd = res["bands"]
    if len(bd):
        out += f"; among members of similar length it is {num(bd.iloc[0].ratio)} times that among random bands"
    return out + "."


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--tag", required=True, help="selection evaluation tag")
    ap.add_argument("--null-tag", help="evaluation with >= 4 samples per item whose adapters give the null")
    ap.add_argument("--step", type=int, help="members evaluated at this step (default: the step with most labels)")
    ap.add_argument("--members", help="comma-separated labels instead of --step")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap draws over items")
    ap.add_argument("--perms", type=int, default=1000, help="relabellings of members for the permutation tests")
    ap.add_argument("--classifier-items", type=int, default=400, help="items per half of the units for authorship "
                                                                      "(0: skip)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    run = data.Run(args.run_dir)
    grid = data.load_eval(run, args.tag, keep_texts=args.classifier_items > 0)
    members = divergence.pick_members(run, grid, args.step, args.members.split(",") if args.members else None)
    null_tag = args.null_tag or (args.tag if grid.samples >= 4 else None)
    null = (grid if null_tag == args.tag else data.load_eval(run, null_tag)) if null_tag else None
    res = analyse(run, grid, members, null, args.classifier_items, args.boot, args.perms, np.random.default_rng(0))
    lines = report_lines(run, args.tag, members, res, null_tag)
    res.pop("clean_items")
    write(args.out or Path("results") / run.name / "length", res, lines)


if __name__ == "__main__":
    with threadpool_limits(stats.THREADS):
        main()
