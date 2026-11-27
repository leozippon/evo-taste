"""Distinctness by likelihood: how much better a member predicts its own answers than another member does, who wrote
an answer by likelihood alone, and whether clones stay closer than unrelated members, from the cross-likelihood
table of `python -m evotaste.evaluate deviation --steps S --cross`.

    python -m evotaste.analysis.distinct runs/<name> --step S [--selection DIR] [--boot 1000] [--perms 5000]
                                          [--classifier-items 150] [--out DIR]

The members are every author at step S other than the base model (scaled probes included). For an ordered pair
(author a, scorer b) the margin is sum_i (log p_a(y_ai) - log p_b(y_ai)) / sum_i tokens(y_ai) over a's answers: an
estimate of KL(a || b) per token, zero when b is a's own weights; with b the base model it is a's deviation. The
distance of two members is the mean of their two margins. Identification: the scorer among the members that gives an
answer the highest likelihood is taken as its author (ties share the credit), against chance 1 / members, overall and
by domain; nothing is fitted, so every domain's rate is out of sample. Coherence: whether members are distinct in the
same way on different domains, the correlation over member pairs of the distances measured on each (Mantel test).
The distance is also set against the pairs' difference in answer length (as in the length command). Clone pairs (a
root lineage's members under one update rule, as in heritability) against unrelated members of the same rule: mean
distance, difference with an interval over items, and a permutation test over family labels. Because likelihood can
be dominated by presentation, the report gives beside it the authorship classifier on the same members' probe
answers, raw and stripped of layout and notation (the likelihood tables hold one sum per answer, so their own share
that survives stripping cannot be computed from them). Writes results/<name>/distinct/: margins.csv, distance.csv,
identification.csv, members.csv, claims.csv, coherence.csv, clones.csv, length.csv, authorship.csv and report.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evotaste.analysis import data, divergence, heritability, length, stats
from evotaste.analysis.report import num, pci, pct, render, write


class Cross:
    """A cross-likelihood table as arrays: lp[scorer, author, item] (summed log-probability of the author's answer)
    and tokens[author, item]; subjects in table order with base first; the items' family, domain and unit."""

    def __init__(self, df: pd.DataFrame):
        names = list(dict.fromkeys(df["author"]))
        self.subjects = ["base"] + [s for s in names if s != "base"]
        self.items = (df.drop_duplicates("uid").sort_values("uid")[["uid", "family", "domain", "unit"]]
                      .reset_index(drop=True))
        pos = {s: k for k, s in enumerate(self.subjects)}
        ip = {u: k for k, u in enumerate(self.items["uid"])}
        s, a, i = (df["scorer"].map(pos).to_numpy(), df["author"].map(pos).to_numpy(), df["uid"].map(ip).to_numpy())
        self.lp = np.zeros((len(pos), len(pos), len(ip)))
        self.lp[s, a, i] = df["logprob_sum"].to_numpy()
        self.tokens = np.zeros((len(pos), len(ip)))
        self.tokens[a, i] = df["tokens"].to_numpy()
        # the same over the layout tokens alone (no letter or digit); NaN in tables from before the layout columns
        self.layout = data.has_layout(df)
        self.layout_lp = np.full(self.lp.shape, np.nan)
        self.layout_lp[s, a, i] = df["layout_logprob_sum"].to_numpy()
        self.layout_tokens = np.full(self.tokens.shape, np.nan)
        self.layout_tokens[a, i] = df["layout_tokens"].to_numpy()

    @property
    def members(self) -> list[str]:
        return self.subjects[1:]


def margins(c: Cross, w: np.ndarray, keep: np.ndarray | None = None, layout: bool = False) -> np.ndarray:
    """margin[a, b, draw]: per-token log-likelihood of author a's answers under a minus under b, over the items
    where keep holds, items weighted by w (draws x items). With `layout`, the part of it that the layout tokens
    contribute (still per token of all tokens, so that it adds to the content part to make the margin)."""
    keep = np.ones(len(c.items), bool) if keep is None else keep
    lp = c.layout_lp if layout else c.lp
    own = np.einsum("aai->ai", lp)
    diff = own[:, None, :] - lp.transpose(1, 0, 2)  # [author, scorer, item]
    wk = w * keep[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (diff @ wk.T) / (c.tokens @ wk.T)[:, None, :]


def hits(c: Cross) -> np.ndarray:
    """hit[author, item] for the members: 1 / (number of tied scorers) when the author is among the members that
    give its answer the highest likelihood, else 0 (identical scorers tie, which gives exactly chance)."""
    m = np.arange(1, len(c.subjects))
    score = c.lp[np.ix_(m, m)]  # [scorer, author, item]
    best = score.max(0)
    tied = (score >= best - 1e-9).sum(0)
    own = np.einsum("aai->ai", score)
    return np.where(own >= best - 1e-9, 1.0 / tied, 0.0)


def identification(c: Cross, w: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Identification rate overall and by domain against chance, with intervals over items; and the hits."""
    domains = c.items["domain"].to_numpy()
    hit = hits(c)
    rows = []
    for scope, keep in [("all", np.ones(len(domains), bool))] + [(d, domains == d) for d in sorted(set(domains))]:
        wk = w * keep[None, :]
        rows.append({"scope": scope, "members": len(c.members), "answers": int(keep.sum()) * len(c.members),
                     "rate": hit[:, keep].mean(), **dict(zip(("lo", "hi"), stats.ci((hit.mean(0) @ wk.T) / wk.sum(1)))),
                     "chance": 1 / len(c.members)})
    return pd.DataFrame(rows), hit


def coherence(c: Cross, w: np.ndarray, perms: int, rng) -> pd.DataFrame:
    """Whether members are distinct in the same way on every domain: for each pair of domains, the correlation over
    member pairs of the distances measured on each, with an interval over items and a Mantel p-value (members of one
    domain's matrix relabelled)."""
    domains = c.items["domain"].to_numpy()
    m = np.arange(1, len(c.subjects))
    if len(m) < 4:
        return pd.DataFrame()
    a, b_ = np.triu_indices(len(m), 1)
    dist = {}
    for d in sorted(set(domains)):
        mp = margins(c, np.vstack([np.ones(len(domains)), w]), domains == d)[np.ix_(m, m)]
        dist[d] = ((mp + mp.transpose(1, 0, 2)) / 2)  # member x member x (1 + draws)
    perm = np.array([rng.permutation(len(m)) for _ in range(perms)])
    rows = []
    names = sorted(dist)
    for i, d1 in enumerate(names):
        for d2 in names[i + 1:]:
            x, y = dist[d1][a, b_], dist[d2][a, b_]  # pairs x (1 + draws)
            r = [np.corrcoef(x[:, k], y[:, k])[0, 1] for k in range(x.shape[1])]
            null = np.array([np.corrcoef(x[:, 0], dist[d2][:, :, 0][np.ix_(p, p)][a, b_])[0, 1] for p in perm])
            rows.append({"domains": f"{d1}/{d2}", "items": f"{(domains == d1).sum()}/{(domains == d2).sum()}",
                         "r": r[0], **dict(zip(("lo", "hi"), stats.ci(r[1:]))), "p_mantel": stats.perm_p(r[0], null)})
    return pd.DataFrame(rows)


def claims(c: Cross) -> pd.DataFrame:
    """Share of each author's answers that each member scorer gives the highest likelihood (ties split)."""
    m = np.arange(1, len(c.subjects))
    score = c.lp[np.ix_(m, m)]
    best = score.max(0)
    top = (score >= best - 1e-9)
    share = (top / top.sum(0, keepdims=True)).mean(2)  # scorer x author
    return pd.DataFrame([{"scorer": c.members[s], "author": c.members[a], "share": share[s, a]}
                         for s in range(len(m)) for a in range(len(m))])


def clone_table(run: data.Run, c: Cross, dist_point: np.ndarray, dist_draws: np.ndarray, perms: int, rng) -> pd.DataFrame:
    """Clone pairs against unrelated members of the same rule (heritability's families) among the member probes:
    mean distances, their difference with an interval over items, the ratio, and the permutation p-value."""
    idx = [k for k, s in enumerate(c.members) if s not in run.scaled]
    lins = [run.resolve(c.members[k])[0] for k in idx]
    fam = np.array([heritability.family_of(run, lin) for lin in lins])
    arm = np.array([lin.arm for lin in lins])
    if not any((fam == f).sum() >= 2 for f in fam):
        return pd.DataFrame()
    rows = []
    for g in ["all"] + sorted(set(arm)):
        sel = np.ones(len(idx), bool) if g == "all" else arm == g
        if g == "all" and len(set(arm)) == 1:
            continue
        f, a, ix = fam[sel], arm[sel], np.array(idx)[sel]
        dp = dist_point[np.ix_(ix, ix)]
        clone, unrelated = heritability.means(dp, f, a)
        if not (np.isfinite(clone) and np.isfinite(unrelated)):
            continue
        perm, distinct = stats.relabellings(f, a, rng, perms)
        null = np.array([np.subtract(*heritability.means(dp, p, a)[::-1]) for p in perm])
        dd = dist_draws[np.ix_(ix, ix)]
        boot = np.array([heritability.means(dd[:, :, k], f, a) for k in range(dd.shape[2])])
        rows.append({"rule": g, "families": len(set(f)), "members": len(f),
                     "pairs": int(sum((f[i] == f[j]) for i in range(len(f)) for j in range(i + 1, len(f)))),
                     "clone": clone, **dict(zip(("clone_lo", "clone_hi"), stats.ci(boot[:, 0]))),
                     "unrelated": unrelated, **dict(zip(("unrelated_lo", "unrelated_hi"), stats.ci(boot[:, 1]))),
                     "difference": unrelated - clone, **dict(zip(("diff_lo", "diff_hi"), stats.ci(boot[:, 1] - boot[:, 0]))),
                     "ratio": clone / unrelated, **dict(zip(("ratio_lo", "ratio_hi"), stats.ci(boot[:, 0] / boot[:, 1]))),
                     "p_perm": stats.perm_p(unrelated - clone, null), "relabellings": distinct})
    return pd.DataFrame(rows)


def analyse(run: data.Run, step: int, selection=None, b: int = 1000, perms: int = 5000, cap: int = 150,
            rng=None) -> dict:
    rng = rng or np.random.default_rng(0)
    df, sel = data.cross_table(run, step, selection)
    c = Cross(df)
    if len(c.members) < 2:
        raise SystemExit(f"{sel}/cross-{step}.csv: fewer than two members")
    w = stats.item_draws(data.strata(c.items), b, rng)
    ones = stats.ones(len(c.items))
    rows = []
    scopes = [("all", np.ones(len(c.items), bool))] + [(d, (c.items["domain"] == d).to_numpy())
                                                       for d in sorted(c.items["domain"].unique())]
    for scope, keep in scopes:
        mp, md = margins(c, ones, keep)[..., 0], margins(c, w, keep)
        lp_, ld_ = margins(c, ones, keep, layout=True)[..., 0], margins(c, w, keep, layout=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            share_d = ld_ / md
        for a, author in enumerate(c.subjects):
            for s, scorer in enumerate(c.subjects):
                if a != s:
                    rows.append({"scope": scope, "author": author, "scorer": scorer, "margin": mp[a, s],
                                 **dict(zip(("lo", "hi"), stats.ci(md[a, s]))), "layout_margin": lp_[a, s],
                                 "layout_share": lp_[a, s] / mp[a, s] if mp[a, s] else np.nan,
                                 **dict(zip(("layout_share_lo", "layout_share_hi"), stats.ci(share_d[a, s])))})
    marg = pd.DataFrame(rows)
    mp, md = margins(c, ones)[..., 0], margins(c, w)
    lp_, ld_ = margins(c, ones, layout=True)[..., 0], margins(c, w, layout=True)
    m = np.arange(1, len(c.subjects))
    dist_point = (mp[np.ix_(m, m)] + mp[np.ix_(m, m)].T) / 2
    dist_draws = (md[np.ix_(m, m)] + md[np.ix_(m, m)].transpose(1, 0, 2)) / 2
    lay_point = (lp_[np.ix_(m, m)] + lp_[np.ix_(m, m)].T) / 2
    lay_draws = (ld_[np.ix_(m, m)] + ld_[np.ix_(m, m)].transpose(1, 0, 2)) / 2
    lins = {s: (run.resolve(s)[0] if s not in run.scaled else run.lineages[run.scaled[s]["lineage"]]) for s in c.members}
    a_, b_ = np.triu_indices(len(m), 1)
    ident, hit = identification(c, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        lay_share = lay_draws / dist_draws
        dist = pd.DataFrame({"a": [c.members[i] for i in a_], "b": [c.members[j] for j in b_],
                             "same_rule": [lins[c.members[i]].arm == lins[c.members[j]].arm for i, j in zip(a_, b_)],
                             "distance": dist_point[a_, b_],
                             "lo": [stats.ci(dist_draws[i, j])[0] for i, j in zip(a_, b_)],
                             "hi": [stats.ci(dist_draws[i, j])[1] for i, j in zip(a_, b_)],
                             "layout_share": lay_point[a_, b_] / dist_point[a_, b_],
                             "layout_share_lo": [stats.ci(lay_share[i, j])[0] for i, j in zip(a_, b_)],
                             "layout_share_hi": [stats.ci(lay_share[i, j])[1] for i, j in zip(a_, b_)]})
        others = ~np.eye(len(m), dtype=bool)
        per = pd.DataFrame({
            "member": c.members, "tokens": c.tokens[m].mean(1), "deviation": mp[m, 0],
            "distinctness": [mp[k, m[others[j]]].mean() for j, k in enumerate(m)],
            "identified": hit.mean(1),
            "layout_tokens": c.layout_tokens[m].sum(1) / c.tokens[m].sum(1),
            "deviation_in_layout": lp_[m, 0] / mp[m, 0],
            "distinctness_in_layout": [lp_[k, m[others[j]]].mean() / mp[k, m[others[j]]].mean() for j, k in enumerate(m)]})
    res_layout = {"pairs_in_layout": float(lay_point[a_, b_].sum() / dist_point[a_, b_].sum()),
                  "pairs_in_layout_ci": stats.ci(lay_draws[a_, b_].sum(0) / dist_draws[a_, b_].sum(0))}
    # distance against the pair's difference in answer length, as in the length command
    rules = np.array([lins[s].arm for s in c.members])
    lt = np.log(per["tokens"].to_numpy())
    cut = (c.tokens[m] >= run.cfg.probes.max_tokens).mean(1)
    lens = length.association_stats(dist_point[a_, b_], dist_draws[a_, b_], rules, lt, cut, min(perms, 2000), rng)
    out = {"margins": marg, "distance": dist, "identification": ident, "members": per, "claims": claims(c),
           "coherence": coherence(c, w, min(perms, 2000), rng),
           "clones": clone_table(run, c, dist_point, dist_draws, perms, rng), "length": lens}
    if cap > 0:
        subjects = {s: ((lins[s], step) if s not in run.scaled else (lins[s], run.scaled[s]["step"])) for s in c.members}
        grid = data.probe_grid(run, subjects, keep_texts=True)
        out["authorship"] = divergence.authorship(grid, c.members, cap, rng, b=b)
    else:
        out["authorship"] = pd.DataFrame()
    out["_selection"] = sel
    out["_n"] = (len(c.members), len(c.items))
    out["_layout"] = res_layout if c.layout else None
    return out


def report_lines(run: data.Run, step: int, res: dict) -> list[str]:
    n_members, n_items = res["_n"]
    lines = [f"distinct {run.name}, step {step}: {n_members} members x {n_items} probe items "
             f"(selection {res['_selection'].name}); nats per token [95% bootstrap over items]"]
    per = res["members"]
    lay = res["_layout"]
    lines += ["", "per member: deviation (margin over the base model), distinctness (mean margin over the other members), "
                  "share of its answers identified by likelihood, mean answer tokens; the share of deviation and of "
                  "distinctness that sits in layout tokens (no letter or digit), against their share of the tokens"
              + ("" if lay else " (not in this table: written before the layout columns)")]
    lines += render(["member", "tokens", "deviation", "distinctness", "identified %", "layout tok %", "dev in layout %",
                     "distinct in layout %"],
                    [[r.member, f"{r.tokens:.0f}", num(r.deviation, digits=3), num(r.distinctness, digits=3),
                      pct(r.identified), pct(r.layout_tokens, 0), pct(r.deviation_in_layout, 0),
                      pct(r.distinctness_in_layout, 0)] for r in per.itertuples()])
    if lay:
        lines += [f"  share of all pairs' distance in layout tokens: {pci(lay['pairs_in_layout'], *lay['pairs_in_layout_ci'])}"]
    idt = res["identification"]
    lines += ["", f"identification by likelihood (nothing is fitted, so every domain's rate is out of sample; chance "
                  f"{pct(1 / n_members)}%): " + "; ".join(f"{r.scope} {pci(r.rate, r.lo, r.hi)}" for r in idt.itertuples())]
    co = res["coherence"]
    if len(co):
        lines += ["coherence: correlation over member pairs of the distances measured on two domains (p: members of "
                  "one matrix relabelled): " + "; ".join(f"{r.domains} ({r.items} items) {num(r.r, r.lo, r.hi)}, "
                                                         f"p {r.p_mantel:.3f}" for r in co.itertuples())]
    cl = res["claims"]
    top = cl.groupby("scorer")["share"].mean().sort_values(ascending=False)
    lines += [f"  answers claimed by each scorer (mean over authors, expected {pct(1 / n_members)}%): "
              + ", ".join(f"{s} {pct(v)}" for s, v in top.head(5).items()) + (" ..." if len(top) > 5 else "")]
    dist = res["distance"]
    parts = [f"mean {dist['distance'].mean():.3f}"] + [
        f"{k} {dist.loc[sel, 'distance'].mean():.3f}" for k, sel in (("same rule", dist["same_rule"]),
                                                                     ("across rules", ~dist["same_rule"])) if sel.any()]
    lines += ["", f"distance (mean of the two margins) over {len(dist)} pairs: " + ", ".join(parts)
              + f"; closest {dist.loc[dist['distance'].idxmin(), 'a']}/{dist.loc[dist['distance'].idxmin(), 'b']} "
                f"{dist['distance'].min():.3f}, farthest {dist.loc[dist['distance'].idxmax(), 'a']}/"
                f"{dist.loc[dist['distance'].idxmax(), 'b']} {dist['distance'].max():.3f}"]
    ln = res["length"].set_index("statistic")
    lines += [f"  against the pair's difference in log length: partial r {num(ln.loc['r_length', 'value'], ln.loc['r_length', 'lo'], ln.loc['r_length', 'hi'])} "
              f"(p {ln.loc['r_length', 'p_perm']:.3f}); in the share of answers at the token limit: partial r "
              f"{num(ln.loc['r_truncation', 'value'], ln.loc['r_truncation', 'lo'], ln.loc['r_truncation', 'hi'])} "
              f"(p {ln.loc['r_truncation', 'p_perm']:.3f}); R^2 gain {num(ln.loc['r2_gain', 'value'])} "
              "(p is nan with fewer than 20 within-rule relabellings)"]
    cs = res["clones"]
    if len(cs):
        lines += ["", "clone pairs against unrelated members of the same rule (distance per token; p: permutation over "
                      "family labels, relab = distinct partitions)"]
        lines += render(["rule", "fam", "n", "clone [95%]", "unrelated [95%]", "unrel - clone [95%]", "ratio [95%]",
                         "p", "relab"],
                        [[r.rule, r.families, r.members, num(r.clone, r.clone_lo, r.clone_hi, 3),
                          num(r.unrelated, r.unrelated_lo, r.unrelated_hi, 3), num(r.difference, r.diff_lo, r.diff_hi, 3),
                          num(r.ratio, r.ratio_lo, r.ratio_hi), f"{r.p_perm:.3f}", r.relabellings]
                         for r in cs.itertuples()])
    au = res["authorship"]
    if len(au):
        lines += ["", "authorship classifier on the same members' probe answers (presentation check; train on half "
                      "the units, test on the others)"]
        lines += [f"  {r.view}: {pci(r.accuracy, r.lo, r.hi)}" + (f" (chance {pct(r.chance)})" if np.isfinite(r.chance)
                                                                   else " (share lost by stripping)")
                  for r in au.itertuples()]
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--step", type=int, required=True, help="the step of the cross-likelihood table")
    ap.add_argument("--selection", type=Path, help="a deviation selection (default: the run's only one)")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap draws over items")
    ap.add_argument("--perms", type=int, default=5000, help="relabellings of clone families")
    ap.add_argument("--classifier-items", type=int, default=150, help="probe items per half of the units for the "
                                                                      "authorship classifier (0: skip)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    run = data.Run(args.run_dir)
    res = analyse(run, args.step, args.selection, args.boot, args.perms, args.classifier_items)
    lines = report_lines(run, args.step, res)
    write(args.out or Path("results") / run.name / "distinct",
          {k: v for k, v in res.items() if not k.startswith("_")}, lines)


if __name__ == "__main__":
    with threadpool_limits(stats.THREADS):
        main()
