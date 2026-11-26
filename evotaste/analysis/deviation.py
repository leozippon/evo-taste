"""Deviation from the base model: how much more likely a member finds its own probe answers than the base model
does, in nats per token, from the per-response tables of `python -m evotaste.evaluate deviation`.

    python -m evotaste.analysis.deviation runs/<name> [--selection DIR] [--partial] [--boot 1000] [--out DIR]

For an author (a member's probe at a step, scored with its checkpoint of that step, or a scaled probe) the deviation
is sum(log p_author - log p_base) / sum(tokens) over its answers to the selection's items: an estimate of the
per-token divergence KL(author || base), zero for the base model's weights. It is reported overall and by domain,
per update rule and step (the trajectory; reference draws of the base weights and clones' first probes left out,
as in health), and against length: the base model's typical length on an item is the median over its probes of the
base weights (its prepare-time answers and the step-0 probes of unseeded members); an author's answers are split
into those within it (no longer) and beyond it, with the deviation of each part, and the rank correlation of an
answer's deviation with its length relative to that typical length. --partial reads a selection still being scored
(data.deviation_table). Intervals resample items within unit kinds and domains, the same draws for every author.
Writes results/<name>/deviation/: authors.csv, trajectory.csv, length.csv and report.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from threadpoolctl import threadpool_limits

from evotaste.analysis import data, health, stats
from evotaste.analysis.report import num, render, write


class Table:
    """A deviation selection as arrays author x item: completion tokens and summed log-ratio, with the items' family,
    domain and unit kind, and per author its lineage, step and factor (NaN for a member's own probe)."""

    def __init__(self, run: data.Run, df: pd.DataFrame):
        self.labels = list(dict.fromkeys(df["subject"]))
        self.items = (df.drop_duplicates("uid").sort_values("uid")[["uid", "family", "domain", "unit"]]
                      .reset_index(drop=True))
        a = df["subject"].map({s: i for i, s in enumerate(self.labels)}).to_numpy()
        i = df["uid"].map({u: j for j, u in enumerate(self.items["uid"])}).to_numpy()
        self.tokens, self.delta = np.zeros((len(self.labels), len(self.items))), np.zeros((len(self.labels), len(self.items)))
        self.tokens[a, i], self.delta[a, i] = df["tokens"].to_numpy(), df["delta_sum"].to_numpy()
        first = df.drop_duplicates("subject").set_index("subject")
        self.factor = {s: first.loc[s, "factor"] for s in self.labels}
        self.subject = {}  # label -> (lineage, step) of the probe it stands for
        for s in self.labels:
            lin = run.lineages[first.loc[s, "lineage"]]
            self.subject[s] = (lin, int(first.loc[s, "step"]))

    def rows(self, labels) -> list[int]:
        return [self.labels.index(s) for s in labels]

    def ratio(self, w: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Per author and row of w (draws x items): sum of w_i delta over sum of w_i tokens, over the answers where
        mask (author x item) holds."""
        m = np.ones(self.tokens.shape) if mask is None else mask.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return ((self.delta * m) @ w.T) / ((self.tokens * m) @ w.T)

    def draws(self, b: int, rng) -> np.ndarray:
        return stats.item_draws(data.strata(self.items), b, rng)


def scopes(items: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    return [("all", np.ones(len(items), bool))] + [(d, (items["domain"] == d).to_numpy())
                                                   for d in sorted(items["domain"].unique())]


def typical_length(run: data.Run, uids) -> tuple[np.ndarray, int]:
    """Per item the median completion tokens over every probe of the base model's weights (data.Run.base_weights),
    and the number of such probes."""
    subjects = health.probe_subjects(run)
    records = [run.base_records()] + [run.probe_records(lin, step) for lin, step in subjects.values()
                                      if lin is not None and run.base_weights(lin, step)]
    tokens = np.array([[{r["uid"]: r["completion_tokens"] for r in recs}[u] for u in uids] for recs in records], float)
    return np.median(tokens, 0), len(records)


def spearman_draws(x: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Spearman correlation of x and z over items for every row of the count weights w (an item drawn twice counts
    twice, its copies tied)."""
    out = np.empty(len(w))
    for k, row in enumerate(w.astype(int)):
        idx = np.repeat(np.arange(len(x)), row)
        out[k] = sps.spearmanr(x[idx], z[idx])[0]
    return out


def author_table(t: Table, w: np.ndarray) -> pd.DataFrame:
    rows = []
    for scope, keep in scopes(t.items):
        mask = np.broadcast_to(keep, t.tokens.shape)
        point, draws = t.ratio(stats.ones(len(t.items)), mask)[:, 0], t.ratio(w, mask)
        for a, s in enumerate(t.labels):
            lin, step = t.subject[s]
            rows.append({"author": s, "lineage": lin.id, "step": step, "factor": t.factor[s], "rule": lin.arm,
                         "scope": scope, "answers": int(keep.sum()), "tokens": float(t.tokens[a, keep].mean()),
                         "deviation": point[a], **dict(zip(("lo", "hi"), stats.ci(draws[a])))})
    return pd.DataFrame(rows)


def at_typical(t: Table, typical: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per author and row of w: the intercept and slope of the item-weighted least-squares line of an answer's
    deviation per token on its relative length log2(tokens / typical): the deviation of an answer as long as the
    base model's typical answer to the item, and its change per doubling of length."""
    x = np.log2(t.tokens / typical[None, :])
    y = t.delta / t.tokens
    s0, s1, s2 = w.sum(1)[None, :], x @ w.T, (x ** 2) @ w.T
    sy, sxy = y @ w.T, (x * y) @ w.T
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (s0 * sxy - s1 * sy) / (s0 * s2 - s1 ** 2)
        return (sy - slope * s1) / s0, slope


def length_table(t: Table, typical: np.ndarray, w: np.ndarray, b_rank: int) -> pd.DataFrame:
    """Per author: the share of its answers longer than the base model's typical length on the item, the deviation
    of the answers within it and beyond it, the deviation of an answer of the typical length and the change per
    doubling (at_typical), and the rank correlation of an answer's deviation with its relative length
    log(tokens / typical); intervals over items (the correlation's over the first b_rank draws)."""
    beyond = t.tokens > typical[None, :]
    ones = stats.ones(len(t.items))
    parts = {"within": t.ratio(ones, ~beyond)[:, 0], "beyond": t.ratio(ones, beyond)[:, 0]}
    draws = {"within": t.ratio(w, ~beyond), "beyond": t.ratio(w, beyond)}
    (parts["at_typical"], parts["per_doubling"]) = (v[:, 0] for v in at_typical(t, typical, ones))
    (draws["at_typical"], draws["per_doubling"]) = at_typical(t, typical, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        share_d = (beyond.astype(float) @ w.T) / w.sum(1)
    rel = np.log(t.tokens / typical[None, :])
    per = t.delta / t.tokens
    rows = []
    for a, s in enumerate(t.labels):
        lin, step = t.subject[s]
        rho = sps.spearmanr(per[a], rel[a])[0] if np.ptp(per[a]) > 0 else np.nan
        rho_d = spearman_draws(per[a], rel[a], w[:b_rank]) if np.isfinite(rho) else np.full(1, np.nan)
        rows.append({"author": s, "lineage": lin.id, "step": step, "factor": t.factor[s], "rule": lin.arm,
                     "share_beyond": float(beyond[a].mean()), **dict(zip(("share_lo", "share_hi"), stats.ci(share_d[a]))),
                     **{k2: v for k in parts for k2, v in ((k, parts[k][a]), (f"{k}_lo", stats.ci(draws[k][a])[0]),
                                                          (f"{k}_hi", stats.ci(draws[k][a])[1]))},
                     "rho": rho, **dict(zip(("rho_lo", "rho_hi"), stats.ci(rho_d)))})
    return pd.DataFrame(rows)


def trajectory_members(run: data.Run, t: Table) -> list[str]:
    """The member probes of the trajectory: no scaled probe, no draw of the base weights, no clone's first probe."""
    return [s for s in t.labels if np.isnan(t.factor[s]) and not run.base_weights(*t.subject[s])
            and not run.restart(*t.subject[s])]


def trajectory_table(run: data.Run, t: Table, typical: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    """Per update rule and step, the mean over member probes of the deviation, the share of answers longer than the
    base model's typical length, and the deviation at that length (at_typical), with intervals from the same item
    draws for every member."""
    ones = stats.ones(len(t.items))
    beyond = (t.tokens > typical[None, :]).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = {"deviation": (t.ratio(ones)[:, 0], t.ratio(w)),
                "share_beyond": (beyond.mean(1), (beyond @ w.T) / w.sum(1)),
                "at_typical": (at_typical(t, typical, ones)[0][:, 0], at_typical(t, typical, w)[0])}
    groups = {}
    for s in trajectory_members(run, t):
        lin, step = t.subject[s]
        groups.setdefault((lin.arm, step), []).append(t.labels.index(s))
    rows = []
    for (rule, step), idx in sorted(groups.items()):
        row = {"rule": rule, "step": step, "members": len(idx), "tokens": float(t.tokens[idx].mean())}
        for k, (p, d) in vals.items():
            row[k] = float(np.nanmean(p[idx]))
            row[f"{k}_lo"], row[f"{k}_hi"] = stats.ci(np.nanmean(d[idx], 0))
        rows.append(row)
    return pd.DataFrame(rows)


def analyse(run: data.Run, selection=None, partial: bool = False, b: int = 1000, rng=None) -> dict:
    rng = rng or np.random.default_rng(0)
    df, sel = data.deviation_table(run, selection, partial)
    t = Table(run, df)
    w = t.draws(b, rng)
    typical, n_typical = typical_length(run, t.items["uid"])
    base = [s for s in t.labels if run.base_weights(*t.subject[s])]
    return {"table": t, "selection": sel, "partial": partial, "typical_draws": n_typical,
            "null": float(np.abs(t.delta[t.rows(base)]).max()) if base else np.nan, "null_authors": base,
            "authors": author_table(t, w), "trajectory": trajectory_table(run, t, typical, w),
            "length": length_table(t, typical, w, min(b, 200))}


def report_lines(run: data.Run, res: dict) -> list[str]:
    t = res["table"]
    lines = [f"deviation {run.name}: selection {res['selection'].name}{' (partial: finished authors only)' if res['partial'] else ''}, "
             f"{len(t.labels)} authors x {len(t.items)} probe items; nats per token [95% bootstrap over items]",
             f"null: the {len(res['null_authors'])} probes of the base weights among the authors have "
             + (f"|log-ratio| at most {res['null']:.2g} (true value 0)" if res["null_authors"] else "none")]
    tr = res["trajectory"]
    if len(tr):
        lines += ["", "per update rule and step, means over member probes: deviation; share of answers longer than the "
                      "base model's typical length on the item (median of its "
                      f"{res['typical_draws']} probes of the base weights); deviation of an answer of that length "
                      "(least-squares line of an answer's deviation on log2 relative length)"]
        lines += render(["rule", "step", "n", "tokens", "deviation [95%]", "longer % [95%]", "at typical length [95%]"],
                        [[r.rule, r.step, r.members, f"{r.tokens:.0f}", num(r.deviation, r.deviation_lo, r.deviation_hi, 3),
                          num(100 * r.share_beyond, 100 * r.share_beyond_lo, 100 * r.share_beyond_hi, 0),
                          num(r.at_typical, r.at_typical_lo, r.at_typical_hi, 3)] for r in tr.itertuples()], 1)
    ln, au = res["length"], res["authors"]
    allscope = au[au["scope"] == "all"].set_index("author")
    lines += ["", "per author: deviation overall and by domain; against length: share of answers longer than the "
                  "typical length, deviation of the answers within it and beyond it, at the typical length and per "
                  "doubling, and Spearman rho of an answer's deviation with its relative length"]
    doms = [s for s in dict.fromkeys(au["scope"]) if s != "all"]
    cell = {(r.author, r.scope): r for r in au.itertuples()}
    rows = []
    for r in ln.itertuples():
        rows.append([r.author, f"{allscope.loc[r.author, 'tokens']:.0f}",
                     num(cell[(r.author, 'all')].deviation, cell[(r.author, 'all')].lo, cell[(r.author, 'all')].hi, 3)]
                    + [num(cell[(r.author, d)].deviation, digits=3) for d in doms]
                    + [f"{100 * r.share_beyond:.0f}", num(r.within, digits=3), num(r.beyond, digits=3),
                       num(r.at_typical, r.at_typical_lo, r.at_typical_hi, 3), num(r.per_doubling, digits=3),
                       num(r.rho, r.rho_lo, r.rho_hi)])
    lines += render(["author", "tokens", "deviation [95%]"] + doms + ["longer %", "within", "beyond",
                                                                      "at typical [95%]", "per x2", "rho [95%]"], rows)
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--selection", type=Path, help="a deviation selection (default: the run's only one)")
    ap.add_argument("--partial", action="store_true", help="read the per-author results of a selection still being "
                                                            "scored")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap draws over items")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    run = data.Run(args.run_dir)
    res = analyse(run, args.selection, args.partial, args.boot)
    write(args.out or Path("results") / run.name / "deviation",
          {k: res[k] for k in ("authors", "trajectory", "length")}, report_lines(run, res))


if __name__ == "__main__":
    with threadpool_limits(stats.THREADS):
        main()
