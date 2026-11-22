"""Analysis of final evaluations: accuracy per adapter, spread across the population, the outlier against its
comparators, the winner's curse, the scaffold effect and compression.

    python -m evotaste.analysis.harvest runs/<name> --tag T [--train-tag T2] [--blind blind1|blind2 --blind-tag T3]
                                                    [--out DIR]

Reads runs/<name>/eval/selection/T (and eval/train/T2); a blind vault only with --blind, and only when the run's
harvest manifest exists. Prints a compact summary and writes CSV/JSON to results/<name>/harvest/. Every statistic is
defined in docs/dev/ANALYSIS_NOTES.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from evotaste.analysis.report import render
from evotaste.analysis.rundata import VAULTS, Evaluation, load_evaluation, manifest_path, member_specs
from evotaste.records import read_json

B = 2000  # bootstrap draws
PERMUTATIONS = 5000
BUDGETS = (64, 128, 256, 384, 512)  # token budgets of the compression table, besides the full limit
ROBUST = 1.4826  # MAD -> SD of a normal distribution


# -- per-suite accuracy and its bootstrap -------------------------------------------------------------------------

class Suite:
    """An evaluation with each adapter's accuracy per family and source, its macro accuracy (mean over procedural
    families), and B paired bootstrap draws of both: instances resampled with replacement within each family or
    source (all samples of an instance together), with the same resampled instances for every adapter."""

    def __init__(self, ev: Evaluation, seed: int):
        self.ev = ev
        self.name = ev.suite
        self.y = ev.reward.mean(-1)  # adapter x instance
        self.pos = {a: i for i, a in enumerate(ev.adapters)}
        rng = np.random.default_rng(seed)
        self.groups = {f: np.flatnonzero(ev.families == f) for f in dict.fromkeys(ev.families)}
        self.procedural = [f for f, idx in self.groups.items() if ev.procedural[idx[0]]]
        self.sources = [f for f in self.groups if f not in self.procedural]
        self.weights = {f: rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=B)
                        for f, idx in self.groups.items()}
        self.acc, self.boot = self.group_stats(self.y)
        self.macro = np.mean([self.acc[f] for f in self.procedural], axis=0)
        self.macro_boot = np.mean([self.boot[f] for f in self.procedural], axis=0)  # adapter x draw

    def group_stats(self, y: np.ndarray):
        """Per family/source: mean of y over its instances and its bootstrap draws (adapter x draw)."""
        return ({f: y[:, idx].mean(1) for f, idx in self.groups.items()},
                {f: y[:, idx] @ self.weights[f].T / len(idx) for f, idx in self.groups.items()})

    def macro_of(self, y: np.ndarray) -> np.ndarray:
        return np.mean([y[:, idx].mean(1) for f, idx in self.groups.items() if f in self.procedural], axis=0)

    def noise_var(self, labels) -> np.ndarray:
        return np.array([self.macro_boot[self.pos[a]].var(ddof=1) for a in labels])


def ci(draws: np.ndarray) -> tuple[float, float]:
    """95% percentile interval of the defined (non-NaN) draws."""
    draws = draws[~np.isnan(draws)]
    if not len(draws):
        return np.nan, np.nan
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


# -- population statistics ----------------------------------------------------------------------------------------

def excess(x: np.ndarray, noise_var: np.ndarray) -> float:
    """Between-member variance beyond evaluation noise: sample variance of the macros minus the mean noise variance."""
    return float(np.var(x, ddof=1) - noise_var.mean())


def spread(x: np.ndarray, noise_var: np.ndarray, rng) -> dict:
    """Spread of members' macro accuracy; the excess-variance interval resamples members (with their noise)."""
    n, med = len(x), float(np.median(x))
    rsd = ROBUST * float(np.median(np.abs(x - med)))
    out = {"n": n, "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if n > 1 else np.nan, "median": med,
           "robust_sd": rsd, "skew": float(stats.skew(x, bias=False)) if n > 2 else np.nan,
           "top_gap_rsd": (float(x.max()) - med) / rsd if rsd > 0 else np.nan,
           "noise_sd": float(np.sqrt(noise_var.mean())), "excess_var": np.nan, "excess_lo": np.nan, "excess_hi": np.nan}
    if n > 1:
        idx = rng.integers(0, n, (B, n))
        boot = x[idx].var(axis=1, ddof=1) - noise_var[idx].mean(axis=1)
        out["excess_var"] = excess(x, noise_var)
        out["excess_lo"], out["excess_hi"] = ci(boot)
    return out


def compare_excess(xa, va, xb, vb, rng) -> dict:
    """excess(a) - excess(b): interval resampling members within each arm, permutation p over arm labels."""
    d = excess(xa, va) - excess(xb, vb)
    ia, ib = rng.integers(0, len(xa), (B, len(xa))), rng.integers(0, len(xb), (B, len(xb)))
    boot = (xa[ia].var(1, ddof=1) - va[ia].mean(1)) - (xb[ib].var(1, ddof=1) - vb[ib].mean(1))
    x, v, na = np.concatenate([xa, xb]), np.concatenate([va, vb]), len(xa)
    perm = np.array([rng.permutation(len(x)) for _ in range(PERMUTATIONS)])
    dp = (x[perm[:, :na]].var(1, ddof=1) - v[perm[:, :na]].mean(1)) - (x[perm[:, na:]].var(1, ddof=1)
                                                                        - v[perm[:, na:]].mean(1))
    lo, hi = ci(boot)
    return {"diff": d, "lo": lo, "hi": hi, "p_perm": float((1 + np.sum(np.abs(dp) >= abs(d))) / (PERMUTATIONS + 1))}


def rowcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of corresponding rows."""
    a, b = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))


def correlation(x: np.ndarray, y: np.ndarray, x_draws: np.ndarray, y_draws: np.ndarray, rng, spearman: bool) -> dict:
    """Correlation across members with an interval from resampling members, one from the evaluation-noise draws
    (member x draw arrays), and a two-sided permutation p-value."""
    rank = (lambda m: stats.rankdata(m, axis=-1)) if spearman else (lambda m: m)
    r = float(rowcorr(rank(x), rank(y)))
    idx = rng.integers(0, len(x), (B, len(x)))
    perm = rng.permuted(np.tile(y, (PERMUTATIONS, 1)), axis=1)
    rp = rowcorr(np.tile(rank(x), (PERMUTATIONS, 1)), rank(perm))
    m_lo, m_hi = ci(rowcorr(rank(x[idx]), rank(y[idx])))
    n_lo, n_hi = ci(rowcorr(rank(x_draws.T), rank(y_draws.T)))
    return {"r": r, "members_lo": m_lo, "members_hi": m_hi, "noise_lo": n_lo, "noise_hi": n_hi,
            "p_perm": float((1 + np.sum(np.abs(rp) >= abs(r))) / (PERMUTATIONS + 1))}


def one_way(values: np.ndarray, labels: np.ndarray, rng) -> dict:
    """One-way random-effects decomposition: between = (MSB - MSW) / n0 (may be negative), within = MSW,
    ICC = between / (between + within); permutation p of F = MSB / MSW over shuffled labels."""
    def f_stat(lab):
        groups = [values[lab == g] for g in np.unique(lab)]
        n, k = len(values), len(groups)
        grand = values.mean()
        msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
        msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n - k)
        n0 = (n - sum(len(g) ** 2 for g in groups) / n) / (k - 1)
        return msb, msw, n0

    msb, msw, n0 = f_stat(labels)
    between = (msb - msw) / n0
    f = msb / msw if msw > 0 else np.inf
    fp = []
    for _ in range(PERMUTATIONS):
        b_, w_, _ = f_stat(rng.permutation(labels))
        fp.append(b_ / w_ if w_ > 0 else np.inf)
    return {"between": between, "within": msw, "icc": between / (between + msw) if between + msw > 0 else np.nan,
            "F": f, "p_perm": float((1 + np.sum(np.array(fp) >= f)) / (PERMUTATIONS + 1))}


# -- analysis -----------------------------------------------------------------------------------------------------

def adapter_roles(adapters, specs: dict[str, dict]) -> dict[str, dict]:
    """label -> {role: base | member | baseline | checkpoint, arm, bias}; lineage ids map to their slot's spec."""
    out = {}
    for a in adapters:
        if a == "base":
            out[a] = {"role": "base", "arm": "", "bias": "base"}
            continue
        lineage, _, step = a.partition("@")
        spec = specs.get(lineage.split("~")[0])
        if spec is None:
            raise ValueError(f"adapter {a!r} is neither base nor a lineage of a configured slot")
        out[a] = {"role": "checkpoint" if step else spec["role"], "arm": spec["arm"],
                  "bias": spec["scaffold"] or spec["arm"]}
    return out


def pick_outlier(name: str, sel: Suite, members: list[str]) -> tuple[str, list[str], str]:
    """The harvested outlier and top three from the manifest, or provisionally the top members by selection macro."""
    path = manifest_path(name)
    if path.exists():
        m = read_json(path)
        return m["outlier"], m["top3"], f"harvested ({path}, tag {m['tag']})"
    ranked = sorted(members, key=lambda a: -sel.macro[sel.pos[a]])
    return ranked[0], ranked[:3], "PROVISIONAL: no harvest manifest; top member by selection macro on this tag"


def analyse(run_dir: Path, tag: str, train_tag: str | None, blind: str | None, blind_tag: str | None) -> dict:
    name, specs = member_specs(run_dir)
    suites = {"selection": Suite(load_evaluation(run_dir, "selection", tag), seed=1)}
    if train_tag:
        suites["train"] = Suite(load_evaluation(run_dir, "train", train_tag), seed=2)
    if blind:
        suites[blind] = Suite(load_evaluation(run_dir, blind, blind_tag, blind=True), seed=3)
    roles = adapter_roles(sorted({a for s in suites.values() for a in s.ev.adapters}), specs)
    sel = suites["selection"]
    members = [a for a in sel.ev.adapters if roles[a]["role"] == "member"]
    if not members:
        raise SystemExit("no population member in the selection evaluation")
    outlier, top3, outlier_source = pick_outlier(name, sel, members)
    rng = np.random.default_rng(0)
    res = {"run": name, "outlier": outlier, "top3": top3, "outlier_source": outlier_source, "roles": roles,
           "suites": {k: {"tag": s.ev.tag, "per_family": s.ev.per_family, "samples": s.ev.samples,
                          "instances": len(s.ev.uids), "procedural_families": len(s.procedural),
                          "sources": s.sources, "dropped": list(s.ev.dropped)} for k, s in suites.items()}}

    # (a) accuracy per adapter.
    rows = []
    for k, s in suites.items():
        for a in s.ev.adapters:
            i = s.pos[a]
            rows.append({"suite": k, "adapter": a, **roles[a], "metric": "macro", "value": s.macro[i],
                         **dict(zip(("lo", "hi"), ci(s.macro_boot[i])))})
            rows += [{"suite": k, "adapter": a, **roles[a], "metric": src, "value": s.acc[src][i],
                      **dict(zip(("lo", "hi"), ci(s.boot[src][i])))} for src in s.sources]
    res["adapters"] = pd.DataFrame(rows)

    # (b) spread across members, overall and per arm; excess variance compared between arms.
    rows, arm_rows = [], []
    for k in ["selection"] + [v for v in VAULTS if v in suites]:
        s = suites[k]
        mem = [a for a in s.ev.adapters if roles[a]["role"] == "member"]
        groups = {"all": mem, **{arm: [a for a in mem if roles[a]["arm"] == arm]
                                 for arm in dict.fromkeys(roles[a]["arm"] for a in mem)}}
        for g, labels in groups.items():
            rows.append({"suite": k, "group": g, **spread(s.macro[[s.pos[a] for a in labels]], s.noise_var(labels), rng)})
        arms = [g for g in groups if g != "all" and len(groups[g]) >= 2]
        for i, ga in enumerate(arms):
            for gb in arms[i + 1:]:
                xa, xb = (s.macro[[s.pos[a] for a in groups[g]]] for g in (ga, gb))
                arm_rows.append({"suite": k, "arm_a": ga, "arm_b": gb, "n_a": len(xa), "n_b": len(xb),
                                 **compare_excess(xa, s.noise_var(groups[ga]), xb, s.noise_var(groups[gb]), rng)})
    res["spread"], res["arms"] = pd.DataFrame(rows), pd.DataFrame(arm_rows)

    # (c) the outlier against base, baselines, the median member and the population mean, on every suite.
    by_sel = sorted(members, key=lambda a: sel.macro[sel.pos[a]])
    median_member = by_sel[(len(by_sel) - 1) // 2]
    comparators = ([("base", "base")] + [(a, "baseline") for a in sel.ev.adapters if roles[a]["role"] == "baseline"]
                   + [(median_member, "median member"), ("mean", "population mean")])
    rows = []
    for k, s in suites.items():
        if outlier not in s.pos:
            continue
        mem = [s.pos[a] for a in s.ev.adapters if roles[a]["role"] == "member"]
        for comp, kind in comparators:
            if comp == outlier:  # the median member can be the outlier itself
                continue
            if comp == "mean":
                m, mb = s.macro[mem].mean(), s.macro_boot[mem].mean(0)
            elif comp in s.pos:
                m, mb = s.macro[s.pos[comp]], s.macro_boot[s.pos[comp]]
            else:
                continue
            d = s.macro_boot[s.pos[outlier]] - mb
            rows.append({"suite": k, "outlier": outlier, "comparator": comp, "kind": kind,
                         "diff": s.macro[s.pos[outlier]] - m, **dict(zip(("lo", "hi"), ci(d)))})
    res["comparisons"] = pd.DataFrame(rows)
    res["median_member"] = median_member

    # (d) winner's curse on the blind vault.
    if blind:
        res["winners_curse"] = winners_curse(sel, suites[blind], suites.get("train"), roles, outlier, top3, rng)

    # (e) scaffold effect among scaffold-arm members.
    rows = []
    for k in ["selection"] + [v for v in VAULTS if v in suites]:
        s = suites[k]
        mem = [a for a in s.ev.adapters if roles[a]["role"] == "member" and roles[a]["arm"] == "scaffold"]
        counts = pd.Series([roles[a]["bias"] for a in mem]).value_counts()
        usable = [a for a in mem if counts[roles[a]["bias"]] >= 2]
        row = {"suite": k, "scaffolds": " ".join(f"{b}({n})" for b, n in counts.items())}
        if counts[counts >= 2].size < 2:
            rows.append({**row, "note": "not supported: needs two scaffolds with two members each"})
            continue
        x = s.macro[[s.pos[a] for a in usable]]
        rows.append({**row, "members": len(usable), "groups": int(counts[counts >= 2].size),
                     "noise_var": float(s.noise_var(usable).mean()),
                     **one_way(x, np.array([roles[a]["bias"] for a in usable]), rng)})
    res["scaffold"] = pd.DataFrame(rows)

    # Compression: tokens and accuracy under token budgets.
    rows = []
    for k, s in suites.items():
        ev = s.ev
        for a in ev.adapters:
            i = s.pos[a]
            correct = ev.reward[i] == 1
            row = {"suite": k, "adapter": a, "tokens_correct": float(ev.tokens[i][correct].mean()) if correct.any()
                   else np.nan, "tokens_all": float(ev.tokens[i].mean())}
            for b in BUDGETS:
                row[f"acc_{b}"] = float(s.macro_of((ev.reward[i:i + 1] * (ev.tokens[i:i + 1] <= b)).mean(-1))[0])
            row["acc_full"] = float(s.macro[i])
            rows.append(row)
    res["compression"] = pd.DataFrame(rows)
    return res


def winners_curse(sel: Suite, bl: Suite, train: Suite | None, roles, outlier, top3, rng) -> dict:
    mem = [a for a in sel.ev.adapters if roles[a]["role"] == "member" and a in bl.pos]
    out = {"members": len(mem)}
    if len(mem) < 3 or outlier not in mem:
        return {**out, "note": "needs at least three members and the outlier on both suites"}
    si, bi = [sel.pos[a] for a in mem], [bl.pos[a] for a in mem]
    x, y, xd, yd = sel.macro[si], bl.macro[bi], sel.macro_boot[si], bl.macro_boot[bi]
    out["spearman"] = correlation(x, y, xd, yd, rng, spearman=True)
    out["pearson"] = correlation(x, y, xd, yd, rng, spearman=False)
    for key, chosen in (("retained_outlier", [outlier]), ("retained_top3", [a for a in top3 if a in mem])):
        c = [mem.index(a) for a in chosen]
        den = x[c].mean() - x.mean()  # zero when the chosen lineages are as good as the mean (e.g. all members)
        r = (y[c].mean() - y.mean()) / den if den != 0 else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            draws = (yd[c].mean(0) - yd.mean(0)) / (xd[c].mean(0) - xd.mean(0))
        undefined = ~np.isfinite(draws)  # draws whose selection differential is zero
        draws[undefined] = np.nan
        out[key] = {"value": float(r), **dict(zip(("lo", "hi"), ci(draws))), "lineages": chosen,
                    "undefined_draws": int(undefined.sum())}
    out["outlier_blind_rank"] = float(stats.rankdata(-y)[mem.index(outlier)])
    if train is not None and all(a in train.pos for a in mem):
        t = train.macro[[train.pos[a] for a in mem]]
        r_sb, r_st, r_bt = (stats.spearmanr(u, v)[0] for u, v in ((x, y), (x, t), (y, t)))
        out["partial_spearman_given_train"] = float((r_sb - r_st * r_bt) / np.sqrt((1 - r_st ** 2) * (1 - r_bt ** 2)))
    return out


# -- terminal -----------------------------------------------------------------------------------------------------

def p(x) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def pci(v, lo, hi) -> str:
    return f"{100 * v:.1f} [{100 * lo:.0f},{100 * hi:.0f}]"


def summary_lines(res: dict) -> list[str]:
    lines = [f"harvest {res['run']}: " + "; ".join(
        f"{k} {v['tag']} ({v['per_family']}/family x {v['samples']}, {v['procedural_families']} families"
        + (f", sources {' '.join(v['sources'])}" if v["sources"] else "") + ")" for k, v in res["suites"].items()),
        f"outlier {res['outlier']} ({res['outlier_source']}); top3 {', '.join(res['top3'])}; rates in %, "
        "[95% bootstrap CI]"]
    for k, v in res["suites"].items():
        if v["dropped"]:
            lines.append(f"  {k}: {len(v['dropped'])} instances dropped (question differs between adapters)")
    a = res["adapters"]
    suites = list(res["suites"])
    cols = [(k, "macro") for k in suites] + [(k, src) for k in suites for src in res["suites"][k]["sources"]]
    cell = {(r.suite, r.adapter, r.metric): r for r in a.itertuples()}
    rows = []
    for label, info in res["roles"].items():
        row = [label, info["bias"] if info["role"] == "member" else info["role"]]
        for k, m in cols:
            r = cell.get((k, label, m))
            row.append("" if r is None else pci(r.value, r.lo, r.hi) if m == "macro" else p(r.value))
        rows.append(row)
    lines += [""] + render(["adapter", "role"] + [f"{k} {m}" for k, m in cols], rows, left=2)

    lines += ["", "spread of members' macro (excess var = variance - mean noise variance, in pp^2; CI resamples members)"]
    lines += render(["suite", "group", "n", "mean", "sd", "median", "rSD", "skew", "top gap/rSD", "noise sd",
                     "excess var [CI]"],
                    [[r.suite, r.group, r.n, p(r.mean), p(r.sd), p(r.median), p(r.robust_sd), f"{r.skew:.2f}",
                      f"{r.top_gap_rsd:.2f}", p(r.noise_sd),
                      "-" if np.isnan(r.excess_var) else
                      f"{1e4 * r.excess_var:.1f} [{1e4 * r.excess_lo:.1f},{1e4 * r.excess_hi:.1f}]"]
                     for r in res["spread"].itertuples()], left=2)
    for r in res["arms"].itertuples():
        lines.append(f"  {r.suite}: excess var {r.arm_a} - {r.arm_b} = {1e4 * r.diff:.1f} pp^2 "
                     f"[{1e4 * r.lo:.1f},{1e4 * r.hi:.1f}], permutation p {r.p_perm:.3f} (n {r.n_a}/{r.n_b})")

    lines += ["", f"outlier {res['outlier']} minus comparator, macro points [paired bootstrap CI]"
                  " (on selection the outlier was chosen on these instances)"]
    c = res["comparisons"]
    if c.empty:
        lines.append("  outlier not evaluated")
    else:
        lines += render(["suite", "comparator", "kind", "diff [CI]"],
                        [[r.suite, r.comparator, r.kind, f"{100 * r.diff:+.1f} [{100 * r.lo:+.1f},{100 * r.hi:+.1f}]"]
                         for r in c.itertuples()], left=3)

    if "winners_curse" in res:
        w = res["winners_curse"]
        lines += ["", f"winner's curse (members on both suites: {w['members']})"]
        if "note" in w:
            lines.append(f"  {w['note']}")
        else:
            for key in ("spearman", "pearson"):
                r = w[key]
                lines.append(f"  {key} {r['r']:+.2f}: members CI [{r['members_lo']:+.2f},{r['members_hi']:+.2f}], "
                             f"noise CI [{r['noise_lo']:+.2f},{r['noise_hi']:+.2f}], permutation p {r['p_perm']:.3f}")
            for key in ("retained_outlier", "retained_top3"):
                r = w[key]
                lines.append(f"  retained fraction ({key[9:]}): {r['value']:.2f} [{r['lo']:.2f},{r['hi']:.2f}]")
            lines.append(f"  outlier rank on blind: {w['outlier_blind_rank']:g} of {w['members']}"
                         + (f"; partial Spearman given train macro: {w['partial_spearman_given_train']:+.2f}"
                            if "partial_spearman_given_train" in w else ""))

    lines += ["", "scaffold effect among scaffold-arm members (one-way random effects on macro)"]
    for r in res["scaffold"].itertuples():
        if isinstance(getattr(r, "note", None), str):
            lines.append(f"  {r.suite}: {r.note}; have {r.scaffolds or 'none'}")
        else:
            lines.append(f"  {r.suite}: {r.scaffolds}: between {1e4 * r.between:.1f} within {1e4 * r.within:.1f} pp^2 "
                         f"(noise {1e4 * r.noise_var:.1f}), ICC {r.icc:.2f}, F {r.F:.2f}, permutation p {r.p_perm:.3f}")

    lines += ["", "compression: mean tokens of correct / all samples, macro accuracy under token budgets"]
    comp = res["compression"]
    lines += render(["suite", "adapter", "tok ok", "tok all"] + [f"@{b}" for b in BUDGETS] + ["full"],
                    [[r["suite"], r["adapter"], f"{r['tokens_correct']:.0f}", f"{r['tokens_all']:.0f}"]
                     + [p(r[f"acc_{b}"]) for b in BUDGETS] + [p(r["acc_full"])] for _, r in comp.iterrows()], left=2)
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--tag", required=True, help="selection evaluation tag")
    ap.add_argument("--train-tag", help="train-suite evaluation tag (in-distribution)")
    ap.add_argument("--blind", choices=VAULTS, help="also read this blind vault (needs the harvest manifest)")
    ap.add_argument("--blind-tag", help="evaluation tag on the blind vault")
    ap.add_argument("--out", help="output directory (default results/<name>/harvest)")
    args = ap.parse_args(argv)
    if bool(args.blind) != bool(args.blind_tag):
        raise SystemExit("--blind and --blind-tag go together")
    res = analyse(args.run_dir, args.tag, args.train_tag, args.blind, args.blind_tag)
    out = Path(args.out or f"results/{res['run']}/harvest")
    out.mkdir(parents=True, exist_ok=True)
    for key in ("adapters", "spread", "arms", "comparisons", "scaffold", "compression"):
        res[key].to_csv(out / f"{key}.csv", index=False)
    meta = {k: res[k] for k in ("run", "outlier", "top3", "outlier_source", "median_member", "roles", "suites")}
    meta["winners_curse"] = res.get("winners_curse")
    (out / "summary.json").write_text(json.dumps(meta, indent=1, default=float) + "\n")
    print("\n".join(summary_lines(res)))
    print(f"\nCSV/JSON: {out}/")


if __name__ == "__main__":
    main()
