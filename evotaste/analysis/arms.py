"""Arms of an amplification pilot: members grouped by the mechanism they train with, compared at every probe step.

    python -m evotaste.analysis.arms runs/<name> [--census results/census/<c>] [--deviation DIR] [--partial]
                                      [--boot 1000] [--classifier-items 150] [--out DIR]

Arms. When the run has members with a taste-shaped reward (pilot E, clones appended to pilot C), the arms are those
members only: own (strength > 0), neu (0) and rev (< 0), and every comparison is paired by parent with the neutral
clone as reference. Otherwise the arms are the update rules (data.rule_name) of every lineage that trains, as in
pilot D (relative-only, exaggeration, practice). Tables:

- history.csv: per arm and step, mean, minimum and maximum over members of the training history: reward,
  completion tokens, truncated share, informative groups, each term's rollouts, tokens and loss, and every field
  of the taste-shaped reward the history holds (fields.csv lists the expected ones that are absent).
- probes.csv: per arm and probe step, the mean over members of (member - reference) of health's metrics (the
  reference: the probes of the base weights), intervals over items; with --census by stratum too.
- pairs.csv (taste arms): per probe step, own - neu and rev - neu averaged over parents, for the same metrics and
  the deviation; at the clone step the three clones' probes are draws of one checkpoint, a null.
- deviation.csv: per arm and step the mean deviation and the deviation at the base model's typical length
  (evotaste.analysis.deviation), from the run's deviation selection or --deviation.
- practice.csv, practice_common.csv: each practising member's source shares at every step, and the share of the
  movement of log shares (from the shared stream's defaults) that is common to all practising members, against
  1 / members, its expectation when members move independently.
- authorship.csv: per probe step and arm, the classifier among the arm's members on their probe answers, raw and
  stripped of layout and notation, and the share lost by stripping.

Writes results/<name>/arms/ and report.txt. Definitions: docs/dev/ANALYSIS_NOTES.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evotaste.analysis import data, deviation, divergence, health, stats
from evotaste.analysis.report import num, pci, pct, render, write

HISTORY = ("reward", "completion_tokens", "truncated", "groups", "adv_sequences", "adv_tokens", "adv_loss",
           "si_sequences", "si_tokens", "si_loss", "ex_sequences", "ex_tokens", "ex_loss")
# What WP-E3 logs per step for the taste-shaped reward (docs/dev/ROUND2_SPEC.md); read under any key with "taste".
TASTE_EXPECTED = ("shaped rollouts", "shaped groups", "mean of s", "SD of s", "sigma", "corr(s, length)",
                  "mean |advantage| from shaping", "mean |advantage| from correctness")
PROBE_METRICS = ("accuracy", "near", "far", "mastered", "frontier", "heldout_complied", "truncated", "tokens")
TASTE_ARMS = ("own", "neu", "rev")


def arms_of(run: data.Run) -> tuple[dict[str, str], bool]:
    """lineage id -> arm of every committed lineage that trains, and whether these are taste arms."""
    trains = {i: lin for i, lin in run.lineages.items()
              if run.committed(lin) and lin.steps > (lin.parent[1] if lin.parent else 0)}
    taste = {i: lin for i, lin in trains.items() if lin.taste is not None}
    if taste:
        return {i: "own" if lin.taste > 0 else "rev" if lin.taste < 0 else "neu" for i, lin in taste.items()}, True
    return {i: lin.arm for i, lin in trains.items()}, False


def flat(line: dict) -> dict[str, float]:
    """The numeric history fields of one update: HISTORY and every field whose name holds 'taste' (nested dicts
    flattened as taste.<key>)."""
    out = {k: float(line[k]) for k in HISTORY if isinstance(line.get(k), (int, float))}
    for k, v in line.items():
        if "taste" in k:
            items = v.items() if isinstance(v, dict) else [(None, v)]
            for k2, v2 in items:
                if isinstance(v2, (int, float)) and not isinstance(v2, bool):
                    out[k if k2 is None else f"{k}.{k2}"] = float(v2)
    return out


def history_table(run: data.Run, arms: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, present = [], {}
    for lid, arm in arms.items():
        for line in run.history(run.lineages[lid])[1]:
            f = flat(line)
            present.setdefault(arm, set()).update(f)
            rows += [{"arm": arm, "lineage": lid, "step": line["step"], "field": k, "value": v} for k, v in f.items()]
    long = pd.DataFrame(rows, columns=["arm", "lineage", "step", "field", "value"])
    g = long.groupby(["arm", "step", "field"])["value"]
    table = pd.concat([g.mean().rename("mean"), g.min().rename("min"), g.max().rename("max"),
                       g.size().rename("members")], axis=1).reset_index()
    fields = [{"arm": a, "field": k, "present": k in present.get(a, set())} for a in sorted(set(arms.values()))
              for k in HISTORY] + [{"arm": a, "field": "taste: " + ", ".join(TASTE_EXPECTED),
                                    "present": any("taste" in k for k in present.get(a, set()))}
                                   for a in sorted(set(arms.values()))]
    return table, pd.DataFrame(fields)


def probe_data(run: data.Run, arms: dict[str, str], strata, b: int, rng):
    """Probe grid of every arm member's probes and the probes of the base weights, with health's metrics."""
    subjects = {label: (lin, step) for label, (lin, step) in health.probe_subjects(run).items()
                if lin is None or lin.id in arms or run.base_weights(lin, step)}
    grid = data.probe_grid(run, subjects, keep_texts=True)
    data.add_strata(grid, strata)
    point, draws = health.evaluate(grid, rng, b)
    ref = health.reference_rows(run, grid.labels, subjects.get)
    return subjects, grid, point, draws, ref


def probe_table(run, arms, subjects, grid, point, draws, ref) -> pd.DataFrame:
    """Per arm and probe step, the mean over members of member - reference (clones' first probes and draws of the
    base weights left out, as in health)."""
    rp, rd = health.pooled(point, ref), health.pooled(draws, ref)
    groups = {}
    for label, (lin, step) in subjects.items():
        if lin is not None and lin.id in arms and grid.labels.index(label) not in ref and not run.restart(lin, step):
            groups.setdefault((arms[lin.id], step), []).append(grid.labels.index(label))
    rows = []
    for (arm, step), idx in sorted(groups.items()):
        for m in PROBE_METRICS:
            if m not in point:
                continue
            d = (draws[m][idx] - rd[m]).mean(0)
            rows.append({"arm": arm, "step": step, "members": len(idx), "metric": m,
                         "value": float(point[m][idx, 0].mean()), "diff": float((point[m][idx, 0] - rp[m][0]).mean()),
                         **dict(zip(("lo", "hi"), stats.ci(d)))})
    return pd.DataFrame(rows)


def taste_pairs(run: data.Run, arms: dict[str, str]) -> dict[str, dict[str, str]]:
    """parent id -> {arm: lineage id} for the parents with exactly one clone in each taste arm."""
    out = {}
    for lid, arm in arms.items():
        out.setdefault(run.lineages[lid].parent[0], {}).setdefault(arm, []).append(lid)
    return {p: {a: v[0] for a, v in d.items()} for p, d in out.items()
            if set(d) == set(TASTE_ARMS) and all(len(v) == 1 for v in d.values())}


def pair_table(run, pairs, grid, point, draws, dev) -> pd.DataFrame:
    """Per probe step at which every clone of a parent has a probe: (arm - neu) averaged over parents for health's
    metrics and, when scored, the deviation and the deviation at typical length; intervals over items."""
    rows = []
    steps = sorted(set.intersection(*[set(run.probe_steps(run.lineages[l])) for d in pairs.values() for l in d.values()]))
    for step in steps:
        for arm in ("own", "rev"):
            for m in PROBE_METRICS:
                if m not in point:
                    continue
                dp = [point[m][grid.labels.index(f"{d[arm]}@{step}"), 0] - point[m][grid.labels.index(f"{d['neu']}@{step}"), 0]
                      for d in pairs.values()]
                dd = np.mean([draws[m][grid.labels.index(f"{d[arm]}@{step}")] - draws[m][grid.labels.index(f"{d['neu']}@{step}")]
                              for d in pairs.values()], 0)
                rows.append({"step": step, "arm": arm, "metric": m, "parents": len(dp), "diff": float(np.mean(dp)),
                             **dict(zip(("lo", "hi"), stats.ci(dd))), "per_parent": " ".join(f"{x:+.4f}" for x in dp)})
            if dev is not None:
                for m, (p_, d_) in dev["values"].items():
                    labs = [(f"{d[arm]}@{step}", f"{d['neu']}@{step}") for d in pairs.values()]
                    if not all(a in dev["labels"] and b in dev["labels"] for a, b in labs):
                        continue
                    ia = [dev["labels"].index(a) for a, _ in labs]
                    ib = [dev["labels"].index(b) for _, b in labs]
                    dp = p_[ia] - p_[ib]
                    rows.append({"step": step, "arm": arm, "metric": m, "parents": len(dp), "diff": float(np.mean(dp)),
                                 **dict(zip(("lo", "hi"), stats.ci((d_[ia] - d_[ib]).mean(0)))),
                                 "per_parent": " ".join(f"{x:+.4f}" for x in dp)})
    return pd.DataFrame(rows)


def deviation_data(run: data.Run, selection, partial: bool, b: int, rng) -> dict | None:
    """Per author of the run's deviation selection: deviation and deviation at typical length, points and draws."""
    if selection is None and len(list((run.dir / "deviation").glob("*/selection.json"))) == 0:
        return None
    df, sel = data.deviation_table(run, data.deviation_selection(run, selection), partial)
    t = deviation.Table(run, df)
    w = t.draws(b, rng)
    typical, _ = deviation.typical_length(run, t.items["uid"])
    ones = stats.ones(len(t.items))
    return {"selection": sel, "labels": t.labels, "table": t,
            "values": {"deviation": (t.ratio(ones)[:, 0], t.ratio(w)),
                       "deviation_at_typical": (deviation.at_typical(t, typical, ones)[0][:, 0],
                                                deviation.at_typical(t, typical, w)[0])}}


def deviation_rows(run, arms, dev) -> pd.DataFrame:
    rows = []
    t = dev["table"]
    groups = {}
    for s in t.labels:
        lin, step = t.subject[s]
        if np.isnan(t.factor[s]) and lin.id in arms and not run.restart(lin, step) and not run.base_weights(lin, step):
            groups.setdefault((arms[lin.id], step), []).append(t.labels.index(s))
    for (arm, step), idx in sorted(groups.items()):
        row = {"arm": arm, "step": step, "members": len(idx)}
        for m, (p_, d_) in dev["values"].items():
            row[m] = float(np.nanmean(p_[idx]))
            row[f"{m}_lo"], row[f"{m}_hi"] = stats.ci(np.nanmean(d_[idx], 0))
        rows.append(row)
    return pd.DataFrame(rows)


def common_share(x: np.ndarray) -> float:
    """For movements x (members x sources, e.g. log share ratios): the share of their sum of squares that the mean
    movement over members carries, M sum_s mean_s^2 / sum x^2; 1 when all move alike, about 1/M when they move
    independently."""
    total = float((x ** 2).sum())
    return float(len(x) * (x.mean(0) ** 2).sum() / total) if total > 0 else np.nan


def practice_tables(run: data.Run, arms: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Source shares of every practising member per step, and the common share of their log-share movement."""
    rows, default = [], None
    for lid in arms:
        lin = run.lineages[lid]
        for line in run.history(lin)[1]:
            if "practice" not in line:
                continue
            if line["step"] == 0 and default is None:  # every average starts equal: the shares are the defaults
                default = line["practice"]["shares"]
            rows += [{"arm": arms[lid], "lineage": lid, "step": line["step"], "source": s, "share": v}
                     for s, v in line["practice"]["shares"].items()]
    shares = pd.DataFrame(rows, columns=["arm", "lineage", "step", "source", "share"])
    if shares.empty or default is None:
        return shares, pd.DataFrame()
    shares["log_ratio"] = np.log(shares["share"] / shares["source"].map(default))
    common = []
    for step, g in shares.groupby("step"):
        x = g.pivot(index="lineage", columns="source", values="log_ratio")
        if len(x) < 2:
            continue
        c = x.mean(0).to_numpy()
        common.append({"step": step, "members": len(x), "total_ss": float((x.to_numpy() ** 2).sum()),
                       "common_share": common_share(x.to_numpy()), "independent": 1 / len(x),
                       "largest_common": x.columns[np.argmax(np.abs(c))], "largest_common_ratio": float(np.exp(c[np.argmax(np.abs(c))]))})
    return shares, pd.DataFrame(common)


def authorship_table(run, arms, subjects, grid, cap: int, taste: bool, b: int, rng) -> pd.DataFrame:
    """Per probe step and arm with two or more members, and over all arms' members together ("all"): the classifier
    among them, raw and stripped. Taste arms keep the clones' first probes (at the clone step they are draws of the
    parents, one per arm and parent; "all" then holds three draws of each parent and is left out); otherwise clones'
    first probes are left out."""
    groups = {}
    for label, (lin, step) in subjects.items():
        if lin is not None and lin.id in arms and (taste or not run.restart(lin, step)):
            groups.setdefault((step, arms[lin.id]), []).append(label)
            if not (taste and run.restart(lin, step)):
                groups.setdefault((step, "all"), []).append(label)
    rows = []
    for (step, arm), labels in sorted(groups.items()):
        if len(labels) < 2:
            continue
        au = divergence.authorship(grid, labels, cap, rng, b=b)
        rows += [{"step": step, "arm": arm, **r} for r in au.to_dict("records")]
    return pd.DataFrame(rows)


def windows(steps, probe_steps) -> dict[int, str]:
    """step -> the window between probes it belongs to (e.g. 0-14)."""
    edges = sorted(set(probe_steps))
    out = {}
    for s in steps:
        lo = max([e for e in edges if e <= s], default=min(steps))
        hi = min([e for e in edges if e > s], default=max(steps) + 1) - 1
        out[s] = f"{lo}-{hi}"
    return out


def report_lines(run, arms, taste, res) -> list[str]:
    n = pd.Series(list(arms.values())).value_counts()
    lines = [f"arms {run.name}: " + ("taste-shaped reward clones paired by parent (own, neu, rev)" if taste
                                     else "update rules of the trained lineages") + "; "
             + ", ".join(f"{a} ({n[a]})" for a in sorted(n.index))]
    h, f = res["history"], res["fields"]
    if len(h):
        probes = sorted({s for lid in arms for s in run.probe_steps(run.lineages[lid])})
        win = windows(sorted(h["step"].unique()), probes)
        h = h.assign(window=h["step"].map(win))
        show = [k for k in ("reward", "completion_tokens", "truncated", "groups", "adv_sequences", "si_sequences",
                            "ex_sequences", "ex_loss") if (h["field"] == k).any()]
        taste_fields = sorted(k for k in h["field"].unique() if "taste" in k)
        lines += ["", "training history: means over members and the steps of each window between probes"]
        rows = []
        for (arm, w), g in h.groupby(["arm", "window"], sort=False):
            m = g.groupby("field")["mean"].mean()
            rows.append([arm, w, int(g["members"].max())] + [
                num(m.get(k, np.nan), digits=0 if k in ("completion_tokens", "adv_sequences", "si_sequences", "ex_sequences")
                    else 3) for k in show + taste_fields])
        lines += render(["arm", "steps", "n"] + show + taste_fields, rows, 2)
        absent = f[(~f["present"]) & f["field"].str.startswith("taste")]
        if taste or len(absent) < len(f[f["field"].str.startswith("taste")]):
            lines += ["  taste-shaped reward fields: " + (", ".join(taste_fields) if taste_fields else
                                                          "absent from the history (expected: " + ", ".join(TASTE_EXPECTED) + ")")]
    else:
        lines += ["", "training history: no committed update yet (records past a lineage's last checkpoint are not read)"]
    pr = res["probes"]
    if not len(pr):
        lines += ["", "probes: none besides draws of the base weights"]
    if len(pr):
        lines += ["", "probes: mean over members of (member - reference), percentage points [95% over items]; tokens "
                      "and accuracy as values"]
        cols = [m for m in ("far", "near", "mastered", "frontier", "heldout_complied", "truncated") if (pr["metric"] == m).any()]
        rows = []
        for (arm, step), g in pr.groupby(["arm", "step"]):
            m = g.set_index("metric")
            rows.append([arm, step, int(m["members"].iloc[0]), pct(m.loc["accuracy", "value"]), f"{m.loc['tokens', 'value']:.0f}"]
                        + [pci(m.loc[k, "diff"], m.loc[k, "lo"], m.loc[k, "hi"], sign=True) for k in cols])
        lines += render(["arm", "step", "n", "acc", "tokens"] + [f"{k} - ref" for k in cols], rows, 2)
    pa = res.get("pairs", pd.DataFrame())
    if len(pa):
        lines += ["", "paired by parent: arm - neu averaged over parents (accuracy and the like in points, tokens, "
                      "deviation in nats per token) [95% over items]; at the clone step the clones are one checkpoint"]
        mets = [m for m in ("accuracy", "far", "frontier", "heldout_complied", "tokens", "deviation", "deviation_at_typical")
                if (pa["metric"] == m).any()]
        rows = []
        for (step, arm), g in pa.groupby(["step", "arm"]):
            m = g.set_index("metric")
            rows.append([step, arm, int(m["parents"].iloc[0])] + [
                (num(m.loc[k, "diff"], m.loc[k, "lo"], m.loc[k, "hi"], 0 if k == "tokens" else 3) if k in ("tokens", "deviation", "deviation_at_typical")
                 else pci(m.loc[k, "diff"], m.loc[k, "lo"], m.loc[k, "hi"], sign=True)) for k in mets])
        lines += render(["step", "arm", "parents"] + mets, rows, 2)
    dv = res["deviation"]
    if len(dv):
        lines += ["", f"deviation (nats per token; selection {res['deviation_selection']}): mean over members, and at the "
                      "base model's typical length [95% over items]"]
        lines += render(["arm", "step", "n", "deviation", "at typical length"],
                        [[r.arm, r.step, r.members, num(r.deviation, r.deviation_lo, r.deviation_hi, 3),
                          num(r.deviation_at_typical, r.deviation_at_typical_lo, r.deviation_at_typical_hi, 3)]
                         for r in dv.itertuples()], 1)
    elif res["deviation_selection"] is None:
        lines += ["", "deviation: no deviation selection (python -m evotaste.evaluate deviation)"]
    pc = res["practice_common"]
    if len(pc):
        lines += ["", "practice: share of the practising members' log-share movement (from the defaults) common to all of "
                      "them, against 1/members if they moved independently; the source moved most in common"]
        lines += [f"  step {r.step}: {pct(r.common_share)}% (independent {pct(r.independent)}%), {r.largest_common} x"
                  f"{r.largest_common_ratio:.2f}" for r in pc.itertuples() if r.total_ss > 0]
    au = res["authorship"]
    if len(au):
        lines += ["", "authorship among an arm's members on their probe answers: raw / stripped of layout and notation "
                      "(chance) and the share lost by stripping (shown when raw is above chance)"]
        rows = []
        for (step, arm), g in au.groupby(["step", "arm"]):
            v = g.set_index("view")
            above = v.loc["raw", "lo"] > v.loc["raw", "chance"]
            rows.append([step, arm, int(v.loc["raw", "members"]), pci(v.loc["raw", "accuracy"], v.loc["raw", "lo"], v.loc["raw", "hi"]),
                         pci(v.loc["stripped", "accuracy"], v.loc["stripped", "lo"], v.loc["stripped", "hi"]),
                         pct(v.loc["raw", "chance"]),
                         pci(v.loc["cosmetic_share", "accuracy"], v.loc["cosmetic_share", "lo"], v.loc["cosmetic_share", "hi"])
                         if above else "-"])
        lines += render(["step", "arm", "n", "raw [95%]", "stripped [95%]", "chance", "lost by stripping [95%]"], rows, 2)
    return lines


def analyse(run: data.Run, census=None, selection=None, partial=False, b: int = 1000, cap: int = 150, rng=None) -> dict:
    rng = rng or np.random.default_rng(0)
    arms, taste = arms_of(run)
    if not arms:
        raise SystemExit(f"{run.name}: no committed lineage that trains")
    strata = data.census_strata(census) if census else None
    res = {}
    res["history"], res["fields"] = history_table(run, arms)
    subjects, grid, point, draws, ref = probe_data(run, arms, strata, b, rng)
    res["probes"] = probe_table(run, arms, subjects, grid, point, draws, ref)
    dev = deviation_data(run, selection, partial, b, rng)
    res["deviation_selection"] = None if dev is None else dev["selection"].name
    res["deviation"] = deviation_rows(run, arms, dev) if dev is not None else pd.DataFrame()
    if taste:
        pairs = taste_pairs(run, arms)
        res["pairs"] = pair_table(run, pairs, grid, point, draws, dev) if pairs else pd.DataFrame()
    res["practice"], res["practice_common"] = practice_tables(run, arms)
    res["authorship"] = authorship_table(run, arms, subjects, grid, cap, taste, b, rng) if cap > 0 else pd.DataFrame()
    return {"arms": arms, "taste": taste, **res}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--census", help="census output directory (results/census/<name>) for the base model's strata")
    ap.add_argument("--deviation", type=Path, help="a deviation selection (default: the run's only one, if any)")
    ap.add_argument("--partial", action="store_true", help="read a deviation selection still being scored")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap draws over items")
    ap.add_argument("--classifier-items", type=int, default=150, help="probe items per half of the units for the "
                                                                      "authorship classifier (0: skip)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    run = data.Run(args.run_dir)
    res = analyse(run, args.census, args.deviation, args.partial, args.boot, args.classifier_items)
    lines = report_lines(run, res["arms"], res["taste"], res)
    tables = {k: v for k, v in res.items() if isinstance(v, pd.DataFrame)}
    write(args.out or Path("results") / run.name / "arms", tables, lines)


if __name__ == "__main__":
    with threadpool_limits(stats.THREADS):
        main()
