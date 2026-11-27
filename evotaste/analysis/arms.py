"""Arms of an amplification pilot: members grouped by the mechanism they train with, compared at every probe step.

    python -m evotaste.analysis.arms runs/<name> [--parents p1,p2 --arms a1,a2,a3 --reference a2]
                                      [--census results/census/<c>] [--deviation DIR] [--partial] [--boot 1000]
                                      [--classifier-items 150] [--out DIR]

Arms. With --parents, the members are the clones <parent>-<arm> of the parents named (pilot E: --arms own,neu,rev
--reference neu; pilot F: --arms fro,yok,pln,mat --reference pln), wherever they live in the run, and every
comparison is paired by parent against the reference arm. Without, the arms are the update rules (data.rule_name)
of every lineage that trains, as in pilot D. Tables:

- history.csv: per arm and step, mean, minimum and maximum over members of the training history: reward,
  completion tokens, truncated share, informative groups, each term's rollouts, tokens and loss, and the numeric
  fields of the taste-shaped reward and of revisits (taste.<key>, revisit.<key>); fields.csv lists which are present.
- history_pairs.csv (with --parents): per arm, window between probes and field (reward, tokens, truncated,
  groups), the mean over parents and steps of arm - reference on the same step, with a normal interval over those
  paired steps.
- probes.csv: per arm and probe step, the mean over members of (member - reference), the reference being the probes
  of the base weights (health), with intervals over items; with --census by stratum too.
- pairs.csv (with --parents): per probe step and arm, arm - reference averaged over parents for health's metrics
  and, when scored, the deviation, the deviation at typical length and its share in layout tokens; intervals over
  items, per-parent values. At the clone step the clones' probes are draws of one checkpoint: a null.
- deviation.csv: per arm and step the mean deviation, at the base model's typical length, and in layout tokens.
- practice.csv, practice_common.csv: practising members' source shares and the share of their movement common to all.
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
           "si_sequences", "si_tokens", "si_loss", "ex_sequences", "ex_tokens", "ex_loss", "rp_sequences",
           "rp_tokens", "rp_loss")
MECHANISMS = ("taste", "revisit")  # nested history dictionaries whose numeric fields are read
PAIRED_HISTORY = ("reward", "completion_tokens", "truncated", "groups")
# shown in the report when present: the taste score's mean and its correlation with length, shaped counts; revisit
# slots, shortfall, frontier size, and reward on revisited against stream prompts
MECHANISM_SHOWN = ("taste.score_mean", "taste.length_corr", "taste.shaped", "taste.groups", "revisit.slots",
                   "revisit.shortfall", "revisit.frontier", "revisit.reward", "revisit.stream_reward")
PROBE_METRICS = ("accuracy", "near", "far", "mastered", "sweet", "frontier", "heldout_complied", "truncated", "tokens")


def explicit_arms(run: data.Run, parents: list[str], arms: list[str], reference: str) -> tuple[dict, dict, list]:
    """lineage id -> arm and -> parent for the committed clones <parent>-<arm>, and the clones not started yet or not
    in the run (pilot F has -mat for one parent only); every arm must exist for some parent."""
    if reference not in arms:
        raise SystemExit(f"--reference {reference} is not among --arms {arms}")
    arm_of, parent_of, waiting = {}, {}, []
    for a in arms:
        if not any(f"{p}-{a}" in run.lineages for p in parents):
            raise SystemExit(f"arm {a}: no clone <parent>-{a} of {parents} in run {run.name}")
    for p in parents:
        for a in arms:
            lid = f"{p}-{a}"
            lin = run.lineages.get(lid)
            if lin is None:
                waiting.append(f"{lid} (not in the run)")
                continue
            if lin.parent is None or lin.parent[0] != p:
                raise SystemExit(f"{lid}: not a clone of {p} in run {run.name}")
            if run.committed(lin):
                arm_of[lid], parent_of[lid] = a, p
            else:
                waiting.append(f"{lid} (not started)")
    return arm_of, parent_of, waiting


def rule_arms(run: data.Run) -> dict[str, str]:
    """lineage id -> update rule of every committed lineage that trains."""
    return {i: lin.arm for i, lin in run.lineages.items()
            if run.committed(lin) and lin.steps > (lin.parent[1] if lin.parent else 0)}


def flat(line: dict) -> dict[str, float]:
    """The numeric history fields of one update: HISTORY, and the numeric fields of the mechanisms' dictionaries
    (taste.<key>, revisit.<key>)."""
    out = {k: float(line[k]) for k in HISTORY if isinstance(line.get(k), (int, float))}
    for k in MECHANISMS:
        for k2, v in (line.get(k) or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[f"{k}.{k2}"] = float(v)
    return out


def history_long(run: data.Run, arms: dict[str, str]) -> pd.DataFrame:
    rows = []
    for lid, arm in arms.items():
        for line in run.history(run.lineages[lid])[1]:
            rows += [{"arm": arm, "lineage": lid, "step": line["step"], "field": k, "value": v}
                     for k, v in flat(line).items()]
    return pd.DataFrame(rows, columns=["arm", "lineage", "step", "field", "value"])


def history_table(long: pd.DataFrame, arms: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = long.groupby(["arm", "step", "field"])["value"]
    table = pd.concat([g.mean().rename("mean"), g.min().rename("min"), g.max().rename("max"),
                       g.size().rename("members")], axis=1).reset_index()
    present = long.groupby("arm")["field"].agg(set).to_dict()
    names = sorted(set(HISTORY) | set(long["field"]))
    fields = [{"arm": a, "field": k, "present": k in present.get(a, set())} for a in sorted(set(arms.values()))
              for k in names]
    return table, pd.DataFrame(fields)


def windows(steps, probe_steps) -> dict[int, str]:
    """step -> the window between probes it belongs to (e.g. 45-59)."""
    edges = sorted(set(probe_steps))
    out = {}
    for s in steps:
        lo = max([e for e in edges if e <= s], default=min(steps))
        hi = min([e for e in edges if e > s], default=max(steps) + 1) - 1
        out[s] = f"{lo}-{hi}"
    return out


def history_pairs(long: pd.DataFrame, parent_of: dict, reference: str, win: dict) -> pd.DataFrame:
    """Per arm, window and field: arm - reference of the same parent at the same step, averaged over the paired
    (parent, step) cells, with a normal 95% interval over them."""
    x = long[long["field"].isin(PAIRED_HISTORY)].assign(parent=lambda d: d["lineage"].map(parent_of))
    ref = x[x["arm"] == reference].set_index(["parent", "step", "field"])["value"]
    rows = []
    for (arm, field), g in x[x["arm"] != reference].groupby(["arm", "field"]):
        d = g.set_index(["parent", "step", "field"])["value"] - ref
        d = d.dropna().reset_index()
        d["window"] = d["step"].map(win)
        for w, dw in d.groupby("window", sort=False):
            v = dw["value"].to_numpy()
            half = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
            rows.append({"arm": arm, "field": field, "window": w, "cells": len(v), "parents": dw["parent"].nunique(),
                         "diff": float(v.mean()), "lo": float(v.mean() - half), "hi": float(v.mean() + half)})
    return pd.DataFrame(rows)


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


def pair_table(run, arm_of, parent_of, reference, grid, point, draws, dev) -> pd.DataFrame:
    """Per probe step and arm: (arm - reference) of the same parent, averaged over the parents that have both at
    that step, for health's metrics and, when scored, the deviation measures; intervals over items."""
    by = {}
    for lid, a in arm_of.items():
        by.setdefault(parent_of[lid], {})[a] = lid
    rows = []
    steps = sorted({s for lid in arm_of for s in run.probe_steps(run.lineages[lid])})
    for step in steps:
        for arm in sorted(set(arm_of.values()) - {reference}):
            labs = [(f"{d[arm]}@{step}", f"{d[reference]}@{step}") for d in by.values()
                    if arm in d and reference in d and f"{d[arm]}@{step}" in grid.labels
                    and f"{d[reference]}@{step}" in grid.labels]
            if not labs:
                continue
            ia, ib = [grid.labels.index(a) for a, _ in labs], [grid.labels.index(b) for _, b in labs]
            for m in PROBE_METRICS:
                if m in point:
                    dp = point[m][ia, 0] - point[m][ib, 0]
                    rows.append({"step": step, "arm": arm, "metric": m, "parents": len(dp), "diff": float(dp.mean()),
                                 **dict(zip(("lo", "hi"), stats.ci((draws[m][ia] - draws[m][ib]).mean(0)))),
                                 "same_sign": int(max((dp > 0).sum(), (dp < 0).sum())),
                                 "per_parent": " ".join(f"{x:+.4f}" for x in dp)})
            if dev is None or not all(a in dev["labels"] and b in dev["labels"] for a, b in labs):
                continue
            ja, jb = [dev["labels"].index(a) for a, _ in labs], [dev["labels"].index(b) for _, b in labs]
            for m, (p_, d_) in dev["values"].items():
                dp = p_[ja] - p_[jb]
                rows.append({"step": step, "arm": arm, "metric": m, "parents": len(dp), "diff": float(np.mean(dp)),
                             **dict(zip(("lo", "hi"), stats.ci((d_[ja] - d_[jb]).mean(0)))),
                             "same_sign": int(max((dp > 0).sum(), (dp < 0).sum())),
                             "per_parent": " ".join(f"{x:+.4f}" for x in dp)})
    return pd.DataFrame(rows)


def deviation_data(run: data.Run, selection, partial: bool, b: int, rng) -> dict | None:
    """Per author of the run's deviation selection: deviation, deviation at typical length and the share in layout
    tokens, points and draws."""
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
                                                deviation.at_typical(t, typical, w)[0]),
                       "deviation_in_layout": (t.layout_parts(ones)["layout_share"][:, 0],
                                               t.layout_parts(w)["layout_share"])}}


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
            row[m] = float(np.nanmean(p_[idx])) if np.isfinite(p_[idx]).any() else np.nan
            row[f"{m}_lo"], row[f"{m}_hi"] = stats.ci(np.nanmean(d_[idx], 0)) if np.isfinite(d_[idx]).any() else (
                np.nan, np.nan)
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
        for line in run.history(run.lineages[lid])[1]:
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
                       "largest_common": x.columns[np.argmax(np.abs(c))],
                       "largest_common_ratio": float(np.exp(c[np.argmax(np.abs(c))]))})
    return shares, pd.DataFrame(common)


def authorship_table(run, arms, subjects, grid, cap: int, clones: bool, b: int, rng) -> pd.DataFrame:
    """Per probe step and arm with two or more members, and over all arms' members together ("all"): the classifier
    among them, raw and stripped. For clone arms the clones' first probes are kept (at the clone step each arm holds
    one draw of every parent; "all", which would hold several draws of each parent, is then left out); otherwise
    clones' first probes are left out."""
    groups = {}
    for label, (lin, step) in subjects.items():
        if lin is not None and lin.id in arms and (clones or not run.restart(lin, step)):
            groups.setdefault((step, arms[lin.id]), []).append(label)
            if not (clones and run.restart(lin, step)):
                groups.setdefault((step, "all"), []).append(label)
    rows = []
    for (step, arm), labels in sorted(groups.items()):
        if len(labels) < 2:
            continue
        au = divergence.authorship(grid, labels, cap, rng, b=b)
        rows += [{"step": step, "arm": arm, **r} for r in au.to_dict("records")]
    return pd.DataFrame(rows)


def analyse(run: data.Run, census=None, selection=None, partial=False, b: int = 1000, cap: int = 150,
            parents=None, arm_names=None, reference=None, rng=None) -> dict:
    rng = rng or np.random.default_rng(0)
    if parents:
        arms, parent_of, waiting = explicit_arms(run, parents, arm_names, reference)
    else:
        arms, parent_of, waiting = rule_arms(run), {}, []
    if not arms:
        raise SystemExit(f"{run.name}: no committed lineage among the arms")
    strata = data.census_strata(census) if census else None
    res = {"arms": arms, "parent_of": parent_of, "waiting": waiting, "reference": reference}
    long = history_long(run, arms)
    res["history"], res["fields"] = history_table(long, arms)
    probes = sorted({s for lid in arms for s in run.probe_steps(run.lineages[lid])})
    res["window"] = windows(sorted(long["step"].unique()), probes) if len(long) else {}
    res["history_pairs"] = (history_pairs(long, parent_of, reference, res["window"]) if parents and len(long)
                            else pd.DataFrame())
    subjects, grid, point, draws, ref = probe_data(run, arms, strata, b, rng)
    res["probes"] = probe_table(run, arms, subjects, grid, point, draws, ref)
    dev = deviation_data(run, selection, partial, b, rng)
    res["deviation_selection"] = None if dev is None else dev["selection"].name
    res["deviation"] = deviation_rows(run, arms, dev) if dev is not None else pd.DataFrame()
    res["pairs"] = pair_table(run, arms, parent_of, reference, grid, point, draws, dev) if parents else pd.DataFrame()
    res["practice"], res["practice_common"] = practice_tables(run, arms)
    res["authorship"] = (authorship_table(run, arms, subjects, grid, cap, bool(parents), b, rng) if cap > 0
                         else pd.DataFrame())
    return res


def report_lines(run, res) -> list[str]:
    arms, ref = res["arms"], res["reference"]
    n = pd.Series(list(arms.values())).value_counts()
    lines = [f"arms {run.name}: " + (f"clones <parent>-<arm> of {', '.join(sorted(set(res['parent_of'].values())))}, "
                                     f"paired by parent against {ref}" if ref else "update rules of the trained lineages")
             + "; " + ", ".join(f"{a} ({n[a]})" for a in sorted(n.index))]
    if res["waiting"]:
        lines.append("left out: " + ", ".join(res["waiting"]))
    h = res["history"]
    if len(h):
        h = h.assign(window=h["step"].map(res["window"]))
        shown = [k for k in ("reward", "completion_tokens", "truncated", "groups", "adv_sequences", "si_sequences",
                             "ex_sequences", "rp_sequences") + MECHANISM_SHOWN if (h["field"] == k).any()]
        lines += ["", "training history: means over members and the steps of each window between probes"]
        rows = []
        for (arm, w), g in h.groupby(["arm", "window"], sort=False):
            m = g.groupby("field")["mean"].mean()
            rows.append([arm, w, int(g["members"].max())] + [
                num(m.get(k, np.nan), digits=0 if k.endswith(("tokens", "sequences", "slots", "shortfall", "frontier",
                                                               "shaped", "groups")) else 3) for k in shown])
        lines += render(["arm", "steps", "n"] + shown, rows, 2)
        hp = res["history_pairs"]
        if len(hp):
            lines += ["", f"training history, arm - {ref} of the same parent at the same step (mean over parents and "
                          "steps, normal 95% interval over those cells); revisiting arms' reward includes the revisited "
                          "prompts (compare revisit.stream_reward)"]
            cell = {(r.arm, r.window, r.field): r for r in hp.itertuples()}
            rows = []
            for (arm, w) in dict.fromkeys(zip(hp["arm"], hp["window"])):
                rows.append([arm, w] + [num(cell[(arm, w, k)].diff, cell[(arm, w, k)].lo, cell[(arm, w, k)].hi,
                                            0 if k == "completion_tokens" else 3) if (arm, w, k) in cell else "-"
                                        for k in PAIRED_HISTORY])
            lines += render(["arm", "steps"] + list(PAIRED_HISTORY), rows, 2)
    else:
        lines += ["", "training history: no committed update yet (records past a lineage's last checkpoint are not read)"]
    pa = res["pairs"]
    if len(pa):
        lines += ["", f"probes paired by parent: arm - {ref} averaged over parents, points [95% over items] (tokens in "
                      "tokens, deviation in nats per token, its layout share in %); same: parents agreeing in sign; at "
                      "the clone step the clones are one checkpoint (a null)"]
        mets = [m for m in ("accuracy", "near", "far", "sweet", "frontier", "mastered", "heldout_complied", "tokens",
                            "deviation", "deviation_at_typical", "deviation_in_layout") if (pa["metric"] == m).any()]
        rows = []
        for (step, arm), g in pa.groupby(["step", "arm"]):
            m = g.set_index("metric")
            rows.append([step, arm, int(m["parents"].iloc[0])] + [
                (num(m.loc[k, "diff"], m.loc[k, "lo"], m.loc[k, "hi"], 0 if k == "tokens" else 3)
                 if k in ("tokens", "deviation", "deviation_at_typical") else
                 pci(m.loc[k, "diff"], m.loc[k, "lo"], m.loc[k, "hi"], 0 if k == "deviation_in_layout" else 1,
                     sign=True)) + f" {int(m.loc[k, 'same_sign'])}" for k in mets])
        lines += render(["step", "arm", "par"] + mets, rows, 2)
    pr = res["probes"]
    if len(pr):
        lines += ["", "probes: mean over members of (member - reference = probes of the base weights), points "
                      "[95% over items]; accuracy and tokens as values"]
        cols = [m for m in ("far", "near", "mastered", "frontier", "heldout_complied", "truncated") if (pr["metric"] == m).any()]
        rows = []
        for (arm, step), g in pr.groupby(["arm", "step"]):
            m = g.set_index("metric")
            rows.append([arm, step, int(m["members"].iloc[0]), pct(m.loc["accuracy", "value"]), f"{m.loc['tokens', 'value']:.0f}"]
                        + [pci(m.loc[k, "diff"], m.loc[k, "lo"], m.loc[k, "hi"], sign=True) for k in cols])
        lines += render(["arm", "step", "n", "acc", "tokens"] + [f"{k} - ref" for k in cols], rows, 2)
    else:
        lines += ["", "probes: none besides draws of the base weights and clones' first probes"]
    dv = res["deviation"]
    if len(dv):
        lines += ["", f"deviation (nats per token; selection {res['deviation_selection']}): mean over members, at the base "
                      "model's typical length, share in layout tokens [95% over items]"]
        lines += render(["arm", "step", "n", "deviation", "at typical length", "in layout %"],
                        [[r.arm, r.step, r.members, num(r.deviation, r.deviation_lo, r.deviation_hi, 3),
                          num(r.deviation_at_typical, r.deviation_at_typical_lo, r.deviation_at_typical_hi, 3),
                          num(100 * r.deviation_in_layout, 100 * r.deviation_in_layout_lo, 100 * r.deviation_in_layout_hi, 0)]
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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--parents", help="comma-separated parents whose clones <parent>-<arm> are the members")
    ap.add_argument("--arms", help="comma-separated arm suffixes (with --parents)")
    ap.add_argument("--reference", help="the arm every comparison is paired against (with --parents)")
    ap.add_argument("--census", help="census output directory (results/census/<name>) for the base model's strata")
    ap.add_argument("--deviation", type=Path, help="a deviation selection (default: the run's only one, if any)")
    ap.add_argument("--partial", action="store_true", help="read a deviation selection still being scored")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap draws over items")
    ap.add_argument("--classifier-items", type=int, default=150, help="probe items per half of the units for the "
                                                                      "authorship classifier (0: skip)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if bool(args.parents) != bool(args.arms) or bool(args.parents) != bool(args.reference):
        raise SystemExit("--parents, --arms and --reference go together")
    run = data.Run(args.run_dir)
    res = analyse(run, args.census, args.deviation, args.partial, args.boot, args.classifier_items,
                  args.parents.split(",") if args.parents else None, args.arms.split(",") if args.arms else None,
                  args.reference)
    lines = report_lines(run, res)
    write(args.out or Path("results") / run.name / "arms", {k: v for k, v in res.items() if isinstance(v, pd.DataFrame)},
          lines)


if __name__ == "__main__":
    with threadpool_limits(stats.THREADS):
        main()
