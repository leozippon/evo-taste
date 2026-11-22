"""Niches: which problems each adapter solves, on one final evaluation.

    python -m evotaste.analysis.niches runs/<name> (--suite train|selection | --blind blind1|blind2) --tag T
                                                   [--adapter X] [--comparators a,b] [--out DIR]

An instance is solved by an adapter if at least half of its samples are correct, ever solved if any sample is.
Reports overlap of members' solved sets, unique solves, population coverage, changes against the base model, family
unlocks, and what the designated adapter (default: the outlier) solves that no comparator (default: base and the
baselines) ever solves, and the reverse. A blind vault is read only with --blind and an existing harvest manifest.
Writes CSV to results/<name>/niches/<suite>/. Definitions: docs/dev/ANALYSIS_NOTES.md.
"""

from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from evotaste.analysis.harvest import Suite, adapter_roles, pick_outlier
from evotaste.analysis.report import render
from evotaste.analysis.rundata import VAULTS, Evaluation, load_evaluation, manifest_path, member_specs

UNLOCK_BASE_MAX, UNLOCK_MIN = 0.02, 0.10


def k_coverage(solvers: np.ndarray, m: int, k: int) -> float:
    """Expected number of instances solved by at least one of k members drawn without replacement from m, given
    the number of solving members per instance: sum over instances of 1 - C(m - s, k) / C(m, k)."""
    return float(sum(1 - math.comb(m - int(s), k) / math.comb(m, k) for s in solvers))


def analyse(ev: Evaluation, roles: dict, designated: str, comparators: list[str]) -> dict:
    pos = {a: i for i, a in enumerate(ev.adapters)}
    rate = ev.reward.mean(-1)
    solved, ever = rate >= 0.5, rate > 0
    members = [a for a in ev.adapters if roles[a]["role"] == "member"]
    s = solved[[pos[a] for a in members]]
    solvers = s.sum(0)
    res = {"rates": pd.DataFrame(rate.T, index=pd.Index(ev.uids, name="uid"), columns=ev.adapters)
           .assign(family=ev.families)}

    pairs = []
    for a, b in combinations(members, 2):
        inter, union = int((solved[pos[a]] & solved[pos[b]]).sum()), int((solved[pos[a]] | solved[pos[b]]).sum())
        arms = sorted((roles[a]["arm"], roles[b]["arm"]))
        pairs.append({"a": a, "b": b, "arm_pair": "-".join(arms), "intersection": inter, "union": union,
                      "jaccard": inter / union if union else np.nan})
    res["jaccard"] = pd.DataFrame(pairs, columns=["a", "b", "arm_pair", "intersection", "union", "jaccard"])

    base = solved[pos["base"]] if "base" in pos else None
    rows = []
    for a in ev.adapters:
        if a == "base":
            continue
        i = pos[a]
        row = {"adapter": a, "role": roles[a]["role"], "arm": roles[a]["arm"], "solved": int(solved[i].sum()),
               "ever": int(ever[i].sum()),
               "unique": int((solved[i] & (solvers == 1)).sum()) if a in members else np.nan}
        if base is not None:
            row.update(preserved=int((solved[i] & base).sum()), gained=int((solved[i] & ~base).sum()),
                       lost=int((~solved[i] & base).sum()))
        rows.append(row)
    res["adapters"] = pd.DataFrame(rows)

    m = len(members)
    ks = sorted({k for k in (2 ** j for j in range(m.bit_length())) if k <= m} | {m}) if m else []
    best = max(members, key=lambda a: solved[pos[a]].sum()) if members else None
    res["coverage"] = {"instances": len(ev.uids), "members": m, "population": int((solvers > 0).sum()),
                       "best_member": best, "best": int(solved[pos[best]].sum()) if best else 0,
                       "base": int(base.sum()) if base is not None else None,
                       "k": {k: k_coverage(solvers, m, k) for k in ks}}

    rows = []
    if "base" in pos:
        for f in dict.fromkeys(ev.families):
            idx = ev.families == f
            base_acc = ev.reward[pos["base"], idx].mean()
            for a in ev.adapters:
                r = ev.reward[pos[a], idx]
                if a != "base" and base_acc < UNLOCK_BASE_MAX and r.mean() >= UNLOCK_MIN:
                    k, n = int(r.sum()), r.size
                    lo, hi = stats.binomtest(k, n).proportion_ci(0.95)
                    rows.append({"adapter": a, "role": roles[a]["role"], "family": f, "base_accuracy": base_acc,
                                 "accuracy": k / n, "lo": lo, "hi": hi, "n": n})
    res["unlocks"] = pd.DataFrame(rows, columns=["adapter", "role", "family", "base_accuracy", "accuracy", "lo", "hi",
                                                 "n"])

    d = pos[designated]
    comp = [pos[c] for c in comparators]
    only = solved[d] & ~ever[comp].any(0)  # the designated adapter solves; no comparator ever does
    reverse = solved[comp].any(0) & ~ever[d]  # some comparator solves; the designated adapter never does
    fam = pd.DataFrame({"family": ev.families, "only_designated": only, "only_comparators": reverse})
    per_family = fam.groupby("family", sort=False).sum().reset_index()
    res["designated"] = pd.concat([pd.DataFrame([{"family": "all", "only_designated": int(only.sum()),
                                                  "only_comparators": int(reverse.sum())}]), per_family])
    res["designated_label"], res["comparators"] = designated, comparators
    return res


def summary_lines(name: str, ev: Evaluation, res: dict, designated_source: str) -> list[str]:
    cov = res["coverage"]
    lines = [f"niches {name} {ev.suite} {ev.tag}: {len(ev.uids)} instances ({ev.per_family}/family x {ev.samples} "
             f"samples); solved = at least half the samples correct, ever = any"]
    if ev.dropped:
        lines.append(f"  {len(ev.dropped)} instances dropped (question differs between adapters)")
    j = res["jaccard"]
    if not j.empty:
        by_arm = j.groupby("arm_pair")["jaccard"].mean()
        lines.append(f"Jaccard of members' solved sets: mean {j['jaccard'].mean():.2f}; by arm pair: "
                     + ", ".join(f"{k} {v:.2f}" for k, v in by_arm.items()))
    lines.append(f"coverage: population {cov['population']} of {cov['instances']} ({cov['members']} members); best "
                 f"member {cov['best_member']} {cov['best']}; base {cov['base']}; mean coverage of k random members: "
                 + ", ".join(f"k={k} {v:.1f}" for k, v in cov["k"].items()))
    a = res["adapters"]
    cols = [c for c in ("solved", "ever", "unique", "preserved", "gained", "lost") if c in a]
    lines += render(["adapter", "role"] + cols,
                    [[r["adapter"], r["role"] if r["role"] != "member" else r["arm"]]
                     + ["-" if pd.isna(r[c]) else str(int(r[c])) for c in cols] for _, r in a.iterrows()], left=2)
    u = res["unlocks"]
    lines.append(f"family unlocks (base < {100 * UNLOCK_BASE_MAX:.0f}%, adapter >= {100 * UNLOCK_MIN:.0f}%): "
                 f"{len(u)}" + (": " if len(u) else "") + ", ".join(
                     f"{r.adapter} {r.family} {100 * r.accuracy:.0f}% [{100 * r.lo:.0f},{100 * r.hi:.0f}]"
                     for r in u.sort_values("accuracy", ascending=False).head(10).itertuples())
                 + (f" ... ({len(u) - 10} more in unlocks.csv)" if len(u) > 10 else ""))
    dsg = res["designated"]
    overall = dsg.iloc[0]
    fams = dsg.iloc[1:].query("only_designated > 0 or only_comparators > 0")
    lines.append(f"{res['designated_label']} ({designated_source}) vs {', '.join(res['comparators'])}: solves "
                 f"{overall['only_designated']} that no comparator ever solves; comparators solve "
                 f"{overall['only_comparators']} it never solves")
    if not fams.empty:
        lines.append("  by family (+designated/-comparators): " + ", ".join(
            f"{r.family} +{r.only_designated}/-{r.only_comparators}" for r in fams.itertuples()))
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--suite", choices=["train", "selection"])
    where.add_argument("--blind", choices=VAULTS, help="a blind vault (needs the harvest manifest)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", help="designated adapter (default: the outlier)")
    ap.add_argument("--comparators", help="comma-separated comparator set (default: base and the baselines)")
    ap.add_argument("--out", help="output directory (default results/<name>/niches/<suite>)")
    args = ap.parse_args(argv)
    name, specs = member_specs(args.run_dir)
    suite = args.blind or args.suite
    ev = load_evaluation(args.run_dir, suite, args.tag, blind=bool(args.blind))
    roles = adapter_roles(ev.adapters, specs)
    members = [a for a in ev.adapters if roles[a]["role"] == "member"]
    if args.adapter:
        designated, source = args.adapter, "given"
    else:
        if not manifest_path(name).exists() and suite != "selection":
            raise SystemExit("no harvest manifest: name the designated adapter with --adapter")
        designated, _, source = pick_outlier(name, Suite(ev, seed=1), members)
        source = source.split(":")[0]
    comparators = (args.comparators.split(",") if args.comparators else
                   [a for a in ev.adapters if roles[a]["role"] in ("base", "baseline")])
    missing = [a for a in [designated, *comparators] if a not in ev.adapters]
    if missing or not comparators:
        raise SystemExit(f"not in this evaluation: {missing}" if missing else "empty comparator set")
    res = analyse(ev, roles, designated, comparators)
    out = Path(args.out or f"results/{name}/niches/{suite}")
    out.mkdir(parents=True, exist_ok=True)
    res["rates"].to_csv(out / "solve_rates.csv")
    for key in ("jaccard", "adapters", "unlocks", "designated"):
        res[key].to_csv(out / f"{key}.csv", index=False)
    pd.DataFrame([{"k": k, "mean_coverage": v} for k, v in res["coverage"]["k"].items()]).to_csv(
        out / "coverage.csv", index=False)
    print("\n".join(summary_lines(name, ev, res, source)))
    print(f"\nCSV: {out}/")


if __name__ == "__main__":
    main()
