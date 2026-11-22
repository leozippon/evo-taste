"""Run report: settings, base model, seeding pools, and training and probe statistics of every lineage.

    python -m evotaste.analysis.report runs/<name> [--families] [--out DIR]

Prints a compact fixed-width report and writes the same numbers as CSV to results/<name>/report/ (or DIR). Works on
runs in progress. Every statistic is defined in docs/dev/ANALYSIS_NOTES.md.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from evotaste.analysis.rundata import SUITES, Lineage, RunData
from evotaste.records import read_json

WINDOW = 10  # amplification steps averaged at the start and at the end
MAX_PROBE_STEPS = 4  # probe steps shown in the terminal (the first and the last three); the CSV has all
NAN = float("nan")


# -- statistics ---------------------------------------------------------------------------------------------------

def mean(xs) -> float:
    return float(np.mean(xs)) if len(xs) else NAN


def ols_slope(x, y) -> tuple[float, float]:
    """Least-squares slope of y on x and its classical standard error (NaN below 2 resp. 3 points)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return NAN, NAN
    xc = x - x.mean()
    sxx = float(xc @ xc)
    b = float(xc @ (y - y.mean())) / sxx
    if len(x) < 3:
        return b, NAN
    resid = y - y.mean() - b * xc
    return b, math.sqrt(float(resid @ resid) / (len(x) - 2) / sxx)


def paired_probe(records: list[dict], base_by_uid: dict[str, dict], suite: str) -> dict:
    """Macro accuracy (mean over families) of a member and of the base model on the same instances of one suite,
    their difference and its standard error: (1/F) sqrt(sum_f s_f^2 / n_f) with s_f^2 the sample variance of the
    paired differences member - base over the n_f instances of family f (NaN if some family has one instance)."""
    by_family: dict[str, list[tuple[float, float]]] = {}
    for r in records:
        if r["suite"] == suite:
            by_family.setdefault(r["family"], []).append((r["reward"], base_by_uid[r["uid"]]["reward"]))
    pairs = [np.array(v) for v in by_family.values()]
    acc = mean([p[:, 0].mean() for p in pairs])
    base = mean([p[:, 1].mean() for p in pairs])
    var = [np.var(p[:, 0] - p[:, 1], ddof=1) / len(p) if len(p) > 1 else NAN for p in pairs]
    return {"accuracy": acc, "base_accuracy": base, "diff": acc - base,
            "se": math.sqrt(sum(var)) / len(pairs) if pairs else NAN,
            "instances": sum(len(p) for p in pairs), "families": len(pairs)}


def lineage_row(lin: Lineage, hist: list[dict]) -> dict:
    """Training statistics of one lineage from its amplification history (steps done so far)."""
    first, last = hist[:WINDOW], hist[-WINDOW:]
    b, se = ols_slope([h["step"] for h in hist], [h["reward"] for h in hist])
    times = [datetime.fromisoformat(h["time"]).timestamp() for h in hist]
    return {"lineage": lin.id, "slot": lin.slot, "generation": lin.generation, "arm": lin.arm,
            "scaffold": lin.scaffold, "bias": lin.bias, "seeding_lr": lin.seeding_lr, "amp_lr": lin.amp_lr,
            "status": lin.status, "checkpoint": lin.checkpoint, "steps_done": len(hist), "steps_target": lin.amp_steps,
            "reward_first": mean([h["reward"] for h in first]), "reward_last": mean([h["reward"] for h in last]),
            "slope_per100": 100 * b, "slope_se_per100": 100 * se,
            "groups_mean": mean([h["groups"] for h in hist]), "prompts": hist[0]["prompts"] if hist else NAN,
            "tokens_first": mean([h["completion_tokens"] for h in first]),
            "tokens_last": mean([h["completion_tokens"] for h in last]),
            "parsed_last": mean([h["parsed"] for h in last]), "truncated_last": mean([h["truncated"] for h in last]),
            "check_token_mean_abs_max": max((h["check"]["token_mean_abs"] for h in hist), default=NAN),
            "s_per_step": float(np.median(np.diff(times))) if len(times) > 1 else NAN,
            "cull_reason": lin.cull_reason, "replaced_by": lin.replaced_by}


# -- formatting ---------------------------------------------------------------------------------------------------

def fmt(x, spec: str = ".1f", scale: float = 1.0) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else format(scale * x, spec)


def pct(x) -> str:
    return fmt(x, ".1f", 100)


def pm(x, se) -> str:
    """A rate difference and its standard error in points ('?' for an undefined SE)."""
    return "-" if math.isnan(x) else f"{100 * x:+.1f}±" + ("?" if math.isnan(se) else f"{100 * se:.1f}")


def render(headers: list[str], rows: list[list[str]], left: int = 1) -> list[str]:
    """Fixed-width table: the first `left` columns left-aligned, the others right-aligned."""
    widths = [max(len(str(c)) for c in col) for col in zip(headers, *rows)]

    def line(cells):
        return "  ".join(str(c).ljust(w) if i < left else str(c).rjust(w)
                         for i, (c, w) in enumerate(zip(cells, widths))).rstrip()

    return [line(headers)] + [line(r) for r in rows]


def lr(x: float) -> str:
    return f"{x:.0e}".replace("e-0", "e-")


# -- report -------------------------------------------------------------------------------------------------------

def build(run: RunData) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    """CSV tables, the report lines and the per-family lines."""
    cfg = run.cfg
    lines = []
    prepared = run.dir / "prepared.json"
    rev = read_json(prepared)["revision"][:12] if prepared.exists() else "not prepared"
    s, a, p = cfg.seeding, cfg.amplification, cfg.probes
    lines.append(f"run {cfg.name} | {cfg.model} | seed {cfg.seed} | revision {rev} | rates and accuracies in %")
    lines.append(f"seeding {s.pool_prompts} prompts x {s.samples} samples, {s.steps} updates x {s.batch}, lr "
                 f"{lr(s.lr)} | amplification {a.steps} steps x {a.prompts} prompts x {a.rollouts}, lr {lr(a.lr)}, "
                 f"max {a.max_tokens} tokens | probes every {p.every}: {p.train_per_family}/{p.selection_per_family} "
                 f"per train/selection family, {p.foundation_per_source} per foundation source")

    tables: dict[str, pd.DataFrame] = {}
    base_summary, base = run.base_summary(), run.base_responses()
    base_by_uid = {r["uid"]: r for r in base} if base else None
    if base_summary is None:
        lines.append("base probe: not yet written")
    else:
        acc = base_summary["accuracy"]
        nfam = {suite: len({r["family"] for r in base if r["suite"] == suite}) for suite in SUITES}
        lines.append(f"base probe (n={base_summary['n']}): " + " | ".join(
            f"{suite} {pct(acc[suite])} ({nfam[suite]} fam)" for suite in SUITES)
            + f" | parsed {pct(base_summary['parsed'])} truncated {pct(base_summary['truncated'])}"
            f" | {base_summary['completion_tokens']:.0f} tokens")
        tables["base"] = pd.DataFrame([{**{suite: acc[suite] for suite in SUITES}, "n": base_summary["n"],
                                        "parsed": base_summary["parsed"], "truncated": base_summary["truncated"],
                                        "completion_tokens": base_summary["completion_tokens"]}])

    # Seeding pools.
    pool_rows = []
    for teacher in run.teachers():
        summ = run.pool_summary(teacher)
        tokens = [len(r["completion_ids"]) for r in run.pool_records(teacher)]
        pool_rows.append({"teacher": teacher, "correct": summ["correct"], "samples": summ["samples"],
                          "prompts_with_correct": summ["prompts_with_correct"], "prompts": summ["prompts"],
                          "families_with_correct": sum(f["correct"] > 0 for f in summ["by_family"].values()),
                          "families": len(summ["by_family"]), "tokens_correct": mean(tokens)})
    tables["seeding"] = pd.DataFrame(pool_rows)
    lines.append("")
    if pool_rows:
        lines += render(["seeding pool", "correct", "of samples", "prompts ok", "of", "families ok", "of", "tokens"],
                        [[r["teacher"], r["correct"], r["samples"], r["prompts_with_correct"], r["prompts"],
                          r["families_with_correct"], r["families"], fmt(r["tokens_correct"], ".0f")]
                         for r in pool_rows])
    else:
        lines.append("seeding pools: none written yet")

    # Lineages: training.
    member_rows, probe_rows, family_rows = [], [], []
    histories = {}
    for lin in run.lineages:
        hist = lin.history()
        histories[lin.id] = hist
        member_rows.append(lineage_row(lin, hist))
        for step in lin.probe_steps():
            recs = lin.probe_responses(step)
            fmt_stats = {"parsed": mean([r["parsed"] for r in recs]), "truncated": mean([r["truncated"] for r in recs]),
                         "completion_tokens": mean([r["completion_tokens"] for r in recs])}
            if base_by_uid is not None and {r["uid"] for r in recs} != set(base_by_uid):
                raise ValueError(f"{lin.id} step {step}: probe instances differ from the base model's")
            for suite in SUITES:
                stats = (paired_probe(recs, base_by_uid, suite) if base_by_uid is not None else
                         {"accuracy": mean([r["reward"] for r in recs if r["suite"] == suite])})
                probe_rows.append({"lineage": lin.id, "step": step, "suite": suite, **stats, **fmt_stats})
    tables["members"] = pd.DataFrame(member_rows)
    tables["probes"] = pd.DataFrame(probe_rows)

    lines.append("")
    lines.append(f"lineages: rew0/rew1, tok0/tok1 over the first/last {WINDOW} steps done; slope of reward per 100 "
                 f"steps ± SE; groups: mean prompts with mixed rewards;")
    lines.append(f"  pars/trunc over the last {WINDOW}; chk: max check.token_mean_abs; s/step: median interval "
                 f"between steps")
    lines += render(
        ["lineage", "bias", "seedLR", "ampLR", "status", "steps", "rew0", "rew1", "slope±se", "groups", "tok0",
         "tok1", "pars", "trunc", "chk", "s/step"],
        [[r["lineage"], r["bias"], lr(r["seeding_lr"]), lr(r["amp_lr"]), r["status"],
          f"{r['steps_done']}/{r['steps_target']}", pct(r["reward_first"]), pct(r["reward_last"]),
          pm(r["slope_per100"], r["slope_se_per100"]), fmt(r["groups_mean"]),
          fmt(r["tokens_first"], ".0f"), fmt(r["tokens_last"], ".0f"), pct(r["parsed_last"]),
          pct(r["truncated_last"]), fmt(r["check_token_mean_abs_max"], ".4f"), fmt(r["s_per_step"], ".0f")]
         for r in member_rows])
    for r in member_rows:
        if r["cull_reason"]:
            lines.append(f"  culled {r['lineage']} at step {r['checkpoint']}: {r['cull_reason']}"
                         + (f" (replaced by {r['replaced_by']})" if r["replaced_by"] else " (slot exhausted)"))

    # Lineages: probes.
    steps = sorted({r["step"] for r in probe_rows})
    shown = steps if len(steps) <= MAX_PROBE_STEPS else steps[:1] + steps[-(MAX_PROBE_STEPS - 1):]
    lines.append("")
    if not steps:
        lines.append("probes: no lineage has a probe yet")
    elif base_by_uid is None:
        lines.append("probes: no base-model probe to compare with; accuracies in probes.csv")
    else:
        cell = {(r["lineage"], r["step"], r["suite"]): r for r in probe_rows}
        short = {"train": "tr", "selection": "sel", "foundation": "fnd"}
        lines.append("probe accuracy minus base, points ± paired SE (base: " + ", ".join(
            f"{suite} {pct(base_summary['accuracy'][suite])}" for suite in SUITES) + ")"
            + (f"; steps {', '.join(map(str, shown))} of {len(steps)}" if shown != steps else ""))
        rows = []
        for lin in run.lineages:
            row = [lin.id]
            for step in shown:
                row += [pm(cell[k]["diff"], cell[k]["se"]) if (k := (lin.id, step, suite)) in cell else ""
                        for suite in SUITES]
            if any(row[1:]):
                rows.append(row)
        lines += render(["lineage"] + [f"{short[suite]}@{step}" for step in shown for suite in SUITES], rows)

    # Per train family: base probe accuracy and rollout reward over the first and the last steps.
    family_lines = []
    if base:
        train_families = sorted({r["family"] for r in base if r["suite"] == "train"})
        base_fam = {f: [r["reward"] for r in base if r["family"] == f] for f in train_families}
        active = [lin for lin in run.lineages if histories[lin.id]]
        for f in train_families:
            for lin in active:
                hist = histories[lin.id]
                family_rows.append({
                    "family": f, "base_accuracy": mean(base_fam[f]), "base_n": len(base_fam[f]), "lineage": lin.id,
                    "reward_first": mean([h["reward_by_family"][f] for h in hist[:WINDOW] if f in h["reward_by_family"]]),
                    "reward_last": mean([h["reward_by_family"][f] for h in hist[-WINDOW:] if f in h["reward_by_family"]])})
        cell = {(r["family"], r["lineage"]): r for r in family_rows}
        family_lines.append(f"train families: base probe accuracy, then rollout reward first>last {WINDOW} steps "
                            f"per lineage (%)")
        family_lines.append("lineages: " + " ".join(f"{i + 1}={lin.id}" for i, lin in enumerate(active)))
        family_lines += render(
            ["family", "base"] + [str(i + 1) for i in range(len(active))],
            [[f, pct(mean(base_fam[f]))] + [f"{fmt(c['reward_first'], '.0f', 100)}>{fmt(c['reward_last'], '.0f', 100)}"
                                            for lin in active for c in [cell[(f, lin.id)]]]
             for f in train_families])
    tables["families"] = pd.DataFrame(family_rows)
    return tables, lines, family_lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--families", action="store_true", help="also print the per-family table")
    ap.add_argument("--out", help="CSV directory (default results/<name>/report)")
    args = ap.parse_args(argv)
    run = RunData(args.run_dir)
    tables, lines, family_lines = build(run)
    out = Path(args.out or f"results/{run.name}/report")
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    if args.families:
        lines += [""] + (family_lines or ["train families: no base-model probe yet"])
    print("\n".join(lines))
    print(f"\nCSV: {out}/")


if __name__ == "__main__":
    main()
