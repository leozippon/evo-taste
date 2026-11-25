"""Wash-out of seeded contexts: the context index of every probe, the specific index of each seeded member against the
controls, identification of the member's own context, stances against tactics paired by theme, and recitation.

    python -m evotaste.analysis.washout runs/<name> [--table runs/<name>/context_score/<selection>/table.csv]
                                         [--out DIR]

Reads the engine's context-score table (python -m evotaste.evaluate context-score; the run's only selection unless
--table names one) and the probe answers (for the jargon share). The contexts are those the table scores. Writes results/<name>/washout/: index.csv, specific.csv, identification.csv, themes.csv, noise.csv,
jargon.csv and report.txt. Definitions: docs/dev/ANALYSIS_NOTES.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from threadpoolctl import threadpool_limits

from evotaste.analysis import data, stats
from evotaste.analysis.report import num, pct, render, write

DOMAINS_ALL = "all"


def index_table(run: data.Run, t: pd.DataFrame) -> pd.DataFrame:
    """Per subject, context and domain (and over all domains, token-weighted): S, the mean log-ratio per token, and
    the index I = (S(subject) - S(base)) / (S(base under the context) - S(base))."""
    over = t.groupby(["subject", "context"], as_index=False)[["delta_sum", "tokens"]].sum().assign(domain=DOMAINS_ALL)
    s = pd.concat([t[["subject", "context", "domain", "delta_sum", "tokens"]], over], ignore_index=True)
    s["S"] = s["delta_sum"] / s["tokens"]
    key = s.set_index(["subject", "context", "domain"])["S"]
    missing = [c for c in sorted(t["context"].unique()) if (f"base:{c}", c, DOMAINS_ALL) not in key.index]
    if "base" not in key.index.get_level_values(0) or missing:
        raise ValueError(f"context-score table lacks the base references (base, base:<context> for {missing})")
    base = key.loc["base"]
    s["S_base"] = [base[(c, d)] for c, d in zip(s["context"], s["domain"])]
    s["S_teacher"] = [key[(f"base:{c}", c, d)] for c, d in zip(s["context"], s["domain"])]
    span = s["S_teacher"] - s["S_base"]
    s["I"] = np.where(span > 0, (s["S"] - s["S_base"]) / span.where(span > 0), np.nan)
    lineage = t.drop_duplicates("subject").set_index("subject")["lineage"]
    step = t.drop_duplicates("subject").set_index("subject")["step"]
    s.insert(1, "lineage", s["subject"].map(lineage))
    s.insert(2, "step", s["subject"].map(step))
    return s.drop(columns=["delta_sum"])


def specific_table(run: data.Run, idx: pd.DataFrame) -> pd.DataFrame:
    """Per seeded member probe and domain: raw = I_c - mean_{j != c} I_j for its own context c, control = the same
    expression for the controls (placebo and unseeded members) at the same step under the same update rule, specific
    = raw - control, and the ratio of specific to that of the lineage's seeded state at step 0."""
    members = idx[idx["lineage"].notna()]
    wide = members.pivot_table(index=["subject", "domain"], columns="context", values="I")
    info = members.drop_duplicates("subject").set_index("subject")[["lineage", "step"]]
    rows = []
    for (subject, domain), r in wide.iterrows():
        lin = run.lineages[info.loc[subject, "lineage"]]
        rows.append({"subject": subject, "lineage": lin.id, "step": int(info.loc[subject, "step"]), "domain": domain,
                     "arm": lin.arm, "level": lin.level, "context": lin.context, "theme": lin.theme,
                     "seeding": lin.seeding, "control_member": lin.control, "root": root(run, lin).id, "I": r})
    out = []
    for r in rows:
        c = r["context"]
        if r["control_member"] or c not in r["I"].index:  # a control, or its context is not scored
            continue
        own = lambda x: x[c] - x.drop(c).mean()  # noqa: E731
        ctrl = [own(q["I"]) for q in rows if q["control_member"] and q["step"] == r["step"]
                and q["arm"] == r["arm"] and q["domain"] == r["domain"]]
        raw = own(r["I"])
        out.append({k: r[k] for k in ("subject", "lineage", "root", "step", "domain", "arm", "level", "context",
                                      "theme", "seeding")} | {"I_own": r["I"][c], "raw": raw, "controls": len(ctrl),
                                                              "control": np.mean(ctrl) if ctrl else np.nan,
                                                              "specific": raw - np.mean(ctrl) if ctrl else np.nan})
    out = pd.DataFrame(out)
    if len(out):
        at0 = out[out["step"] == 0].set_index(["lineage", "domain"])["specific"]
        out["ratio_to_0"] = [s / at0.get((lin, d), np.nan) for s, lin, d in zip(out["specific"], out["root"],
                                                                                out["domain"])]
    return out


def root(run: data.Run, lin: data.Lineage) -> data.Lineage:
    """The lineage whose seeding a clone continues."""
    return run.lineages[lin.parent[0]] if lin.parent else lin


def candidates(idx: pd.DataFrame) -> list[str]:
    """The contexts a member's own context is ranked among: every scored context but the placebo."""
    return sorted(c for c in idx["context"].unique() if c != "placebo")


def own_context_hits(run: data.Run, idx: pd.DataFrame) -> pd.DataFrame:
    """Per seeded member probe with controls at the same step and update rule: whether its own context ranks first
    among the candidates after subtracting the controls' mean index per context (over all domains)."""
    members = idx[idx["lineage"].notna() & (idx["domain"] == DOMAINS_ALL)]
    wide = members.pivot_table(index="subject", columns="context", values="I")
    meta = members.drop_duplicates("subject").set_index("subject")
    lin_of = {s: run.lineages[meta.loc[s, "lineage"]] for s in wide.index}
    step_of = {s: int(meta.loc[s, "step"]) for s in wide.index}
    hits = []
    for subject, r in wide.iterrows():
        lin, step = lin_of[subject], step_of[subject]
        ctrl = [s for s in wide.index if lin_of[s].control and step_of[s] == step and lin_of[s].arm == lin.arm]
        if lin.control or not ctrl or lin.context not in candidates(idx):
            continue
        corrected = (r - wide.loc[ctrl].mean())[candidates(idx)]
        hits.append({"step": step, "arm": lin.arm, "subject": subject, "lineage": lin.id,
                     "hit": corrected.idxmax() == lin.context})
    return pd.DataFrame(hits, columns=["step", "arm", "subject", "lineage", "hit"])


def identification(idx: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    """Per step and update rule (and pooled over rules): the identification rate, its Clopper-Pearson interval and the
    exact binomial test against chance (one over the number of candidates)."""
    chance = 1 / len(candidates(idx))
    rows = []
    for (step, arm), g in list(hits.groupby(["step", "arm"])) + [((s, "all"), g) for s, g in hits.groupby("step")]:
        k, n = int(g["hit"].sum()), len(g)
        rows.append({"step": step, "arm": arm, "members": n, "identified": k, "rate": k / n,
                     **dict(zip(("lo", "hi"), stats.clopper_pearson(k, n))), "chance": chance,
                     "p": float(sps.binomtest(k, n, chance, alternative="greater").pvalue)})
    return pd.DataFrame(rows)


def themes(run: data.Run, spec: pd.DataFrame, rng, b: int = stats.B) -> pd.DataFrame:
    """Stances against tactics paired by theme, per step, update rule, seeding variant and domain: the difference of
    specific indices (members of one context averaged), its mean over themes with a bootstrap interval over themes and
    the exact sign-flip p-value."""
    rows = []
    ok = spec[spec["specific"].notna() & spec["level"].isin(["stance", "tactic"])]
    for key, g in ok.groupby(["step", "arm", "seeding", "domain"]):
        per = g.groupby(["theme", "level"])["specific"].mean().unstack()
        if not {"stance", "tactic"} <= set(per.columns):
            continue
        d = (per["stance"] - per["tactic"]).dropna()
        if not len(d):
            continue
        draws = d.to_numpy()[rng.integers(0, len(d), (b, len(d)))].mean(1)
        rows.append(dict(zip(("step", "arm", "seeding", "domain"), key)) | {
            "themes": len(d), "pairs": " ".join(f"{t}:{v:+.3f}" for t, v in d.items()),
            "stance_minus_tactic": d.mean(), **dict(zip(("lo", "hi"), stats.ci(draws))),
            "p_sign_flip": stats.sign_flip_p(d.to_numpy())})
    return pd.DataFrame(rows)


def twin_noise(run: data.Run, idx: pd.DataFrame) -> pd.DataFrame:
    """The sampling noise of one index value, from probes of one adapter: at a clone point, the parent and each of
    its clones probe the same checkpoint with their own seeds (a twin at step 0 included). Per domain, over every pair
    of such probes: noise_I = sqrt(mean over contexts j of (I_j(a) - I_j(b))^2 / 2), and the same for the raw
    own-context contrast I_c - mean_{j != c} I_j taken over every context c, which is the noise of a raw value."""
    wide = idx[idx["lineage"].notna()].pivot_table(index=["lineage", "step", "domain"], columns="context", values="I")
    names = sorted(idx["context"].unique())
    groups = {}
    for lin in run.lineages.values():
        if lin.parent:
            groups.setdefault(lin.parent, {lin.parent[0]}).add(lin.id)
    rows = []
    for domain in sorted(idx["domain"].unique()):
        di, dr, pairs = [], [], 0
        for (parent, step), lineages in groups.items():
            have = sorted(x for x in lineages if (x, step, domain) in wide.index)
            for i, a in enumerate(have):
                for b in have[i + 1:]:
                    ia, ib = wide.loc[(a, step, domain), names], wide.loc[(b, step, domain), names]
                    di += list(ia - ib)
                    dr += [(ia[c] - ia.drop(c).mean()) - (ib[c] - ib.drop(c).mean()) for c in names]
                    pairs += 1
        if pairs:  # contexts without an index in this domain (see index_table) are left out
            rows.append({"domain": domain, "pairs": pairs, "noise_I": float(np.sqrt(np.nanmean(np.square(di)) / 2)),
                         "noise_raw": float(np.sqrt(np.nanmean(np.square(dr)) / 2))})
    return pd.DataFrame(rows, columns=["domain", "pairs", "noise_I", "noise_raw"])


def jargon_table(run: data.Run) -> pd.DataFrame:
    """Share of probe answers containing any term of the jargon list (case-insensitive substring)."""
    terms = [t.lower() for t in run.contexts.jargon]

    def row(subject, lineage, step, records):
        hits = [any(t in r["text"].lower() for t in terms) for r in records]
        return {"subject": subject, "lineage": lineage, "step": step, "responses": len(hits),
                "with_jargon": int(sum(hits)), "share": sum(hits) / len(hits)}

    rows = [row("base", None, None, run.base_records())]
    rows += [row(f"base:{c}", None, None, run.base_records(c)) for c in run.contexts.names]
    for lin in run.lineages.values():
        rows += [row(f"{lin.id}@{s}", lin.id, s, run.probe_records(lin, s)) for s in run.probe_steps(lin)]
    return pd.DataFrame(rows)


NO_ITEM_INTERVALS = ("intervals over items are not available: the context-score table has no per-response rows "
                     "(see ANALYSIS_NOTES.md); intervals below resample members or themes")


def noise_line(noise: pd.DataFrame) -> str:
    return "sampling noise of one value (SD), from probes of one checkpoint by a parent and its clones: " + (
        ", ".join(f"{r.domain} index {r.noise_I:.3f}, raw {r.noise_raw:.3f} ({r.pairs} pairs)"
                  for r in noise.itertuples()) if len(noise) else "no clone point with two scored probes")


def report_lines(run, idx, spec, ident, th, jargon, noise) -> list[str]:
    lines = [f"washout {run.name}: index 0 = base model, 1 = its answers under the context; specific = own-context "
             "index minus the mean of the other contexts, minus the same for the controls at that step and rule",
             NO_ITEM_INTERVALS, noise_line(noise)]
    bad = idx[(idx["subject"] == "base") & ~(idx["S_teacher"] > idx["S_base"])]
    if len(bad):
        lines.append("no index (the context does not make its instructed answers more likely than the plain ones): "
                     + ", ".join(f"{r.context}/{r.domain}" for r in bad.itertuples()))
    if len(spec):
        a = spec[spec["domain"] == DOMAINS_ALL].sort_values(["root", "lineage", "step"])
        lines += ["", "specific index over steps (all domains; ratio to the seeded state at step 0)"]
        lines += render(["probe", "rule", "level", "context", "seeding", "I own", "raw", "control", "specific",
                         "ratio"],
                        [[r.subject, r.arm, r.level, r.context, r.seeding, num(r.I_own, digits=3), num(r.raw, digits=3),
                          num(r.control, digits=3), num(r.specific, digits=3), num(r.ratio_to_0)]
                         for r in a.itertuples()])
        cross = spec[spec["seeding"].str.contains(r"\[") & (spec["domain"] != DOMAINS_ALL)]
        if len(cross):
            lines += ["", "cross-domain coherence: specific index by domain for members seeded on some domains only"]
            lines += render(["probe", "seeding", "domain", "specific"],
                            [[r.subject, r.seeding, r.domain, num(r.specific, digits=3)] for r in cross.itertuples()])
    else:
        lines += ["", "no seeded member with a context score yet"]
    lines += ["", "identification of the own context (rank 1 of the candidates after subtracting the controls)"]
    lines += render(["step", "rule", "members", "identified", "rate [CP 95%]", "chance", "p"],
                    [[r.step, r.arm, r.members, r.identified, f"{pct(r.rate, 0)} [{pct(r.lo, 0)}, {pct(r.hi, 0)}]",
                      pct(r.chance, 0), f"{r.p:.3f}"] for r in ident.itertuples()]) if len(ident) else [
        "  needs seeded members and controls at the same step"]
    if len(th):
        lines += ["", "stance - tactic, paired by theme (specific index; interval resamples themes)"]
        lines += render(["step", "rule", "seeding", "domain", "themes", "mean [95%]", "sign-flip p"],
                        [[r.step, r.arm, r.seeding, r.domain, r.themes, num(r.stance_minus_tactic, r.lo, r.hi, 3),
                          f"{r.p_sign_flip:.3f}"] for r in th.itertuples()])
    else:
        lines += ["", "stance - tactic: no theme with both a stance and a tactic member scored at the same step"]
    j = jargon[jargon["with_jargon"] > 0]
    lines += ["", f"jargon: {int(jargon['with_jargon'].sum())} of {int(jargon['responses'].sum())} probe answers "
                  "contain a listed term" + ("" if not len(j) else ": " + ", ".join(
                      f"{r.subject} {r.with_jargon}" for r in j.itertuples()))]
    return lines


def analyse(run: data.Run, rng, table: Path | None = None) -> dict[str, pd.DataFrame]:
    idx = index_table(run, data.context_scores(run, table))
    spec = specific_table(run, idx)
    hits = own_context_hits(run, idx)
    return {"index": idx, "specific": spec, "hits": hits, "identification": identification(idx, hits),
            "themes": themes(run, spec, rng), "noise": twin_noise(run, idx), "jargon": jargon_table(run)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--table", type=Path, help="a context-score table (default: the run's only selection)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    run = data.Run(args.run_dir)
    tables = analyse(run, np.random.default_rng(0), args.table)
    write(args.out or Path("results") / run.name / "washout", tables,
          report_lines(run, tables["index"], tables["specific"], tables["identification"], tables["themes"],
                       tables["jargon"], tables["noise"]))


if __name__ == "__main__":
    with threadpool_limits(stats.THREADS):
        main()
