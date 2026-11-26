"""Wash-out of seeded contexts: the context index of every probe, the specific index of each seeded member against the
controls with its two parts in nats (what the member gained, how far the context moves the teacher), identification
of the member's own context, stances against tactics paired by theme, and recitation.

    python -m evotaste.analysis.washout runs/<name> [--selection runs/<name>/context_score/<selection>] [--out DIR]

Reads the engine's per-response context scores (python -m evotaste.evaluate context-score; the run's only selection
unless --selection names one) and the probe answers (for the jargon share). Intervals resample the selection's items
within unit kinds and domains (data.strata), the same draws for every subject. Writes results/<name>/washout/: index.csv, specific.csv, hits.csv,
identification.csv, themes.csv, noise.csv, rules.csv, jargon.csv and report.txt. Definitions: docs/dev/ANALYSIS_NOTES.md.
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


class Index:
    """S_j(x), the mean log-ratio per token of context j on subject x's answers, over all items and within each
    domain, for the point estimate (draw 0) and B item draws (draws 1..B): arrays scope -> draw x subject x context.
    g = S - S(base); e_j = S_j(base under j) - S_j(base), the context's effect on the teacher; I = g / e, set to NaN
    for a context and scope where the point estimate of e is not positive."""

    def __init__(self, sc: data.Scores, units: dict[str, str], rng, b: int = stats.B):
        self.sc = sc
        kinds = pd.DataFrame({"unit": [units[u.split("/")[0]] for u in sc.uids], "domain": sc.domains})
        w = np.vstack([np.ones(len(sc.uids)), stats.item_draws(data.strata(kinds), b, rng)])
        self.scopes = [DOMAINS_ALL] + sorted(set(sc.domains))
        pos = {s: i for i, s in enumerate(sc.subjects)}
        missing = [c for c in sc.contexts if f"base:{c}" not in pos]
        if "base" not in pos or missing:
            raise ValueError(f"{sc.dir}: no base references (base, base:<context> for {missing})")
        self.base = pos["base"]
        teacher = [pos[f"base:{c}"] for c in sc.contexts]
        self.S, self.g, self.e, self.I = {}, {}, {}, {}
        for scope in self.scopes:
            keep = np.ones(len(sc.uids), bool) if scope == DOMAINS_ALL else sc.domains == scope
            num_ = np.einsum("scn,bn->bsc", sc.delta[:, :, keep], w[:, keep])
            den = (sc.tokens[:, keep] @ w[:, keep].T).T[:, :, None]
            s = num_ / den
            g = s - s[:, self.base:self.base + 1, :]
            e = s[:, teacher, np.arange(len(sc.contexts))] - s[:, self.base, :]
            ok = e[0] > 0
            with np.errstate(invalid="ignore", divide="ignore"):
                self.I[scope] = np.where(ok, g / np.where(ok, e, 1)[:, None, :], np.nan)
            self.S[scope], self.g[scope], self.e[scope] = s, g, e

    def table(self) -> pd.DataFrame:
        rows = []
        for scope in self.scopes:
            for si, subject in enumerate(self.sc.subjects):
                for ci, c in enumerate(self.sc.contexts):
                    i = self.I[scope][:, si, ci]
                    rows.append({"subject": subject, "lineage": self.sc.lineage[si], "step": self.sc.step[si],
                                 "context": c, "domain": scope, "S": self.S[scope][0, si, ci],
                                 "S_base": self.S[scope][0, self.base, ci], "effect": self.e[scope][0, ci],
                                 "I": i[0], **dict(zip(("I_lo", "I_hi"), stats.ci(i[1:])))})
        return pd.DataFrame(rows)


def member_probes(run: data.Run, idx: Index) -> list[tuple[int, data.Lineage, int]]:
    """(subject position, lineage, step) of the scored member probes, without clones' first probes (see
    data.Run.restart)."""
    out = []
    for si, (lin_id, step) in enumerate(zip(idx.sc.lineage, idx.sc.step)):
        if lin_id is not None and not run.restart(run.lineages[lin_id], step):
            out.append((si, run.lineages[lin_id], step))
    return out


def own(i: np.ndarray, c: int) -> np.ndarray:
    """I_c - mean_{j != c} I_j over the last axis (contexts), per draw."""
    return i[..., c] - np.nanmean(np.delete(i, c, axis=-1), axis=-1)


def specific_table(run: data.Run, idx: Index) -> tuple[pd.DataFrame, dict]:
    """Per seeded member probe and scope: raw = I_c - mean_{j != c} I_j for its own context c; control = the mean of
    the same for the controls (placebo and unseeded members) at the same step under the same update rule; specific =
    raw - control; its two parts, effect = e_c (nats per token by which the context moves the teacher) and gain =
    specific x effect (the member's specific gain in nats per token, so that specific = gain / effect); the plain gain
    g_c = S_c(member) - S_c(base); and the ratio of specific to the root's at step 0. Point values and intervals; also
    the specific index's draws by (lineage, step, scope)."""
    probes = member_probes(run, idx)
    ctx = {c: i for i, c in enumerate(idx.sc.contexts)}
    rows, draws = [], {}
    for scope in idx.scopes:
        I, g, e = idx.I[scope], idx.g[scope], idx.e[scope]
        for si, lin, step in probes:
            if lin.control or lin.context not in ctx:
                continue
            c = ctx[lin.context]
            ctrl = [k for k, kl, ks in probes if kl.control and ks == step and kl.arm == lin.arm]
            raw = own(I[:, si, :], c)
            control = np.mean([own(I[:, k, :], c) for k in ctrl], axis=0) if ctrl else np.full_like(raw, np.nan)
            spec = raw - control
            draws[(lin.id, step, scope)] = spec
            values = {"I_own": I[:, si, c], "raw": raw, "control": control, "specific": spec, "effect": e[:, c],
                      "gain": spec * e[:, c], "plain_gain": g[:, si, c]}
            row = {"subject": idx.sc.subjects[si], "lineage": lin.id, "root": run.root(lin).id, "step": step,
                   "domain": scope, "arm": lin.arm, "level": lin.level, "context": lin.context, "theme": lin.theme,
                   "seeding": lin.seeding, "controls": len(ctrl)}
            for k, v in values.items():
                row[k] = v[0]
                row[f"{k}_lo"], row[f"{k}_hi"] = stats.ci(v[1:])
            rows.append(row)
    for row in rows:
        at0 = draws.get((row["root"], 0, row["domain"]))
        ratio = draws[(row["lineage"], row["step"], row["domain"])] / at0 if at0 is not None else np.full(1, np.nan)
        row["ratio_to_0"] = ratio[0]
        row["ratio_lo"], row["ratio_hi"] = stats.ci(ratio[1:])
    return pd.DataFrame(rows), draws


def candidates(idx: Index) -> list[str]:
    """The contexts a member's own context is ranked among: every scored context but the placebo."""
    return [c for c in idx.sc.contexts if c != "placebo"]


def own_context_hits(run: data.Run, idx: Index) -> pd.DataFrame:
    """Per seeded member probe with controls at the same step and update rule: whether its own context ranks first
    among the candidates after subtracting the controls' mean index per context (over all domains, point values)."""
    cand = [idx.sc.contexts.index(c) for c in candidates(idx)]
    hits = []
    probes = member_probes(run, idx)
    I = idx.I[DOMAINS_ALL][0]
    for si, lin, step in probes:
        ctrl = [k for k, kl, ks in probes if kl.control and ks == step and kl.arm == lin.arm]
        if lin.control or not ctrl or lin.context not in candidates(idx) or len(cand) < 2:
            continue
        corrected = (I[si] - I[ctrl].mean(0))[cand]
        if np.isfinite(corrected).any():
            top = idx.sc.contexts[cand[int(np.nanargmax(corrected))]]
            hits.append({"step": step, "arm": lin.arm, "subject": idx.sc.subjects[si], "lineage": lin.id,
                         "hit": top == lin.context, "top": top})
    return pd.DataFrame(hits, columns=["step", "arm", "subject", "lineage", "hit", "top"])


def identification(idx: Index, hits: pd.DataFrame) -> pd.DataFrame:
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


def themes(spec: pd.DataFrame, draws: dict, rng, b: int = stats.B) -> pd.DataFrame:
    """Stances against tactics paired by theme, per step, update rule, seeding variant and scope: the difference of
    specific indices (members of one context averaged), its mean over themes with an interval over items (the
    measurement) and one over themes (generalisation to other themes), and the exact sign-flip p-value."""
    rows = []
    ok = spec[spec["specific"].notna() & spec["level"].isin(["stance", "tactic"])]
    for key, g in ok.groupby(["step", "arm", "seeding", "domain"]):
        per = g.groupby(["theme", "level"])["specific"].mean().unstack()
        if not {"stance", "tactic"} <= set(per.columns):
            continue
        d = (per["stance"] - per["tactic"]).dropna()
        if not len(d):
            continue
        boot = rng.choice(d.to_numpy(), (b, len(d))).mean(1)
        per_draw = {(t, lv): np.mean([draws[(r.lineage, r.step, r.domain)] for r in g.itertuples()
                                      if (r.theme, r.level) == (t, lv)], axis=0)
                    for t in d.index for lv in ("stance", "tactic")}
        mean_draws = np.mean([per_draw[(t, "stance")] - per_draw[(t, "tactic")] for t in d.index], axis=0)
        rows.append(dict(zip(("step", "arm", "seeding", "domain"), key)) | {
            "themes": len(d), "pairs": " ".join(f"{t}:{v:+.3f}" for t, v in d.items()),
            "stance_minus_tactic": d.mean(), **dict(zip(("lo_themes", "hi_themes"), stats.ci(boot))),
            **dict(zip(("lo_items", "hi_items"), stats.ci(mean_draws[1:]))),
            "p_sign_flip": stats.sign_flip_p(d.to_numpy())})
    return pd.DataFrame(rows)


def twin_noise(run: data.Run, idx: Index) -> pd.DataFrame:
    """The sampling noise of one value, from probes of one checkpoint: at a clone point, the parent and each of its
    clones probe the same checkpoint with their own seeds (step-0 twins included). Per step of the shared checkpoint
    and scope, over every pair of such probes: noise_I = sqrt(mean over contexts j of (I_j(a) - I_j(b))^2 / 2), and
    the same for the raw contrast I_c - mean_{j != c} I_j taken over every context c."""
    groups = {}
    for si, (lin_id, step) in enumerate(zip(idx.sc.lineage, idx.sc.step)):
        if lin_id is not None:
            groups.setdefault(run.state(run.lineages[lin_id], step), []).append(si)
    rows = []
    for at in sorted({st for _, st in groups}):
        for scope in idx.scopes:
            I = idx.I[scope][0]
            di, dr, pairs = [], [], 0
            for (_, st), members in groups.items():
                if st != at:
                    continue
                for x, a in enumerate(members):
                    for b_ in members[x + 1:]:
                        di += list(I[a] - I[b_])
                        dr += [own(I[a], c) - own(I[b_], c) for c in range(I.shape[1])]
                        pairs += 1
            if pairs:  # contexts without an index in this scope are left out
                rows.append({"step": at, "domain": scope, "pairs": pairs,
                             "noise_I": float(np.sqrt(np.nanmean(np.square(di)) / 2)),
                             "noise_raw": float(np.sqrt(np.nanmean(np.square(dr)) / 2))})
    return pd.DataFrame(rows, columns=["step", "domain", "pairs", "noise_I", "noise_raw"])


def rules_table(run: data.Run, spec: pd.DataFrame, draws: dict) -> pd.DataFrame:
    """Between update rules: per seeded state (root) trained from step 0 under both rules (the root itself or its
    step-0 twins), the difference of specific indices at each step and scope with an item interval; and per step,
    scope and level the mean over roots, with an item interval and the exact sign-flip p-value over roots."""
    rules = sorted(spec["arm"].unique())
    if len(rules) != 2:
        return pd.DataFrame()
    a, b_ = rules
    rows = []
    trained = spec[spec["step"] > 0]
    for (root, step, scope), g in trained.groupby(["root", "step", "domain"]):
        by = {r.arm: r for r in g.itertuples() if run.state(run.lineages[r.lineage], 0) == (root, 0)}
        if set(by) != {a, b_}:
            continue
        d = draws[(by[a].lineage, step, scope)] - draws[(by[b_].lineage, step, scope)]
        rows.append({"what": "root", "root": root, "level": by[a].level, "step": step, "domain": scope,
                     "roots": 1, "difference": d[0], **dict(zip(("lo", "hi"), stats.ci(d[1:]))), "_d": d})
    per = pd.DataFrame(rows)
    out = [r for r in rows]
    if len(per):
        for (level, step, scope), g in per.groupby(["level", "step", "domain"]):
            d = np.mean(list(g["_d"]), axis=0)
            out.append({"what": "mean", "root": "", "level": level, "step": step, "domain": scope,
                        "roots": len(g), "difference": d[0], **dict(zip(("lo", "hi"), stats.ci(d[1:]))),
                        "p_sign_flip": stats.sign_flip_p(g["difference"].to_numpy())})
    return pd.DataFrame(out).drop(columns="_d").assign(rules=f"{a} - {b_}")


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


def noise_line(noise: pd.DataFrame) -> str:
    return "noise of one value (SD), from probes of one checkpoint by a parent and its clones: " + (
        "; ".join(f"step {r.step} {r.domain}: index {r.noise_I:.3f}, raw {r.noise_raw:.3f} ({r.pairs} pairs)"
                  for r in noise.itertuples()) if len(noise) else "no clone point with two scored probes")


def selection_lines(idx: Index) -> list[str]:
    sc = idx.sc
    lines = [f"selection {sc.dir.name}: {len(sc.uids)} probe items (" + ", ".join(
        f"{d} {int((sc.domains == d).sum())}" for d in idx.scopes[1:]) + f"), contexts {', '.join(sc.contexts)}",
        "[95% intervals resample the items within unit kinds and domains]"]
    bad = [f"{c}/{s}" for s in idx.scopes for ci, c in enumerate(sc.contexts) if not idx.e[s][0, ci] > 0]
    if bad:
        lines.append("no index (the context does not make its instructed answers more likely than the plain ones): "
                     + ", ".join(bad))
    return lines


def report_lines(run, res) -> list[str]:
    idx, spec, ident, th, jargon, noise, rules = (res[k] for k in ("_index", "specific", "identification", "themes",
                                                                  "jargon", "noise", "rules"))
    lines = [f"washout {run.name}: index 0 = base model, 1 = its answers under the context; specific = own-context "
             "index minus the mean of the other contexts, minus the same for the controls at that step and rule; "
             "specific = gain / effect, both in nats per token"] + selection_lines(idx) + [noise_line(noise)]
    if len(spec):
        a = spec[spec["domain"] == DOMAINS_ALL].sort_values(["root", "lineage", "step"])
        lines += ["", "specific index over steps (all domains; ratio to the seeded state at step 0)"]
        lines += render(["probe", "rule", "level", "context", "seeding", "raw", "specific [95%]", "gain [95%]",
                         "effect [95%]", "ratio"],
                        [[r.subject, r.arm, r.level, r.context, r.seeding, num(r.raw, digits=3),
                          num(r.specific, r.specific_lo, r.specific_hi, 3), num(r.gain, r.gain_lo, r.gain_hi, 4),
                          num(r.effect, r.effect_lo, r.effect_hi, 4), ratio(r, a)] for r in a.itertuples()])
        cross = spec[spec["seeding"].str.contains(r"\[") & (spec["domain"] != DOMAINS_ALL)]
        if len(cross):
            lines += ["", "cross-domain coherence: specific index by domain for members seeded on some domains only"]
            lines += render(["probe", "seeding", "domain", "specific [95%]"],
                            [[r.subject, r.seeding, r.domain, num(r.specific, r.specific_lo, r.specific_hi, 3)]
                             for r in cross.itertuples()])
    else:
        lines += ["", "no seeded member with a context score yet"]
    lines += ["", "identification of the own context (rank 1 of the candidates after subtracting the controls)"]
    lines += render(["step", "rule", "members", "identified", "rate [CP 95%]", "chance", "p"],
                    [[r.step, r.arm, r.members, r.identified, f"{pct(r.rate, 0)} [{pct(r.lo, 0)}, {pct(r.hi, 0)}]",
                      pct(r.chance, 0), f"{r.p:.3f}"] for r in ident.itertuples()]) if len(ident) else [
        "  needs seeded members and controls at the same step, and at least two candidate contexts"]
    if len(th):
        lines += ["", "stance - tactic, paired by theme (specific index; intervals over items and over themes)"]
        lines += render(["step", "rule", "seeding", "domain", "themes", "mean [items]", "[themes]", "sign-flip p"],
                        [[r.step, r.arm, r.seeding, r.domain, r.themes,
                          num(r.stance_minus_tactic, r.lo_items, r.hi_items, 3),
                          f"[{r.lo_themes:.3f}, {r.hi_themes:.3f}]", f"{r.p_sign_flip:.3f}"] for r in th.itertuples()])
    else:
        lines += ["", "stance - tactic: no theme with both a stance and a tactic member scored at the same step"]
    if len(rules):
        m = rules[(rules["what"] == "mean") & (rules["domain"] == DOMAINS_ALL)]
        lines += ["", f"update rules, {rules['rules'].iloc[0]}: specific index, mean over seeded states trained under "
                      "both (interval over items; sign-flip p over states)"]
        lines += render(["step", "level", "states", "difference [95%]", "sign-flip p"],
                        [[r.step, r.level, r.roots, num(r.difference, r.lo, r.hi, 3), f"{r.p_sign_flip:.3f}"]
                         for r in m.itertuples()], 2)
    base = jargon.loc[jargon["subject"] == "base", "share"].iloc[0]
    members = jargon[jargon["lineage"].notna()]
    top = members.nlargest(3, "share")
    lines += ["", f"jargon: {pct(members['with_jargon'].sum() / members['responses'].sum())}% of member probe answers "
                  f"contain a listed term (base model {pct(base)}%); highest: "
                  + ", ".join(f"{r.subject} {pct(r.share)}%" for r in top.itertuples())]
    return lines


def ratio(r, a: pd.DataFrame) -> str:
    """The ratio to step 0, shown only when the root's step-0 specific index is clearly non-zero."""
    at0 = a[(a["lineage"] == r.root) & (a["step"] == 0)]
    return num(r.ratio_to_0) if len(at0) and at0["specific_lo"].iloc[0] > 0 else "-"


def analyse(run: data.Run, rng, selection: Path | None = None) -> dict:
    """Every washout table; "_index" holds the Index itself (not written)."""
    idx = Index(data.context_scores(run, selection), run.units, rng)
    spec, draws = specific_table(run, idx)
    hits = own_context_hits(run, idx)
    return {"_index": idx, "index": idx.table(), "specific": spec, "hits": hits,
            "identification": identification(idx, hits), "themes": themes(spec, draws, rng),
            "noise": twin_noise(run, idx), "rules": rules_table(run, spec, draws), "jargon": jargon_table(run)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--selection", type=Path, help="a context-score selection directory (default: the run's only one)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    run = data.Run(args.run_dir)
    res = analyse(run, np.random.default_rng(0), args.selection)
    write(args.out or Path("results") / run.name / "washout", {k: v for k, v in res.items() if k[0] != "_"},
          report_lines(run, res))


if __name__ == "__main__":
    with threadpool_limits(stats.THREADS):
        main()
