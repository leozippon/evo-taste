"""Validation of the estimators on simulated members with known truth, and the power of the planned pilot sizes.

    python -m evotaste.analysis.validate [--reps 200] [--studies pilot,length,likelihood] [--out results/validation]

Members answer items with solve probabilities p[m, i] = logistic(theta_i + a_m + g_mi): item difficulty theta_i
~ N(0, 3.2^2), which reproduces the census's spread of base-model pass rates (a third of selection items never solved
in 4 samples, a third always), an overall level a_m and an interaction g_mi. Clones share part of both with their
family. Every estimate is compared with the truth computed from p itself (the quantity the estimator targets for
these members and items); intervals are the item bootstrap the commands use. The length study adds answer lengths
and the token limit (simulate_length), the likelihood study members as next-token distributions whose KL is known
exactly (simulate_cross); each runs reps / 5 replicates per case. The tests run the same studies small.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from scipy import stats as sps

from evotaste.analysis import stats
from evotaste.analysis.report import render

THETA_SD = 3.2


def simulate(rng, m: int, n: int, k: int, interaction: float = 0.0, level: float = 0.0, family=None,
             shared: float = 0.0, shared_level: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(p, samples): p is m x n, samples m x n x k Bernoulli draws. `interaction` and `level` are logit SDs of g and
    a; with `family` (a family index per member), a share `shared` of g's variance and `shared_level` of a's is common
    to the members of a family."""
    theta = rng.normal(0, THETA_SD, n)
    fam = np.arange(m) if family is None else np.asarray(family)
    nf = fam.max() + 1
    g = (np.sqrt(shared) * rng.normal(0, interaction, (nf, n))[fam]
         + np.sqrt(1 - shared) * rng.normal(0, interaction, (m, n)))
    a = np.sqrt(shared_level) * rng.normal(0, level, nf)[fam] + np.sqrt(1 - shared_level) * rng.normal(0, level, m)
    p = 1 / (1 + np.exp(-(theta[None, :] + a[:, None] + g)))
    return p, (rng.random((m, n, k)) < p[..., None]).astype(float)


def units(n: int, size: int = 150) -> np.ndarray:
    """Item units of the simulated evaluation (the bootstrap resamples within them)."""
    return np.arange(n) // size


def interaction_study(rng, reps: int, m: int, n: int, k: int, interaction: float = 0.0, level: float = 0.0,
                      b: int = 500) -> pd.DataFrame:
    """Per replicate and estimator (plain, adjusted): estimate, truth, and whether the 95% interval covers it."""
    rows = []
    for rep in range(reps):
        p, x = simulate(rng, m, n, k, interaction, level)
        y, v = stats.cells(x)
        w = stats.item_draws(units(n), b, rng)
        for est, classes in (("plain", stats.one_class(y)), ("adjusted", stats.difficulty_classes(y))):
            e = stats.interaction(y, v, classes, stats.ones(n))[0]
            truth = stats.interaction(p, np.zeros_like(p), classes, stats.ones(n))[0]
            lo, hi = stats.ci(stats.interaction(y, v, classes, w))
            rows.append({"rep": rep, "estimator": est, "estimate": e, "truth": truth, "covered": lo <= truth <= hi,
                         "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def distance_study(rng, reps: int, n: int, k: int, interaction: float = 0.0, b: int = 500) -> pd.DataFrame:
    """Two members (identical when interaction is 0, as two copies of one adapter): adjusted distance, truth,
    coverage. Twelve further members set the difficulty classes, as in a real evaluation."""
    rows = []
    for rep in range(reps):
        fam = np.r_[0, 0, np.arange(1, 13)] if interaction == 0 else None
        p, x = simulate(rng, 14, n, k, 0.5, 0.2, family=fam, shared=1.0, shared_level=1.0) if interaction == 0 \
            else simulate(rng, 14, n, k, interaction)
        y, v = stats.cells(x)
        classes = stats.difficulty_classes(y)
        e = stats.distances(y[:2], v[:2], classes, stats.ones(n))[0, 0, 1]
        truth = stats.distances(p[:2], np.zeros_like(p[:2]), classes, stats.ones(n))[0, 0, 1]
        lo, hi = stats.ci(stats.distances(y[:2], v[:2], classes, stats.item_draws(units(n), b, rng))[:, 0, 1])
        rows.append({"rep": rep, "estimate": e, "truth": truth, "covered": lo <= truth <= hi})
    return pd.DataFrame(rows)


def clone_study(rng, reps: int, parents: int, n: int, k: int, interaction: float, shared: float,
                shared_level: float, perms: int = 2000) -> pd.DataFrame:
    """Parents with two clones each: the mean adjusted distance of clone pairs against unrelated pairs, its
    permutation p-value over family labels, and the ICC of accuracy with its permutation p-value."""
    fam = np.repeat(np.arange(parents), 2)
    arm = np.zeros(len(fam), int)
    rows = []
    for rep in range(reps):
        p, x = simulate(rng, len(fam), n, k, interaction, 0.2, family=fam, shared=shared, shared_level=shared_level)
        y, v = stats.cells(x)
        d = stats.distances(y, v, stats.difficulty_classes(y), stats.ones(n))[0]
        truth = stats.distances(p, np.zeros_like(p), stats.difficulty_classes(y), stats.ones(n))[0]
        perm, _ = stats.relabellings(fam, arm, rng, perms)
        contrast = clone_contrast(d, fam)
        acc = y.mean(1)
        f0 = stats.icc(acc, fam, arm)
        fp = np.array([stats.icc(acc, f, arm)["F"] for f in perm])
        rows.append({"rep": rep, "clone": pair_means(d, fam)[0], "unrelated": pair_means(d, fam)[1],
                     "clone_truth": pair_means(truth, fam)[0], "unrelated_truth": pair_means(truth, fam)[1],
                     "p_distance": stats.perm_p(contrast, np.array([clone_contrast(d, f) for f in perm])),
                     "icc": f0["icc"], "p_icc": stats.perm_p(f0["F"], fp)})
    return pd.DataFrame(rows)


CAP = 4096  # tokens: a longer sample is cut off and scores 0
LENGTH_SD, ITEM_LENGTH_SD, SAMPLE_LENGTH_SD = 0.35, 0.7, 0.6  # log-length SDs: member, item, sample


def simulate_length(rng, m: int, n: int, k: int, case: str, level: float = 0.3, effect: float = 1.0):
    """Members that differ in answer length: a sample's log tokens are lambda_i + ell_m + eps (SDs ITEM_LENGTH_SD,
    LENGTH_SD, SAMPLE_LENGTH_SD around log 600, which gives pilot C's spread of member means, 440-1,700 tokens), and
    its success is a draw of p[m, i] = logistic(theta_i + a_m + h_mi). case "length": h = beta_i ell_m / LENGTH_SD
    (beta_i ~ N(0, effect^2)) and a sample longer than CAP is cut off and scores 0, so all interaction comes from
    length; "none": h = g_mi ~ N(0, effect^2), no cap, so lengths differ but cause nothing; "half": both, each with
    half the variance, and the cap. Returns the true expected reward q = p P(not cut off), and reward, tokens and
    truncated samples (m x n x k); members alternate between two rules."""
    theta = rng.normal(0, THETA_SD, n)
    ell = rng.normal(0, LENGTH_SD, m)
    lam = rng.normal(np.log(600), ITEM_LENGTH_SD, n)
    a = rng.normal(0, level, m)
    share = {"length": 1.0, "none": 0.0, "half": 0.5}[case]
    h = (np.sqrt(share) * rng.normal(0, effect, n)[None, :] * ell[:, None] / LENGTH_SD
         + np.sqrt(1 - share) * rng.normal(0, effect, (m, n)))
    p = 1 / (1 + np.exp(-(theta[None, :] + a[:, None] + h)))
    cap = np.inf if case == "none" else CAP
    tokens = np.exp(lam[None, :, None] + ell[:, None, None] + rng.normal(0, SAMPLE_LENGTH_SD, (m, n, k)))
    truncated = (tokens > cap).astype(float)
    reward = ((rng.random((m, n, k)) < p[..., None]) & (truncated == 0)).astype(float)
    q = p * sps.norm.cdf((np.log(cap) - lam[None, :] - ell[:, None]) / SAMPLE_LENGTH_SD)
    return q, reward, np.minimum(tokens, cap), truncated


def length_study(rng, reps: int, m: int, n: int, k: int, case: str, b: int = 200, perms: int = 200) -> pd.DataFrame:
    """The length decomposition (evotaste.analysis.length) on simulated members: per replicate the adjusted
    interaction and its residual after rule, length and truncation, each with the truth computed from q with the same
    classes and covariates, the interval's coverage of the residual's truth, the length part and its permutation
    p-value, the similar-length band ratio and its p-value, and the distance-length partial correlation and p."""
    from evotaste.analysis import length

    rows = []
    rules = np.array(["r0", "r1"] * (m // 2))
    items = pd.DataFrame({"unit": "far", "domain": "math", "family": units(n)})
    for rep in range(reps):
        q, reward, tokens, truncated = simulate_length(rng, m, n, k, case)
        y, v = stats.cells(reward)
        lt, tr = np.log(tokens.mean((1, 2))), truncated.mean((1, 2))
        classes = stats.difficulty_classes(y)
        w = stats.item_draws(units(n), b, rng)
        perm = length.within_rule_permutations(rules, perms, rng)
        res = length.interaction_rows(y, v, classes, w, rules, lt, tr, perm)
        zs = length.models(rules, lt, tr)
        truth = {kk: stats.interaction(q, np.zeros_like(q), classes, stats.ones(n), z)[0] for kk, z in zs.items()}
        none, full = res[0], res[-1]
        bands = length.band_table(y, v, items, rules, lt, b, perms, rng).iloc[0]
        asc = length.association_table(y, v, items, rules, lt, tr, b, perms, rng)[0].set_index("statistic")
        rows.append({"rep": rep, "case": case, "interaction": none["var"], "interaction_truth": truth["none"],
                     "residual": full["var"], "residual_truth": truth[full["model"]],
                     "covered": full["var_lo"] <= truth[full["model"]] <= full["var_hi"],
                     "length_part": full["length_part"], "length_part_truth":
                         (truth[res[1]["model"]] - truth[full["model"]]) / truth["none"],
                     "p_length": full["p_perm"], "band_ratio": bands["ratio"], "p_band": bands["p_perm"],
                     "r_length": asc.loc["r_length", "value"], "p_association": asc.loc["r_length", "p_perm"]})
    return pd.DataFrame(rows)


def simulate_cross(rng, m: int, n: int, length: int = 20, vocab: int = 8, spread: float = 0.5, family=None,
                   shared: float = 0.0) -> tuple[pd.DataFrame, np.ndarray]:
    """Members as next-token distributions, for the likelihood measures: at each of `length` positions of item i the
    base model's logits are N(0, 1) and member m's add a deviation of SD `spread` (with `family`, a family index per
    member, a share `shared` of its variance common to the family). Each member and the base model answer every item
    once (tokens drawn position by position) and every one scores every answer exactly. Returns a cross-likelihood
    table in the engine's columns (plus family and unit) with subjects base, m0, m1, ..., and the true KL(a || b)
    per token between subjects (base first)."""
    fam = np.arange(m) if family is None else np.asarray(family)
    base = rng.normal(0, 1, (n, length, vocab))
    dev = (np.sqrt(shared) * rng.normal(0, spread, (fam.max() + 1, n, length, vocab))[fam]
           + np.sqrt(1 - shared) * rng.normal(0, spread, (m, n, length, vocab)))
    logits = np.concatenate([base[None], base[None] + dev])  # subject x item x position x token
    logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    p = np.exp(logp)
    kl = (np.einsum("aipv,aipv->a", p, logp)[:, None] - np.einsum("aipv,bipv->ab", p, logp)) / (n * length)
    cum = p.cumsum(-1)
    tokens = (rng.random((m + 1, n, length, 1)) > cum).sum(-1).clip(max=vocab - 1)  # author x item x position
    names = ["base"] + [f"m{j}" for j in range(m)]
    rows = []
    for a in range(m + 1):
        picked = np.take_along_axis(logp, tokens[a][None, ..., None], -1)[..., 0]  # scorer x item x position
        lp = picked.sum(-1)
        for s in range(m + 1):
            for i in range(n):
                rows.append({"scorer": names[s], "author": names[a], "step": 0, "uid": f"u{i:04d}",
                             "domain": ("math", "logic")[i % 2], "tokens": length, "logprob_sum": lp[s, i],
                             "family": f"f{i // 25}", "unit": "far"})
    return pd.DataFrame(rows), kl


def pair_means(d: np.ndarray, fam: np.ndarray) -> tuple[float, float]:
    same = (fam[:, None] == fam[None, :]) & ~np.eye(len(fam), dtype=bool)
    other = fam[:, None] != fam[None, :]
    return float(d[same].mean()), float(d[other].mean())


def clone_contrast(d: np.ndarray, fam: np.ndarray) -> float:
    """Mean distance of unrelated pairs minus that of clone pairs."""
    clone, unrelated = pair_means(d, fam)
    return unrelated - clone


def icc_power(families: int, rho: float, alpha: float = 0.05) -> float:
    """Power of the one-sided F test of ICC > 0 with families of two (normal theory): MSB / MSW is
    (1 + rho) / (1 - rho) times F(f - 1, f)."""
    crit = sps.f.ppf(1 - alpha, families - 1, families)
    return float(sps.f.sf(crit * (1 - rho) / (1 + rho), families - 1, families))


def detectable_icc(families: int, power: float = 0.8) -> float:
    lo, hi = 0.0, 0.999
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if icc_power(families, mid) < power else (lo, mid)
    return hi


def detectable_paired(n: int, power: float = 0.8, alpha: float = 0.05) -> float:
    """Smallest mean paired difference, in SDs of the difference, a two-sided paired t test detects (normal
    approximation of the sign-flip test)."""
    return float((sps.t.ppf(1 - alpha / 2, n - 1) + sps.t.ppf(power, n - 1)) / np.sqrt(n))


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by)
    return pd.DataFrame({"reps": g.size(), "mean_estimate": g["estimate"].mean(), "mean_truth": g["truth"].mean(),
                         "bias": g.apply(lambda x: (x["estimate"] - x["truth"]).mean(), include_groups=False),
                         "bias_se": g.apply(lambda x: (x["estimate"] - x["truth"]).std(ddof=1) / np.sqrt(len(x)),
                                            include_groups=False),
                         "sd_estimate": g["estimate"].std(ddof=1), "coverage": g["covered"].mean()}).reset_index()


def length_block(rng, reps: int, out: Path) -> list[str]:
    """The length decomposition at pilot C's size (28 members, 1,840 items, two samples) in three cases."""
    df = pd.concat([length_study(rng, reps, 28, 1840, 2, case) for case in ("length", "half", "none")])
    df.to_csv(out / "length.csv", index=False)
    g = df.groupby("case", sort=False)
    lines = ["length decomposition on simulated members (28 x 1840 x 2; members' lengths spread as in pilot C; "
             "'length': all interaction through length and the token limit, 'half': half of it, 'none': lengths "
             "differ but cause nothing); adjusted interaction in pp^2; residual after rule, length and truncation"]
    lines += render(["case", "reps", "interaction (truth)", "residual (truth)", "bias (se)", "coverage",
                     "length part (truth)", "reject at 5%", "band ratio", "reject", "r length", "reject"],
                    [[c, len(x), f"{1e4 * x['interaction'].mean():.1f} ({1e4 * x['interaction_truth'].mean():.1f})",
                      f"{1e4 * x['residual'].mean():.2f} ({1e4 * x['residual_truth'].mean():.2f})",
                      f"{1e4 * (x['residual'] - x['residual_truth']).mean():+.2f} "
                      f"({1e4 * (x['residual'] - x['residual_truth']).std() / np.sqrt(len(x)):.2f})",
                      f"{x['covered'].mean():.2f}",
                      f"{x['length_part'].mean():.3f} ({x['length_part_truth'].mean():.3f})",
                      f"{(x['p_length'] <= 0.05).mean():.2f}", f"{x['band_ratio'].mean():.2f}",
                      f"{(x['p_band'] <= 0.05).mean():.2f}", f"{x['r_length'].mean():.2f}",
                      f"{(x['p_association'] <= 0.05).mean():.2f}"] for c, x in g], 1)
    return lines


def likelihood_block(rng, reps: int, out: Path, b: int = 200) -> list[str]:
    """Margins against the exact KL, identification of identical and of different members, and the clone test by
    likelihood distance at pilot C's clone design (seven parents with two clones each)."""
    from evotaste.analysis import distinct, heritability

    rows = []
    for spread in (0.0, 0.1, 0.4):
        for rep in range(reps):
            fam = np.zeros(6, int) if spread == 0 else None
            df, kl = simulate_cross(rng, 6, 80, spread=0.4 if spread == 0 else spread, family=fam,
                                    shared=1.0 if spread == 0 else 0.0)
            c = distinct.Cross(df)
            w = stats.item_draws(np.zeros(80), b, rng)
            mp, md = distinct.margins(c, stats.ones(80))[..., 0], distinct.margins(c, w)
            off = ~np.eye(7, dtype=bool)
            off[0, :] = off[:, 0] = False  # member pairs only (identical members still differ from the base model)
            lo, hi = np.quantile(md, 0.025, axis=2), np.quantile(md, 0.975, axis=2)
            ident = distinct.identification(c, w)[0].set_index("scope").loc["all"]
            rows.append({"case": "identical members" if spread == 0 else f"deviation SD {spread}", "rep": rep,
                         "margin": mp[off].mean(), "truth": kl[off].mean(), "error": (mp - kl)[off].mean(),
                         "covered": ((lo <= kl) & (kl <= hi))[off].mean(), "identified": ident["rate"],
                         "chance": ident["chance"]})
    marg = pd.DataFrame(rows)
    fam = np.repeat(np.arange(7), 2)
    arm = np.zeros(14, int)
    crows = []
    for shared in (0.0, 0.5):
        for rep in range(reps):
            df, kl = simulate_cross(rng, 14, 80, length=10, spread=0.3, family=fam, shared=shared)
            mp = distinct.margins(distinct.Cross(df), stats.ones(80))[..., 0][1:, 1:]
            d, t = (mp + mp.T) / 2, (kl[1:, 1:] + kl[1:, 1:].T) / 2
            clone, unrelated = heritability.means(d, fam, arm)
            perm, _ = stats.relabellings(fam, arm, rng, 1000)
            null = np.array([np.subtract(*heritability.means(d, q, arm)[::-1]) for q in perm])
            crows.append({"shared": shared, "rep": rep, "clone": clone, "unrelated": unrelated,
                          "clone_truth": heritability.means(t, fam, arm)[0],
                          "unrelated_truth": heritability.means(t, fam, arm)[1],
                          "p": stats.perm_p(unrelated - clone, null)})
    clones = pd.DataFrame(crows)
    marg.to_csv(out / "likelihood.csv", index=False)
    clones.to_csv(out / "likelihood_clones.csv", index=False)
    lines = ["likelihood measures on simulated members (next-token distributions: base logits N(0,1) per position, a "
             "member's deviation added; 80 items of 20 tokens, 6 members; exact KL from the distributions)"]
    lines += render(["case", "reps", "margin (KL truth)", "bias (se)", "coverage", "identified (chance)"],
                    [[c, len(x), f"{x['margin'].mean():.4f} ({x['truth'].mean():.4f})",
                      f"{x['error'].mean():+.5f} ({x['error'].std() / np.sqrt(len(x)):.5f})", f"{x['covered'].mean():.2f}",
                      f"{x['identified'].mean():.3f} ({x['chance'].mean():.3f})"]
                     for c, x in marg.groupby("case", sort=False)], 1)
    lines += ["", "clone test by likelihood distance, seven parents x two clones, 80 items of 10 tokens (shared = share "
                  "of a clone's deviation common to its family): mean distances (truth) and rejection at 5%"]
    lines += render(["shared", "reps", "clone (truth)", "unrelated (truth)", "reject"],
                    [[s, len(x), f"{x['clone'].mean():.4f} ({x['clone_truth'].mean():.4f})",
                      f"{x['unrelated'].mean():.4f} ({x['unrelated_truth'].mean():.4f})", f"{(x['p'] <= 0.05).mean():.2f}"]
                     for s, x in clones.groupby("shared")], 1)
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--studies", default="pilot,length,likelihood",
                    help="comma-separated: pilot (interaction, distance, clones, power; report.txt), length "
                         "(length.txt), likelihood (likelihood.txt)")
    ap.add_argument("--out", type=Path, default=Path("results/validation"))
    args = ap.parse_args(argv)
    studies = args.studies.split(",")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, block, seed in (("length", length_block, 20261126), ("likelihood", likelihood_block, 20261127)):
        if name in studies:
            lines = block(np.random.default_rng(seed), max(args.reps // 5, 10), args.out)
            (args.out / f"{name}.txt").write_text("\n".join(lines) + "\n")
            print("\n".join(lines))
    if "pilot" in studies:
        pilot_block(np.random.default_rng(20261125), args.reps, args.out)


def pilot_block(rng, r: int, out: Path) -> None:
    """The WP-A studies: interaction, distance, clone pairs, and the power of the planned sizes (report.txt)."""
    lines = ["validation on simulated members (truth computed from the simulated solve probabilities); variances in pp^2"]
    inter = []
    for m, n, k in ((14, 2000, 2), (28, 3000, 2), (28, 4000, 2)):
        for case, (gi, lv) in {"null": (0.0, 0.0), "level only": (0.0, 0.3), "interaction 0.25": (0.25, 0.0),
                               "interaction 0.5 + level": (0.5, 0.3)}.items():
            inter.append(interaction_study(rng, r if m == 14 else max(r // 2, 20), m, n, k, gi, lv)
                         .assign(m=m, n=n, k=k, case=case))
    inter = summarise(pd.concat(inter), ["m", "n", "k", "case", "estimator"])
    lines += ["", "interaction: estimate against truth, its spread and the coverage of the 95% interval"]
    lines += render(["m", "n", "k", "case", "estimator", "truth", "bias (se)", "sd", "coverage"],
                    [[x.m, x.n, x.k, x.case, x.estimator, f"{1e4 * x.mean_truth:.2f}",
                      f"{1e4 * x.bias:+.2f} ({1e4 * x.bias_se:.2f})", f"{1e4 * x.sd_estimate:.2f}", f"{x.coverage:.2f}"]
                     for x in inter.itertuples()], 5)
    dist = pd.concat([distance_study(rng, r, 3000, 2, 0.0).assign(case="copies of one adapter"),
                      distance_study(rng, r, 3000, 2, 0.5).assign(case="independent, interaction 0.5")])
    dist = summarise(dist, ["case"])
    lines += ["", "distance of two members (3000 items, 2 samples)"]
    lines += render(["case", "truth", "bias (se)", "sd", "coverage"],
                    [[x.case, f"{1e4 * x.mean_truth:.2f}", f"{1e4 * x.bias:+.2f} ({1e4 * x.bias_se:.2f})",
                      f"{1e4 * x.sd_estimate:.2f}", f"{x.coverage:.2f}"] for x in dist.itertuples()])
    clones = []
    for parents in (3, 6, 14):
        for shared in (0.0, 0.5):
            clones.append(clone_study(rng, max(r // 4, 20), parents, 3000, 2, 0.5, shared, shared)
                          .assign(parents=parents, shared=shared))
    clones = pd.concat(clones)
    cs = clones.groupby(["parents", "shared"]).agg(
        clone=("clone", "mean"), unrelated=("unrelated", "mean"), clone_truth=("clone_truth", "mean"),
        unrelated_truth=("unrelated_truth", "mean"), reject_distance=("p_distance", lambda p: (p <= 0.05).mean()),
        icc=("icc", "mean"), reject_icc=("p_icc", lambda p: (p <= 0.05).mean())).reset_index()
    lines += ["", "clone pairs (interaction logit SD 0.5, level SD 0.2; shared = share of both common to a family): "
                  "distances in pp^2, rejection rates of the permutation tests at 5%"]
    lines += render(["parents", "shared", "clone (truth)", "unrelated (truth)", "reject dist", "ICC acc", "reject ICC"],
                    [[x.parents, x.shared, f"{1e4 * x.clone:.1f} ({1e4 * x.clone_truth:.1f})",
                      f"{1e4 * x.unrelated:.1f} ({1e4 * x.unrelated_truth:.1f})", f"{x.reject_distance:.2f}",
                      f"{x.icc:.2f}", f"{x.reject_icc:.2f}"] for x in cs.itertuples()], 2)
    z = sps.norm.ppf(0.95) + sps.norm.ppf(0.8)
    null = inter[inter["case"] == "null"]
    lines += ["", "detectable interaction SD (one-sided 5%, power 80%, from the null spread of the estimator): "
              + ", ".join(f"{x.estimator} m {x.m} n {x.n}: {100 * np.sqrt(z * x.sd_estimate):.1f} pp"
                          for x in null.itertuples())]
    k_needed = {n: next(k for k in range(n + 1) if sps.binom.sf(k - 1, n, 1 / 11) < 0.05) for n in (5, 11, 22)}
    lines += ["identification against chance 1/11 (exact binomial, p < 0.05): " + ", ".join(
        f"{k} of {n} members" for n, k in k_needed.items())]
    lines += ["power (80%, 5% level): ICC of a scalar trait with parents x 2 clones, normal theory: "
              + ", ".join(f"{f} parents {detectable_icc(f):.2f}" for f in (3, 4, 6, 8, 14)),
              "paired stance - tactic difference by theme, in SDs of the difference: "
              + ", ".join(f"{t} themes {detectable_paired(t):.2f}" for t in (5, 10))
              + "; the exact sign-flip test over 5 themes cannot reach p < 0.05 two-sided (minimum 2/32)"]
    inter.to_csv(out / "interaction.csv", index=False)
    dist.to_csv(out / "distance.csv", index=False)
    clones.to_csv(out / "clones.csv", index=False)
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    with threadpool_limits(stats.THREADS):
        main()
