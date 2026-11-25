"""Validation of the estimators on simulated members with known truth, and the power of the planned pilot sizes.

    python -m evotaste.analysis.validate [--reps 200] [--out results/validation]

Members answer items with solve probabilities p[m, i] = logistic(theta_i + a_m + g_mi): item difficulty theta_i
~ N(0, 3.2^2), which reproduces the census's spread of base-model pass rates (a third of selection items never solved
in 4 samples, a third always), an overall level a_m and an interaction g_mi. Clones share part of both with their
family. Every estimate is compared with the truth computed from p itself (the quantity the estimator targets for
these members and items); intervals are the item bootstrap the commands use. The tests run the same studies small.
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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("results/validation"))
    args = ap.parse_args(argv)
    rng = np.random.default_rng(20261125)
    r = args.reps
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
    args.out.mkdir(parents=True, exist_ok=True)
    inter.to_csv(args.out / "interaction.csv", index=False)
    dist.to_csv(args.out / "distance.csv", index=False)
    clones.to_csv(args.out / "clones.csv", index=False)
    (args.out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    with threadpool_limits(stats.THREADS):
        main()
