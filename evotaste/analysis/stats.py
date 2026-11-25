"""Estimators of the pilot analysis and their bootstrap (docs/dev/ANALYSIS_NOTES.md, "Estimators").

Data are cells: Y[m, i], member m's mean over its samples of item i, and V[m, i], an unbiased estimate of the
sampling variance of that mean (s^2 / n with s^2 the sample variance, ddof 1; for K Bernoulli draws Y(1 - Y)/(K - 1)).
Samples of different cells are independent given the members. Every variance, distance and interaction here is a
quadratic form in Y; its plug-in value carries the noise sum_cells (diagonal weight of the form) Var(Y) on top of
the truth, and subtracting the same sum with V in place of Var(Y) makes it exactly unbiased. For each cell this is the
U-statistic of its samples, i.e. the cross-half product averaged over every split of the samples into two halves,
hence the unique minimum-variance unbiased estimator given the cell counts. It needs at least two samples per cell.

Every estimator takes item weights W (draws x items): ones for the estimate itself, multinomial counts for bootstrap
draws, and returns one value per row of W. In a draw, an item drawn twice is two items whose samples are the same
draws: the noise terms count it with weight w^2 where a mean over items is squared, so that every bootstrap
replicate is itself unbiased for its resampled items. (Treating the copies as independent biases the replicates
downward by about (number of item classes) x (mean cell noise variance) / items, half a bootstrap SD for the
adjusted interaction at the pilot sizes, and makes the intervals undercover.)
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats

THREADS = 8  # BLAS threads of every command: the host's CPUs are shared with running experiments
B = 2000  # bootstrap draws
PERMUTATIONS = 5000
CLASSES = 10  # difficulty classes of the adjusted estimators


def ones(n: int) -> np.ndarray:
    return np.ones((1, n))


def item_draws(groups, b: int, rng) -> np.ndarray:
    """b x N resampling counts, items drawn with replacement within each group (the unit), so every draw keeps the
    composition of the evaluation."""
    groups = np.asarray(groups)
    w = np.zeros((b, len(groups)))
    for g in dict.fromkeys(groups):
        idx = np.flatnonzero(groups == g)
        w[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=b)
    return w


def ci(draws) -> tuple[float, float]:
    """95% percentile interval of the finite draws."""
    draws = np.asarray(draws, float)
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return math.nan, math.nan
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def sd(var) -> float:
    """An SD from a variance estimate that may be negative (no effect beyond noise): 0 then."""
    return float(np.sqrt(max(var, 0.0))) if np.isfinite(var) else math.nan


def cells(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(Y, V) from samples x[m, i, k] (NaN where a response has no value). A cell with one value takes, as its
    variance, the member's mean within-cell variance over its cells with two or more; a cell without values is NaN."""
    n = np.isfinite(x).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.nanmean(np.where(n[..., None] > 0, x, 0.0), -1) * np.where(n > 0, 1, np.nan)
        s2 = np.where(n >= 2, np.nansum((x - y[..., None]) ** 2, -1) / (n - 1), np.nan)
    if (n == 1).any():
        pooled = np.nanmean(s2, axis=1, keepdims=True)
        s2 = np.where(n == 1, np.broadcast_to(pooled, s2.shape), s2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return y, np.where(n > 0, s2 / n, np.nan)


def need_samples(x: np.ndarray, what: str) -> None:
    if x.shape[2] < 2:
        raise SystemExit(f"{what} needs at least two samples per item (an evaluation with --samples 2 or more); "
                         f"this one has {x.shape[2]}")


def difficulty_classes(y: np.ndarray, within=None, n: int = CLASSES) -> np.ndarray:
    """Item classes by the pooled solve rate over all members, y.mean(0): -1 for items that no sample of any member
    solved or every sample solved; the others split into n classes of nearly equal size along the pooled rate (items
    of equal rate together), separately within each group of `within`. Under identical members the class of an item
    depends on its data only through its total, which leaves the estimators below exactly unbiased."""
    pooled = np.nanmean(y, 0)
    out = np.full(len(pooled), -1)
    groups = np.zeros(len(pooled), int) if within is None else np.unique(np.asarray(within), return_inverse=True)[1]
    for g in np.unique(groups):
        idx = np.flatnonzero((groups == g) & (pooled > 0) & (pooled < 1))
        if not len(idx):
            continue
        levels, inv, counts = np.unique(pooled[idx], return_inverse=True, return_counts=True)
        before = np.concatenate([[0], np.cumsum(counts)[:-1]])
        out[idx] = g * n + np.minimum((before[inv] * n) // len(idx), n - 1)
    return out


def one_class(y: np.ndarray, nonconstant: bool = False) -> np.ndarray:
    """All items in one class, or only the items that are neither never nor always solved."""
    pooled = np.nanmean(y, 0)
    return np.where((pooled > 0) & (pooled < 1), 0, -1) if nonconstant else np.zeros(len(pooled), int)


def interaction(y: np.ndarray, v: np.ndarray, classes: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Member-by-item interaction variance within item classes: the mean square of
    Y[m,i] - Y[m,class] - Y[.,i] + Y[.,class] over the cells of items with class >= 0, on (M - 1)(N_c - 1) degrees of
    freedom per class, minus the noise it contains. One class of all items is the plain two-way interaction; difficulty
    classes also remove each member's effect per difficulty class (see ANALYSIS_NOTES.md)."""
    m = y.shape[0]
    d = y - y.mean(0)  # removes item effects
    s, vs = (d ** 2).sum(0), v.sum(0)
    num, den = np.zeros(len(w)), np.zeros(len(w))
    for c in np.unique(classes[classes >= 0]):
        idx = classes == c
        wc = w[:, idx]
        n = wc.sum(1)
        mean = d[:, idx] @ wc.T / np.maximum(n, 1)  # member x draw: the member's effect in the class
        ok = n >= 2
        noise = (1 - 1 / m) * (wc @ vs[idx] - (wc ** 2) @ vs[idx] / np.maximum(n, 1))
        num += np.where(ok, wc @ s[idx] - n * (mean ** 2).sum(0) - noise, 0)
        den += np.where(ok, (m - 1) * (n - 1), 0)
    return num / den


def group_means(y: np.ndarray, v: np.ndarray, groups, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    """Per group (e.g. domain): every member's mean and the noise variance of that mean, arrays group x member x draw.
    NaN cells (no value) are left out of their member's mean."""
    groups = np.asarray(groups)
    names = list(dict.fromkeys(groups))
    a, av = [], []
    for g in names:
        idx = groups == g
        ok = np.isfinite(y[:, idx])
        yc, vc = np.where(ok, y[:, idx], 0.0), np.where(ok, v[:, idx], 0.0)
        n = ok.astype(float) @ w[:, idx].T  # member x draw
        with np.errstate(invalid="ignore", divide="ignore"):
            a.append(yc @ w[:, idx].T / n)
            av.append(vc @ (w[:, idx] ** 2).T / n ** 2)
    return np.array(a), np.array(av), names


def profile(y: np.ndarray, v: np.ndarray, groups, w: np.ndarray) -> np.ndarray:
    """Member-by-group interaction variance of group accuracies (each member's group profile relative to its own
    mean, its between-member variance beyond noise): two-way mean square on (M - 1)(G - 1) degrees of freedom."""
    a, av, _ = group_means(y, v, groups, w)
    g, m = a.shape[:2]
    r = a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean((0, 1), keepdims=True)
    return ((r ** 2).sum((0, 1)) - (1 - 1 / m) * (1 - 1 / g) * av.sum((0, 1))) / ((m - 1) * (g - 1))


def member_means(y: np.ndarray, v: np.ndarray, w: np.ndarray, units=None) -> tuple[np.ndarray, np.ndarray]:
    """Each member's mean over items, or macro mean over units (the engine's accuracy) when units are given, and the
    noise variance of that mean: arrays member x draw."""
    a, av, _ = group_means(y, v, np.zeros(y.shape[1], int) if units is None else units, w)
    return a.mean(0), av.sum(0) / len(a) ** 2


def between(t: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Between-member variance of member means beyond their noise: sample variance (ddof 1) minus the mean noise
    variance, per draw (t and noise: member x draw)."""
    return t.var(0, ddof=1) - noise.mean(0)


def distances(y: np.ndarray, v: np.ndarray, classes: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Functional distance of every pair of members: the mean square over items (classes >= 0) of the difference of
    their solve rates after removing its mean within each class, on N_c - 1 degrees of freedom per class, minus the
    noise it contains. Array draw x member x member."""
    b, m = len(w), y.shape[0]
    num, den = np.zeros((b, m, m)), np.zeros(b)
    for c in np.unique(classes[classes >= 0]):
        idx = classes == c
        yc, vc, wc = y[:, idx], v[:, idx], w[:, idx]
        n = wc.sum(1)
        gram = np.concatenate([(yc[None] * wc[j:j + 64, None, :]) @ yc.T for j in range(0, b, 64)])  # draw x m x m
        sq = np.einsum("bii->bi", gram)
        s1 = (yc @ wc.T).T  # draw x member
        vw, vw2 = (vc @ wc.T).T, (vc @ (wc ** 2).T).T
        ok = (n >= 2)[:, None, None]
        nn = np.maximum(n, 1)[:, None, None]
        pair = sq[:, :, None] + sq[:, None, :] - 2 * gram - (s1[:, :, None] - s1[:, None, :]) ** 2 / nn
        noise = vw[:, :, None] + vw[:, None, :] - (vw2[:, :, None] + vw2[:, None, :]) / nn
        num += np.where(ok, pair - noise, 0)
        den += np.where(ok[:, 0, 0], n - 1, 0)
    out = num / den[:, None, None]
    out[:, np.arange(m), np.arange(m)] = 0.0
    return out


# -- families: intraclass correlation and relabelling ---------------------------------------------------------------

def icc(values: np.ndarray, family: np.ndarray, arm: np.ndarray) -> dict:
    """One-way random-effects decomposition across families (members sharing a state, e.g. clones of one parent; a
    family lies within one arm), with each arm's mean as a fixed effect: between = (MSB - MSW) / n0 (may be negative),
    within = MSW, ICC = between / (between + within), F = MSB / MSW on (F - A, N - F) degrees of freedom. `values`
    is one value per member, or members x traits (then every result is an array over traits)."""
    values = np.asarray(values, float)
    y = values.reshape(len(values), -1)
    fam = np.unique(np.asarray(family), return_inverse=True)[1]
    arm_ = np.unique(np.asarray(arm), return_inverse=True)[1]
    nf, na, n = fam.max() + 1, arm_.max() + 1, len(y)
    df_b, df_w = nf - na, n - nf
    shape = values.shape[1:]
    if df_b < 1 or df_w < 1:
        nan = np.full(shape, math.nan) if shape else math.nan
        return {"between": nan, "within": nan, "icc": nan, "F": nan, "df_b": df_b, "df_w": df_w}
    one = np.zeros((nf, n))
    one[fam, np.arange(n)] = 1
    size = one.sum(1)
    mean = one @ y / size[:, None]
    f_arm = np.zeros(nf, int)
    f_arm[fam] = arm_
    a_one = np.zeros((na, n))
    a_one[arm_, np.arange(n)] = 1
    arm_mean = a_one @ y / a_one.sum(1)[:, None]
    msb = (size[:, None] * (mean - arm_mean[f_arm]) ** 2).sum(0) / df_b
    msw = ((y - mean[fam]) ** 2).sum(0) / df_w
    n0 = (n - float((size ** 2 / a_one.sum(1)[f_arm]).sum())) / df_b
    b_ = (msb - msw) / n0
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {"between": b_, "within": msw, "icc": np.where(b_ + msw > 0, b_ / (b_ + msw), np.nan),
               "F": np.where(msw > 0, msb / msw, np.inf)}
    out = {k: (v if shape else float(v[0])) for k, v in out.items()}
    return out | {"df_b": df_b, "df_w": df_w}


def drawn_members(family: np.ndarray, drawn: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For a draw of families (indices into the distinct families in order of appearance): the members of the drawn
    families, in order, their slot (a family drawn twice is two slots) and their original family index."""
    family = np.asarray(family)
    names = list(dict.fromkeys(family))
    idx = [np.flatnonzero(family == names[j]) for j in drawn]
    return (np.concatenate(idx), np.concatenate([np.full(len(i), s) for s, i in enumerate(idx)]),
            np.concatenate([np.full(len(i), j) for i, j in zip(idx, drawn)]))


def between_ss(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """sum over families of (family sum)^2 / (family size), for each row of labels (relabellings x members). With the
    family sizes and arms fixed, as under relabelling within arms, the F statistic of `icc` increases with it."""
    values, labels = np.asarray(values, float), np.atleast_2d(labels)
    codes = np.unique(labels, return_inverse=True)[1].reshape(labels.shape)
    f = codes.max() + 1
    idx = (codes + f * np.arange(len(codes))[:, None]).ravel()
    n = np.bincount(idx, minlength=f * len(codes)).reshape(len(codes), f)
    s = np.bincount(idx, weights=np.tile(values, len(codes)), minlength=f * len(codes)).reshape(len(codes), f)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s ** 2 / n, 0.0).sum(1)


def relabellings(family: np.ndarray, arm: np.ndarray, rng, n: int = PERMUTATIONS) -> tuple[np.ndarray, int]:
    """n random reassignments of members to the family slots within each arm (family sizes kept), as arrays of family
    labels, and the number of distinct partitions they are drawn from (the resolution of a permutation p-value)."""
    family, arm = np.asarray(family), np.asarray(arm)
    out = np.tile(family, (n, 1))
    distinct = 1
    for a in dict.fromkeys(arm):
        idx = np.flatnonzero(arm == a)
        out[:, idx] = rng.permuted(np.tile(family[idx], (n, 1)), axis=1)
        sizes = np.unique(family[idx], return_counts=True)[1]
        distinct *= math.factorial(len(idx)) // (math.prod(math.factorial(s) for s in sizes)
                                                 * math.prod(math.factorial(c) for c in np.unique(sizes, return_counts=True)[1]))
    return out, distinct


def perm_p(observed: float, permuted: np.ndarray) -> float:
    """One-sided permutation p-value, (1 + #{permuted >= observed}) / (1 + #permuted)."""
    permuted = permuted[np.isfinite(permuted)]
    return float((1 + np.sum(permuted >= observed - 1e-12)) / (1 + len(permuted)))


def family_draws(family: np.ndarray, arm: np.ndarray, b: int, rng) -> list[np.ndarray]:
    """b draws of families with replacement within each arm, each as an array of (original family index) per drawn
    family slot."""
    family, arm = np.asarray(family), np.asarray(arm)
    fams = list(dict.fromkeys(family))
    by_arm = {}
    for f in fams:
        by_arm.setdefault(arm[family == f][0], []).append(fams.index(f))
    return [np.concatenate([rng.choice(v, len(v)) for v in by_arm.values()]) for _ in range(b)]


# -- small-sample helpers ------------------------------------------------------------------------------------------

def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    lo = stats.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def sign_flip_p(d: np.ndarray) -> float:
    """Exact two-sided sign-flip p-value of the mean of paired differences."""
    d = np.asarray(d, float)
    flips = np.array(list(itertools.product((1, -1), repeat=len(d))))
    means = np.abs((flips * d).mean(1))
    return float(np.mean(means >= abs(d.mean()) - 1e-12))
