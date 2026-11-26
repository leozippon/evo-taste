"""The pilot estimators on simulated members with known truth (evotaste.analysis.validate, small): zero under the null,
recovery of planted effects, coverage of the item bootstrap; and their exact algebra on tiny cases."""

import itertools

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from evotaste.analysis import stats, validate


@pytest.fixture(autouse=True)
def few_threads():
    with threadpool_limits(4):
        yield


def summary(df, by):
    return {k: r for k, r in validate.summarise(df, by).set_index(by).iterrows()}


def test_estimator_is_the_cross_half_product_averaged_over_every_split():
    """With two samples per cell the estimator equals the cross-half estimate <C x, C z> / df averaged over all
    2^(cells) ways to call either sample the first half: the U-statistic, hence unbiased."""
    rng = np.random.default_rng(0)
    x = (rng.random((3, 3, 2)) < 0.5).astype(float)
    y, v = stats.cells(x)
    classes = np.zeros(3, int)
    ours = stats.interaction(y, v, classes, stats.ones(3))[0]

    def centred(a):
        return a - a.mean(0) - a.mean(1, keepdims=True) + a.mean()

    halves = []
    for flips in itertools.product((0, 1), repeat=9):
        f = np.array(flips).reshape(3, 3)
        a, b = np.where(f, x[..., 1], x[..., 0]), np.where(f, x[..., 0], x[..., 1])
        halves.append((centred(a) * centred(b)).sum() / 4)  # (M - 1)(N - 1) = 4
    assert ours == pytest.approx(np.mean(halves), abs=1e-12)


def test_interaction_is_zero_under_the_null_and_its_interval_covers():
    s = summary(validate.interaction_study(np.random.default_rng(1), 60, 14, 2000, 2, b=300), ["estimator"])
    for est in ("plain", "adjusted"):
        assert abs(s[est]["bias"]) < 3 * s[est]["bias_se"], est
        assert s[est]["coverage"] >= 0.85, est


def test_differences_in_level_inflate_the_plain_interaction_but_not_the_adjusted_one():
    """Members that differ only in overall level (logit SD 0.3, accuracy SD about 3 points) solve the same problems,
    yet on the probability scale they show a plain interaction; the difficulty-adjusted estimator removes it."""
    df = validate.interaction_study(np.random.default_rng(2), 30, 28, 3000, 2, 0.0, 0.3, b=100)
    s = summary(df, ["estimator"])
    assert s["plain"]["mean_estimate"] > 4 * s["plain"]["bias_se"]  # a spurious interaction SD of about 2.5 points
    assert abs(s["adjusted"]["mean_estimate"]) < 0.15 * s["plain"]["mean_estimate"]


def test_planted_interaction_is_recovered_within_its_interval():
    df = validate.interaction_study(np.random.default_rng(3), 30, 28, 3000, 2, 0.5, 0.3, b=300)
    s = summary(df, ["estimator"])
    for est in ("plain", "adjusted"):
        assert abs(s[est]["bias"]) < 3 * s[est]["bias_se"] + 0.03 * s[est]["mean_truth"], est
        assert s[est]["coverage"] >= 0.85, est
    assert s["adjusted"]["mean_truth"] > 10 * s["adjusted"]["sd_estimate"] / np.sqrt(30)


def test_distance_of_two_copies_of_one_adapter_is_zero():
    df = validate.distance_study(np.random.default_rng(4), 60, 3000, 2, 0.0, b=300).assign(c=0)
    s = summary(df, ["c"])[0]
    assert s["mean_truth"] == 0 and abs(s["bias"]) < 3 * s["bias_se"] and s["coverage"] >= 0.85


def test_clone_pairs_are_closer_only_when_they_share_their_parent_s_profile():
    rng = np.random.default_rng(5)
    null = validate.clone_study(rng, 20, 6, 3000, 2, 0.5, 0.0, 0.0, perms=500)
    assert (null["p_distance"] <= 0.05).mean() <= 0.2 and (null["p_icc"] <= 0.05).mean() <= 0.2
    assert abs((null["clone"] - null["unrelated"]).mean()) < 3 * (null["clone"] - null["unrelated"]).std() / np.sqrt(20)
    shared = validate.clone_study(rng, 20, 6, 3000, 2, 0.5, 0.5, 0.5, perms=500)
    assert (shared["p_distance"] <= 0.05).mean() >= 0.8
    ratio, truth = (shared["clone"] / shared["unrelated"]).mean(), (shared["clone_truth"] / shared["unrelated_truth"]).mean()
    assert abs(ratio - truth) < 0.1 and truth < 0.65


def test_profile_and_trait_spread_are_zero_under_the_null():
    rng = np.random.default_rng(6)
    prof, spread = [], []
    groups = np.repeat(np.arange(3), 700)
    for _ in range(80):
        p, x = validate.simulate(rng, 14, 2100, 2)
        y, v = stats.cells(x)
        prof.append(stats.profile(y, v, groups, stats.ones(2100))[0])
        t, n = stats.member_means(y, v, stats.ones(2100))
        spread.append(stats.between(t, n)[0])
    for e in (np.array(prof), np.array(spread)):
        assert abs(e.mean()) < 3 * e.std(ddof=1) / np.sqrt(len(e))


def test_cells_handle_missing_and_single_values():
    x = np.array([[[1.0, 0.0], [2.0, np.nan], [np.nan, np.nan]], [[0.0, 0.0], [1.0, 3.0], [4.0, 4.0]]])
    y, v = stats.cells(x)
    assert y[0].tolist()[:2] == [0.5, 2.0] and np.isnan(y[0, 2]) and v[0, 0] == pytest.approx(0.25)
    assert v[0, 1] == pytest.approx(0.5)  # one value: the member's mean within-cell variance (0.5) over one sample
    assert v[1, 1] == pytest.approx(1.0) and v[1, 2] == 0.0
    with pytest.raises(SystemExit, match="two samples"):
        stats.need_samples(x[..., :1], "divergence")


def test_icc_and_relabelling():
    fam = np.repeat(np.arange(5), 2)
    arm = np.zeros(10, int)
    assert stats.icc(np.repeat([0.1, 0.4, 0.2, 0.9, 0.5], 2) + [0, 1e-3] * 5, fam, arm)["icc"] > 0.99
    rng = np.random.default_rng(7)
    perm, distinct = stats.relabellings(fam, arm, rng, 200)
    assert distinct == 945 and perm.shape == (200, 10)  # 10! / (2^5 5!) pairings
    assert all(sorted(np.unique(p, return_counts=True)[1]) == [2] * 5 for p in perm)
    y = rng.normal(size=10)
    f = [stats.icc(y, p, arm)["F"] for p in perm[:50]]
    t = stats.between_ss(y, perm[:50])
    assert np.all(np.diff(np.array(f)[np.argsort(t)]) >= -1e-9)  # F increases with the between sum


def test_difficulty_classes_keep_ties_together_and_drop_constant_items():
    y = np.array([[0, 0.5, 0.5, 1, 0.5, 0.25], [0, 0.5, 0.0, 1, 1, 0.25]])
    c = stats.difficulty_classes(y, n=2)
    assert c[0] == -1 and c[3] == -1 and c[1] == c[4] and len(set(c[c >= 0])) == 2


def test_harvest_simulation_retains_a_real_lead_and_finds_none_in_the_null():
    """Members that differ in level keep part of the select-half lead on the test half; identical members keep
    none. Twenty units of 100 items in two domains, 14 members, two samples."""
    import pandas as pd

    from evotaste.analysis import harvest

    items = pd.DataFrame({"family": [f"u{i // 100}" for i in range(2000)], "unit": "far",
                          "domain": ["math" if i < 1000 else "logic" for i in range(2000)]})
    rng = np.random.default_rng(8)
    out = {}
    for case, level in (("null", 0.0), ("level", 0.3)):
        _, x = validate.simulate(rng, 14, 2000, 2, 0.0, level)
        y, _ = stats.cells(x)
        sim = harvest.simulate(y, items, {"all": list(range(14))}, 40, rng)
        out[case] = harvest.summarise(sim).set_index(["mode", "statistic"])
    for mode in ("units", "items"):
        null, real = out["null"].loc[mode], out["level"].loc[mode]
        assert null.loc["test_lead_top1", "lo"] < 0 < null.loc["test_lead_top1", "hi"]
        assert real.loc["test_lead_top1", "lo"] > 0 and real.loc["retained_top1", "value"] > 0.5
        assert real.loc["best_of_14", "value"] > real.loc["best_of_2", "value"] > 0
