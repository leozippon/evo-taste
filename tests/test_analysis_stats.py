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


def test_residual_interaction_is_the_cross_half_product_of_the_projected_cells():
    """With member covariates the estimator equals <P x C, P z C> / df averaged over every split of the two samples:
    unbiased for the projection of the true solve probabilities, and the plain estimator when z is None."""
    rng = np.random.default_rng(10)
    x = (rng.random((4, 3, 2)) < 0.5).astype(float)
    y, v = stats.cells(x)
    z = np.array([[0.1], [0.7], [0.2], [1.5]])
    p = stats.member_projection(4, z)
    ours = stats.interaction(y, v, np.zeros(3, int), stats.ones(3), z)[0]
    halves = []
    for flips in itertools.product((0, 1), repeat=12):
        f = np.array(flips).reshape(4, 3)
        a, b = np.where(f, x[..., 1], x[..., 0]), np.where(f, x[..., 0], x[..., 1])
        ca, cb = p @ (a - a.mean(1, keepdims=True)), p @ (b - b.mean(1, keepdims=True))
        halves.append((ca * cb).sum() / (2 * 2))  # (M - 1 - k)(N - 1) = 2 x 2
    assert ours == pytest.approx(np.mean(halves), abs=1e-12)
    assert stats.interaction(y, v, np.zeros(3, int), stats.ones(3))[0] == pytest.approx(
        stats.interaction(y, v, np.zeros(3, int), stats.ones(3), None)[0])
    with pytest.raises(ValueError, match="collinear"):
        stats.member_projection(4, np.c_[z, 2 * z])


def test_length_decomposition_on_simulated_truth():
    """All interaction caused by length (and the token limit): the length part is nearly all of it, the residual is
    unbiased for its truth, similar-length members interact far less than random bands, and distance follows the
    length difference. None caused by length: the length part, the band contrast and the association vanish."""
    rng = np.random.default_rng(11)
    caused = validate.length_study(rng, 4, 28, 1500, 2, "length", b=60, perms=60)
    assert caused["length_part"].mean() > 0.85 and caused["length_part_truth"].mean() > 0.85
    assert abs((caused["residual"] - caused["residual_truth"]).mean()) < 0.15 * caused["interaction"].mean()
    assert (caused["p_length"] <= 0.05).all() and caused["band_ratio"].max() < 0.6 and caused["r_length"].min() > 0.7
    none = validate.length_study(rng, 6, 28, 1500, 2, "none", b=60, perms=60)
    assert abs(none["length_part"].mean()) < 0.03 and abs(none["length_part_truth"].mean()) < 0.02
    assert abs((none["residual"] - none["residual_truth"]).mean()) < 3 * (none["residual"] - none["residual_truth"]).std()
    assert 0.9 < none["band_ratio"].mean() < 1.1 and abs(none["r_length"].mean()) < 0.2
    half = validate.length_study(rng, 3, 28, 1500, 2, "half", b=60, perms=60)
    assert abs(half["length_part"].mean() - half["length_part_truth"].mean()) < 0.1 and 0.3 < half["length_part"].mean() < 0.7


def test_deviation_at_typical_length_recovers_a_planted_line():
    """An answer's deviation per token = 0.1 + 0.05 log2(tokens / typical) + noise: the intercept is 0.1, whatever the
    answers' lengths, here all longer than typical."""
    from types import SimpleNamespace

    from evotaste.analysis import deviation

    rng = np.random.default_rng(12)
    typical = rng.uniform(300, 900, 400)
    tokens = typical[None, :] * rng.uniform(1.2, 4.0, (3, 400))
    per = 0.1 + 0.05 * np.log2(tokens / typical) + rng.normal(0, 0.02, tokens.shape)
    t = SimpleNamespace(tokens=tokens, delta=per * tokens)
    a, s = deviation.at_typical(t, typical, stats.ones(400))
    assert np.allclose(a[:, 0], 0.1, atol=0.01) and np.allclose(s[:, 0], 0.05, atol=0.01)


def test_likelihood_margins_are_kl_and_identification_is_chance_for_identical_members():
    from evotaste.analysis import distinct

    rng = np.random.default_rng(13)
    errors = []
    for _ in range(6):
        df, kl = validate.simulate_cross(rng, 5, 60, spread=0.4)
        c = distinct.Cross(df)
        mp = distinct.margins(c, stats.ones(len(c.items)))[..., 0]
        errors.append((mp - kl)[~np.eye(6, dtype=bool)])
        assert np.allclose(np.diag(mp), 0)
    errors = np.concatenate(errors)
    assert abs(errors.mean()) < 3 * errors.std() / np.sqrt(len(errors)) + 1e-3
    df, kl = validate.simulate_cross(rng, 4, 50, family=np.zeros(4, int), shared=1.0)  # four copies of one member
    c = distinct.Cross(df)
    ident, hit = distinct.identification(c, stats.item_draws(np.zeros(50), 50, rng))
    assert np.allclose(hit, 0.25) and ident.loc[ident["scope"] == "all", "rate"].item() == pytest.approx(0.25)
    df, _ = validate.simulate_cross(rng, 4, 50, spread=0.5)  # different members: identified far above chance
    ident, _ = distinct.identification(distinct.Cross(df), stats.item_draws(np.zeros(50), 50, rng))
    assert ident.loc[ident["scope"] == "all", "lo"].item() > 0.5


def test_clone_pairs_by_likelihood_are_closer_only_when_they_share_a_deviation():
    from evotaste.analysis import distinct, heritability

    rng = np.random.default_rng(14)
    fam = np.repeat(np.arange(6), 2)
    arm = np.zeros(12, int)
    p = {}
    for shared in (0.0, 0.6):
        ps = []
        for _ in range(8):
            df, _ = validate.simulate_cross(rng, 12, 40, length=10, spread=0.4, family=fam, shared=shared)
            mp = distinct.margins(distinct.Cross(df), stats.ones(40))[..., 0][1:, 1:]
            d = (mp + mp.T) / 2
            clone, unrelated = heritability.means(d, fam, arm)
            perm, _ = stats.relabellings(fam, arm, rng, 300)
            ps.append(stats.perm_p(unrelated - clone, np.array([np.subtract(*heritability.means(d, q, arm)[::-1])
                                                                for q in perm])))
        p[shared] = np.array(ps)
    assert (p[0.0] <= 0.05).mean() <= 0.25 and (p[0.6] <= 0.05).mean() >= 0.75


def test_common_share_of_practice_movement():
    from evotaste.analysis import arms

    rng = np.random.default_rng(15)
    common = rng.normal(0, 0.3, 9)
    assert arms.common_share(np.tile(common, (3, 1))) == pytest.approx(1.0)
    shares = [arms.common_share(rng.normal(0, 0.3, (3, 9))) for _ in range(400)]
    assert abs(np.mean(shares) - 1 / 3) < 0.03


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
