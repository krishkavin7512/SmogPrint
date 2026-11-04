"""The from-scratch models, checked against synthetic truth and against scikit-learn and SciPy."""
import numpy as np
import pandas as pd
import pytest

from backend import evidence
from backend.cluster import GaussianMixture, KMeans, adjusted_rand, agglomerative, cophenetic_corr, cut_tree, dendrogram_segments, leaf_order, silhouette
from backend.fa import FactorAnalysis, bartlett, kmo, pairwise_corr, parallel_analysis, pca, principal_axis, varimax, varimax_score


def synthetic_factors(n=4000, seed=0, missing=0.0):
    """Two known factors behind six variables."""
    rng = np.random.default_rng(seed)
    W = np.array([[.9, 0], [.8, 0], [.7, 0], [0, .9], [0, .8], [0, .7]])
    psi = 1 - (W ** 2).sum(1)
    z = rng.standard_normal((n, 2))
    X = z @ W.T + rng.standard_normal((n, 6)) * np.sqrt(psi)
    if missing:
        X[rng.random(X.shape) < missing] = np.nan
    return X, W, psi, z


def test_em_recovers_known_loadings():
    X, W, psi, _ = synthetic_factors()
    m = FactorAnalysis(2, 80).fit(X)
    L, _ = varimax(m.W)
    assert np.allclose(np.sort(np.abs(L).max(1)), np.sort(W.max(1)), atol=0.05)
    assert np.allclose(m.psi, psi, atol=0.06)


def test_em_likelihood_never_falls_with_missing_values():
    X, *_ = synthetic_factors(missing=0.3)
    m = FactorAnalysis(2, 40).fit(X)
    assert np.all(np.diff(m.history) > -1e-6)


def test_em_recovers_loadings_through_gaps():
    X, W, *_ = synthetic_factors(n=8000, missing=0.35)
    L, _ = varimax(FactorAnalysis(2, 100).fit(X).W)
    assert np.allclose(np.sort(np.abs(L).max(1)), np.sort(W.max(1)), atol=0.07)


def test_matches_sklearn_on_complete_data():
    from sklearn.decomposition import FactorAnalysis as SK
    X, *_ = synthetic_factors()
    m = FactorAnalysis(2, 200).fit(X)
    s = SK(2, tol=1e-6, max_iter=2000).fit(X)
    assert m.loglik(X) == pytest.approx(s.score(X) * len(X), rel=1e-3)
    assert np.allclose(m.covariance(), s.get_covariance(), atol=0.02)


def test_rotation_changes_nothing_about_the_fit():
    X, *_ = synthetic_factors(missing=0.2)
    m = FactorAnalysis(2, 30).fit(X)
    before = m.loglik(X)
    Q, _ = np.linalg.qr(np.random.default_rng(1).standard_normal((2, 2)))
    assert m.rotate(Q).loglik(X) == pytest.approx(before, rel=1e-9)


def test_varimax_is_orthogonal_and_raises_its_score():
    X, *_ = synthetic_factors()
    m = FactorAnalysis(2, 40).fit(X)
    Q = np.array([[np.cos(.6), -np.sin(.6)], [np.sin(.6), np.cos(.6)]])
    twisted = m.W @ Q
    L, R = varimax(twisted)
    assert np.allclose(R.T @ R, np.eye(2), atol=1e-8)
    assert varimax_score(L) >= varimax_score(twisted)


def test_varimax_undoes_a_rotation_from_the_worst_start():
    """Start at 45 degrees, where PCA-style loadings sit and an SVD iteration crawls."""
    W = np.array([[.9, 0], [.8, 0], [.7, 0], [0, .9], [0, .8], [0, .7]])
    c = np.sqrt(.5)
    L, R = varimax(W @ np.array([[c, -c], [c, c]]))
    assert np.allclose(np.sort(np.abs(L), axis=1)[:, 1], W.max(1), atol=1e-6)
    assert np.allclose(np.sort(np.abs(L), axis=1)[:, 0], 0, atol=1e-6)


def test_varimax_reaches_scikit_learns_quality():
    from sklearn.decomposition import FactorAnalysis as SK
    X, *_ = synthetic_factors()
    mine = varimax(FactorAnalysis(2, 80).fit(X).W)[0]
    theirs = SK(2, rotation="varimax", tol=1e-6, max_iter=2000).fit(X).components_.T
    assert varimax_score(mine) == pytest.approx(varimax_score(theirs), rel=0.02)


def test_rotated_loadings_equal_loadings_times_rotation():
    X, *_ = synthetic_factors()
    W = FactorAnalysis(2, 40).fit(X).W
    L, R = varimax(W)
    assert np.allclose(W @ R, L, atol=1e-8)


def test_scores_recover_the_hidden_values():
    X, _, _, z = synthetic_factors()
    m = FactorAnalysis(2, 40).fit(X)
    m.W = varimax(m.W)[0]
    s = m.scores(X)
    assert max(abs(np.corrcoef(s[:, 0], z[:, 0])[0, 1]), abs(np.corrcoef(s[:, 1], z[:, 0])[0, 1])) > 0.85


def test_row_with_no_readings_scores_nan():
    X, *_ = synthetic_factors(n=300)
    X[0] = np.nan
    s = FactorAnalysis(2, 5).fit(X).scores(X)
    assert np.isnan(s[0]).all() and not np.isnan(s[1:]).any()


def test_parallel_analysis_keeps_the_real_factors():
    X, *_ = synthetic_factors()
    ev, _ = pca(pairwise_corr(X))
    assert (ev > parallel_analysis(X, 10)).sum() == 2


def test_adequacy_measures():
    X, *_ = synthetic_factors()
    R = pairwise_corr(X)
    assert kmo(R)[0] > 0.6
    chi2, dof = bartlett(R, len(X))
    assert dof == 15 and chi2 > 1000
    pure = pairwise_corr(np.random.default_rng(0).standard_normal((4000, 6)))
    assert bartlett(pure, 4000)[0] < 40


def test_principal_axis_agrees_with_em():
    X, *_ = synthetic_factors()
    L1, _ = varimax(FactorAnalysis(2, 60).fit(X).W)
    L2, _ = varimax(principal_axis(pairwise_corr(X), 2))
    assert np.allclose(np.sort(np.abs(L1).max(1)), np.sort(np.abs(L2).max(1)), atol=0.04)


def blobs(n=600, seed=0):
    rng = np.random.default_rng(seed)
    c = np.array([[0, 0], [6, 0], [0, 6]])
    y = rng.integers(0, 3, n)
    return c[y] + rng.standard_normal((n, 2)), y


def test_kmeans_matches_sklearn():
    from sklearn.cluster import KMeans as SK
    X, y = blobs()
    m = KMeans(3, seed=1).fit(X)
    s = SK(3, n_init=5, random_state=1).fit(X)
    assert m.inertia == pytest.approx(s.inertia_, rel=1e-3)
    assert adjusted_rand(m.predict(X), y) > 0.97


def test_gmm_matches_sklearn_and_climbs():
    from sklearn.mixture import GaussianMixture as SK
    X, y = blobs()
    m = GaussianMixture(3, seed=1).fit(X)
    s = SK(3, reg_covar=1e-4, random_state=1).fit(X)
    assert np.all(np.diff(m.history) > -1e-6)
    assert m.loglik(X) / len(X) == pytest.approx(s.score(X), rel=1e-3)
    assert np.allclose(m.proba(X).sum(1), 1)
    assert m.bic(X) == pytest.approx(s.bic(X), rel=1e-3)


def test_silhouette_and_rand_match_sklearn():
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    X, y = blobs(300)
    assert silhouette(X, y, sample=300) == pytest.approx(silhouette_score(X, y), rel=1e-6)
    other = (y + (np.arange(300) % 7 == 0)) % 3
    assert adjusted_rand(y, other) == pytest.approx(adjusted_rand_score(y, other), rel=1e-9)


@pytest.mark.parametrize("method", ["ward", "average", "complete", "single"])
def test_hierarchy_matches_scipy(method):
    from scipy.cluster.hierarchy import linkage
    X, _ = blobs(40)
    assert np.allclose(np.sort(agglomerative(X, method)[:, 2]), np.sort(linkage(X, method)[:, 2]), atol=1e-8)


def test_cut_tree_and_cophenetic_match_scipy():
    from scipy.cluster.hierarchy import cophenet, fcluster, linkage
    from scipy.spatial.distance import pdist
    X, y = blobs(60)
    Z = agglomerative(X, "average")
    S = linkage(X, "average")
    assert adjusted_rand(cut_tree(Z, 3), fcluster(S, 3, "maxclust")) == pytest.approx(1)
    assert cophenetic_corr(X, Z) == pytest.approx(cophenet(S, pdist(X))[0], rel=1e-6)


def test_dendrogram_covers_every_leaf():
    X, _ = blobs(12)
    Z = agglomerative(X)
    order, segs = dendrogram_segments(Z)
    assert sorted(order) == list(range(12)) and order == leaf_order(Z) and len(segs) == 3 * 11


def frame(cols=("A",), effect=0.0):
    """Ten cities, three years, flat at 0 except a shift of `effect` in the 2020 lockdown window."""
    dates = pd.date_range("2018-01-01", "2020-06-30")
    rows = []
    rng = np.random.default_rng(0)
    for city in range(10):
        v = rng.normal(0, 0.01, len(dates)) + city * 0.1
        v[(dates >= "2020-03-25") & (dates <= "2020-04-30")] += effect
        rows.append(pd.DataFrame({"City": f"C{city}", "Date": dates, "A": v}))
    return pd.concat(rows, ignore_index=True)


def test_difference_in_differences_recovers_a_planted_effect():
    r = evidence.lockdown(frame(effect=-0.5), ["A"])
    assert r[2020]["effect"]["A"] == pytest.approx(-0.5, abs=0.01)
    assert r[2020]["t"]["A"] < -10
    assert abs(r[2019]["effect"]["A"]) < 0.02


def test_difference_in_differences_finds_nothing_when_nothing_happened():
    assert abs(evidence.lockdown(frame(effect=0.0), ["A"])[2020]["effect"]["A"]) < 0.02
