"""Analytical geometry invariants and train-only readout fitting."""

import numpy as np
import pytest

from msi_autoencoder_wrapper.analysis.autoencoder.latent.predictive_geometry import (
    geometry_tables, intrinsic_dimension_estimate, label_structure_correlation,
    pairwise_geometry_battery, representation_similarity, ridge_probe, structure_summary,
)


def _canonicalized_sphere_sample(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    # A genuine canonicalized code: zero row-sum (the LayerNorm-affine-removed constraint)
    # and constant norm sqrt(d), which structure_test/two_nn/procrustes all assume.
    raw = rng.normal(size=(n, d))
    centered = raw - raw.mean(axis=1, keepdims=True)
    return centered / np.linalg.norm(centered, axis=1, keepdims=True) * np.sqrt(d)


def test_rank_one_cloud_and_rotation_invariance():
    t = np.arange(-3, 4, dtype=float)
    z = np.column_stack([t, 2 * t, -t])
    labels = (t[:, None] > 0)
    tables = geometry_tables(z, labels, np.arange(7), k=2)
    summary = tables["geometry"].set_index("metric").value
    assert summary["effective_rank"] == pytest.approx(1., abs=1e-12)
    assert summary["participation_ratio"] == pytest.approx(1., abs=1e-12)
    assert tables["geometry_pairs"].query("metric == 'euclidean'")["count"].sum() == 21
    assert representation_similarity(z, z[:, ::-1]) == pytest.approx(1., abs=1e-12)


def test_collapsed_cloud_is_reported_and_cka_is_undefined():
    z = np.ones((5, 3))
    tables = geometry_tables(z, np.ones((5, 2)), np.arange(5))
    summary = tables["geometry"].set_index("metric").value
    assert summary["collapsed"] == 1
    assert summary["effective_rank"] == 0
    assert summary["duplicate_pair_fraction"] == 1
    assert np.isnan(representation_similarity(z, z))


def test_ridge_probe_recovers_known_order_without_fitting_evaluation_data():
    train = np.array([[-2., 1.], [-1., 1.], [1., 1.], [2., 1.]])
    labels = np.array([[0.], [0.], [1.], [1.]])
    evaluated = np.array([[-3., 1.], [0., 1.], [3., 1.]])
    scores = ridge_probe(train, labels, evaluated)
    assert scores.shape == (3, 1)
    assert np.isfinite(scores).all()
    assert np.all(np.diff(scores[:, 0]) > 0)
    assert scores[1, 0] == pytest.approx(.5, abs=1e-12)
    # Adding a held-out extreme cannot change the score of an existing sample.
    extended = ridge_probe(train, labels, np.concatenate([evaluated, [[1e8, 1.]]]))
    np.testing.assert_allclose(scores, extended[:3], atol=1e-12, rtol=0)


def test_structure_summary_reports_an_isotropic_sample_as_close_to_the_uniform_null():
    rng = np.random.default_rng(11)
    u = _canonicalized_sphere_sample(rng, n=800, d=10)
    result = structure_summary(u, np.arange(800), np.random.default_rng(12), pair_count=20000)
    assert "cos_theta_samples" not in result
    assert result["observed_mean_cos_theta"] == pytest.approx(0.0, abs=0.05)
    assert result["observed_sd_cos_theta"] == pytest.approx(result["uniform_baseline_sd_cos_theta"], rel=0.25)
    assert result["effective_dimension"] == 8.0


def test_intrinsic_dimension_estimate_is_near_the_ambient_manifold_dimension():
    rng = np.random.default_rng(21)
    d = 10
    u = _canonicalized_sphere_sample(rng, n=500, d=d)
    estimate = intrinsic_dimension_estimate(u, np.arange(500))
    assert estimate == pytest.approx(d - 2, abs=3)  # Facco et al. estimator, small-sample noise.


def test_pairwise_geometry_battery_is_perfect_for_identical_representations():
    rng = np.random.default_rng(31)
    u = _canonicalized_sphere_sample(rng, n=100, d=6)
    result = pairwise_geometry_battery(u, u, np.arange(100), k=10)
    assert result["procrustes_distance"] == pytest.approx(0.0, abs=1e-8)
    assert result["linear_cka"] == pytest.approx(1.0, abs=1e-8)
    assert result["knn_overlap"] == pytest.approx(1.0, abs=1e-12)
    assert result["trustworthiness"] == pytest.approx(1.0, abs=1e-8)
    assert result["continuity"] == pytest.approx(1.0, abs=1e-8)


def test_pairwise_geometry_battery_detects_a_reshuffled_representation():
    rng = np.random.default_rng(41)
    u = _canonicalized_sphere_sample(rng, n=200, d=6)
    shuffled = rng.permutation(u)  # Same point cloud, unrelated pixel-to-row assignment.
    result = pairwise_geometry_battery(u, shuffled, np.arange(200), k=10)
    assert result["knn_overlap"] < 0.5
    assert result["trustworthiness"] < 0.9


def test_label_structure_correlation_detects_a_planted_cluster_label_relationship():
    # Two well-separated canonicalized directions (D=3, zero row-sum, norm sqrt(3)),
    # replicated three times each; label matches cluster membership exactly, so angle
    # and label dissimilarity must be positively associated.
    direction_a = np.array([2., -1., -1.])
    direction_b = np.array([-1., 2., -1.])
    point_a = direction_a / np.linalg.norm(direction_a) * np.sqrt(3)
    point_b = direction_b / np.linalg.norm(direction_b) * np.sqrt(3)
    u = np.vstack([np.tile(point_a, (3, 1)), np.tile(point_b, (3, 1))])
    labels = np.array([[1, 0]] * 3 + [[0, 1]] * 3)
    result = label_structure_correlation(u, np.arange(6), labels, np.random.default_rng(51), pair_count=500)
    assert result["spearman_r"] > 0.5
    assert result["pairs"] > 0
    assert np.isfinite(result["p_value"])
