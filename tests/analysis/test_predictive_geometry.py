"""Analytical geometry invariants and train-only readout fitting."""

import numpy as np
import pytest

from msi_autoencoder_wrapper.analysis.autoencoder.latent.predictive_geometry import (
    geometry_tables, representation_similarity, ridge_probe,
)


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
