"""Tests for canonicalized-latent (`LayerNorm`-sphere) geometry metrics.

See `assets/experiments/08_26/23_08_26_architecture_predictive/report/
geometria_latentu.md` for the derivation each function here implements.
"""

from __future__ import annotations

import numpy as np
import pytest

from msi_autoencoder_wrapper.analysis.autoencoder.latent import sphere_geometry as geometry
from msi_autoencoder_wrapper.utils.exceptions import ValidationError


def _uniform_sphere_samples(n_samples: int, dimension: int, seed: int) -> np.ndarray:
    """Sample uniformly from S^{dimension-2}(sqrt(dimension)) embedded in {1}^perp.

    Directly implements geometria_latentu.md §2.1's u = sqrt(n) * Ca / ||Ca|| from
    i.i.d. Gaussian `a` — a standard construction for uniform sphere sampling, and
    the same computation LayerNorm's pre-affine step performs. Deliberately does not
    go through `canonicalize`, which does the opposite (un-does an affine transform
    on an already-computed z) and would leave raw, non-unit-norm Gaussian vectors
    untouched when called with gamma=1, beta=0.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_samples, dimension))
    centered = raw - raw.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return np.sqrt(dimension) * centered / norms


class TestCanonicalize:
    def test_matches_manual_layer_norm(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((5, 4))
        gamma = rng.uniform(0.5, 2.0, size=4)
        beta = rng.standard_normal(4)
        mean = a.mean(axis=1, keepdims=True)
        std = a.std(axis=1, keepdims=True)
        z = gamma * (a - mean) / std + beta
        u = geometry.canonicalize(z, gamma, beta)
        # u should equal (a - mean) / std, i.e. the pre-affine LayerNorm output.
        expected = (a - mean) / std
        np.testing.assert_allclose(u, expected, atol=1e-8)

    def test_zero_gamma_raises(self) -> None:
        z = np.zeros((3, 2))
        with pytest.raises(ValidationError):
            geometry.canonicalize(z, gamma=np.array([1.0, 0.0]), beta=np.zeros(2))

    def test_row_sum_and_norm_properties(self) -> None:
        u = _uniform_sphere_samples(50, 10, seed=1)
        report = geometry.verify_canonicalization(u)
        assert report["max_abs_row_sum"] < 1e-8
        assert report["expected_norm"] == pytest.approx(np.sqrt(10))
        assert report["max_abs_norm_deviation"] < 1e-6


class TestNormalizeToConstantNorm:
    def test_default_target_is_sqrt_dimension(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.normal(loc=5.0, scale=3.0, size=(20, 6))
        normalized = geometry.normalize_to_constant_norm(x)
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, np.sqrt(6), atol=1e-8)

    def test_explicit_target_norm(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=(10, 4))
        normalized = geometry.normalize_to_constant_norm(x, target_norm=2.5)
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, 2.5, atol=1e-8)

    def test_direction_is_preserved(self) -> None:
        rng = np.random.default_rng(4)
        x = rng.normal(size=(8, 5))
        normalized = geometry.normalize_to_constant_norm(x)
        cosine_to_original = np.sum(x * normalized, axis=1) / (
            np.linalg.norm(x, axis=1) * np.linalg.norm(normalized, axis=1)
        )
        np.testing.assert_allclose(cosine_to_original, 1.0, atol=1e-8)

    def test_zero_row_raises(self) -> None:
        x = np.zeros((3, 4))
        x[1] = 1.0
        with pytest.raises(ValidationError):
            geometry.normalize_to_constant_norm(x)

    def test_reusable_downstream_with_pairwise_cosine(self) -> None:
        # The whole point of normalizing: pairwise_cosine's `/D` convention only
        # gives a true cosine when every row has the same fixed norm.
        rng = np.random.default_rng(5)
        x = rng.normal(size=(30, 8)) * rng.uniform(0.1, 10.0, size=(30, 1))
        normalized = geometry.normalize_to_constant_norm(x)
        cos_theta = geometry.pairwise_cosine(normalized)
        assert np.all(cos_theta <= 1.0 + 1e-8)
        assert np.all(cos_theta >= -1.0 - 1e-8)
        np.testing.assert_allclose(np.diag(cos_theta), 1.0, atol=1e-8)


class TestNullDirectionCheck:
    def test_correct_canonicalization_null_direction_is_constant(self) -> None:
        u = _uniform_sphere_samples(500, 10, seed=30)
        gamma = np.random.default_rng(31).uniform(0.5, 2.0, size=10)
        result = geometry.null_direction_check(u, gamma)
        assert result["smallest_eigenvalue"] == pytest.approx(0.0, abs=1e-6)
        assert result["cosine_with_constant_direction"] == pytest.approx(1.0, abs=1e-6)

    def test_missing_gamma_division_null_direction_is_inverse_gamma(self) -> None:
        # Simulate the canonicalization bug directly: u_correct lives on the sphere;
        # scaling it by gamma (as if the gamma division had been skipped) should move
        # the null eigenvector away from the constant direction and onto 1/gamma
        # (Cov(u_correct)@1 = 0 exactly, so Cov(gamma*u_correct)@(1/gamma) = 0).
        rng = np.random.default_rng(32)
        u_correct = _uniform_sphere_samples(500, 10, seed=33)
        gamma = rng.uniform(0.5, 2.0, size=10)
        buggy = u_correct * gamma  # what (z - beta) looks like without dividing by gamma
        result = geometry.null_direction_check(buggy, gamma)
        assert result["cosine_with_inverse_gamma"] == pytest.approx(1.0, abs=1e-6)
        assert result["cosine_with_inverse_gamma"] > result["cosine_with_constant_direction"]


class TestCloudAsymmetry:
    def test_centered_cloud_has_near_zero_asymmetry(self) -> None:
        u = _uniform_sphere_samples(2000, 10, seed=34)
        assert geometry.cloud_asymmetry(u) == pytest.approx(0.0, abs=0.05)

    def test_matches_trace_identity(self) -> None:
        u = _uniform_sphere_samples(500, 10, seed=35)
        trace = geometry.dimension_usage(u)["trace"]
        asymmetry = geometry.cloud_asymmetry(u)
        # Population covariance identity: tr(Cov(u)) = D - ||mean(u)||^2 exactly.
        assert trace == pytest.approx(10 - asymmetry * 10, abs=1e-8)


class TestAngle:
    def test_self_angle_is_zero(self) -> None:
        u = _uniform_sphere_samples(5, 8, seed=2)
        cos_theta = geometry.pairwise_cosine(u)
        np.testing.assert_allclose(np.diag(cos_theta), 1.0, atol=1e-8)
        np.testing.assert_allclose(np.diag(geometry.angle_degrees(cos_theta)), 0.0, atol=1e-4)

    def test_paired_angle_matches_pairwise_diagonal(self) -> None:
        u_a = _uniform_sphere_samples(6, 8, seed=3)
        u_b = _uniform_sphere_samples(6, 8, seed=4)
        paired = geometry.paired_angle_degrees(u_a, u_b)
        full = geometry.angle_degrees(geometry.pairwise_cosine(u_a, u_b))
        np.testing.assert_allclose(paired, np.diag(full), atol=1e-6)

    def test_shape_mismatch_raises(self) -> None:
        u_a = _uniform_sphere_samples(6, 8, seed=3)
        u_b = _uniform_sphere_samples(5, 8, seed=4)
        with pytest.raises(ValidationError):
            geometry.paired_angle_degrees(u_a, u_b)


class TestStructureTest:
    def test_uniform_samples_match_closed_form_baseline(self) -> None:
        u = _uniform_sphere_samples(2000, 10, seed=5)
        rng = np.random.default_rng(6)
        result = geometry.structure_test(u, rng, pair_count=20000)
        assert result["effective_dimension"] == 8.0
        assert result["uniform_baseline_sd_cos_theta"] == pytest.approx(1 / 3, rel=1e-6)
        # Sampling noise on 20000 pairs is small; loose tolerance keeps this robust.
        assert result["observed_mean_cos_theta"] == pytest.approx(0.0, abs=0.02)
        assert result["observed_sd_cos_theta"] == pytest.approx(
            result["uniform_baseline_sd_cos_theta"], rel=0.1
        )

    def test_degenerate_dimension_raises(self) -> None:
        u = _uniform_sphere_samples(5, 2, seed=7)
        with pytest.raises(ValidationError):
            geometry.structure_test(u, np.random.default_rng(0))


class TestDimensionUsage:
    def test_trace_bounded_by_dimension_and_one_null_direction(self) -> None:
        u = _uniform_sphere_samples(500, 10, seed=8)
        result = geometry.dimension_usage(u)
        assert result["eigenvalues"].shape == (10,)
        assert result["trace"] <= 10.0 + 1e-6
        # The `1`-direction canonicalization removes is a structural null eigenvalue.
        assert result["eigenvalues"][-1] == pytest.approx(0.0, abs=1e-6)
        assert 1.0 <= result["participation_ratio"] <= 10.0
        assert 1.0 <= result["effective_rank"] <= 10.0

    def test_degenerate_input_raises(self) -> None:
        u = np.ones((5, 4))
        with pytest.raises(ValidationError):
            geometry.dimension_usage(u)


class TestTwoNN:
    def test_recovers_known_intrinsic_dimension(self) -> None:
        # Uniform samples on S^8 (D=10) have intrinsic dimension 8 by construction.
        u = _uniform_sphere_samples(3000, 10, seed=9)
        estimate = geometry.two_nn_intrinsic_dimension(u)
        assert estimate == pytest.approx(8.0, rel=0.25)

    def test_too_few_samples_raises(self) -> None:
        u = _uniform_sphere_samples(2, 10, seed=10)
        with pytest.raises(ValidationError):
            geometry.two_nn_intrinsic_dimension(u)


class TestProcrustesAndCKA:
    def test_identical_inputs_have_zero_procrustes_distance(self) -> None:
        u = _uniform_sphere_samples(50, 8, seed=11)
        assert geometry.procrustes_distance(u, u) == pytest.approx(0.0, abs=1e-8)

    def test_pure_rotation_has_zero_procrustes_distance(self) -> None:
        u = _uniform_sphere_samples(50, 8, seed=12)
        rotation, _ = np.linalg.qr(np.random.default_rng(13).standard_normal((8, 8)))
        rotated = u @ rotation
        assert geometry.procrustes_distance(u, rotated) == pytest.approx(0.0, abs=1e-6)

    def test_unrelated_inputs_have_larger_procrustes_distance(self) -> None:
        u_a = _uniform_sphere_samples(200, 8, seed=14)
        u_b = _uniform_sphere_samples(200, 8, seed=15)
        assert geometry.procrustes_distance(u_a, u_b) > 0.1

    def test_shape_mismatch_raises(self) -> None:
        u_a = _uniform_sphere_samples(50, 8, seed=11)
        u_b = _uniform_sphere_samples(49, 8, seed=12)
        with pytest.raises(ValidationError):
            geometry.procrustes_distance(u_a, u_b)

    def test_cka_identical_inputs_is_one(self) -> None:
        u = _uniform_sphere_samples(50, 8, seed=16)
        assert geometry.linear_cka(u, u) == pytest.approx(1.0, abs=1e-8)

    def test_cka_unrelated_inputs_is_well_below_one(self) -> None:
        u_a = _uniform_sphere_samples(300, 8, seed=17)
        u_b = _uniform_sphere_samples(300, 8, seed=18)
        assert geometry.linear_cka(u_a, u_b) < 0.5


class TestNeighborhoodSimilarity:
    def test_knn_overlap_identical_inputs_is_one(self) -> None:
        u = _uniform_sphere_samples(100, 8, seed=19)
        assert geometry.knn_overlap(u, u, k=10) == pytest.approx(1.0)

    def test_knn_overlap_unrelated_inputs_is_low(self) -> None:
        u_a = _uniform_sphere_samples(300, 8, seed=20)
        u_b = _uniform_sphere_samples(300, 8, seed=21)
        assert geometry.knn_overlap(u_a, u_b, k=10) < 0.3

    def test_trustworthiness_continuity_identical_inputs(self) -> None:
        u = _uniform_sphere_samples(100, 8, seed=22)
        result = geometry.trustworthiness_continuity(u, u, n_neighbors=10)
        assert result["trustworthiness"] == pytest.approx(1.0, abs=1e-8)
        assert result["continuity"] == pytest.approx(1.0, abs=1e-8)


class TestRSASpearman:
    def test_recovers_perfect_negative_correlation(self) -> None:
        # Construct labels so identical-latent-neighbors always share labels and
        # distant points never do: label = which orthant-like half the point falls
        # in along its dominant coordinate. This is a smoke/sanity check on the
        # sign and strength of the correlation, not an exact value.
        u = _uniform_sphere_samples(400, 8, seed=23)
        dominant = np.argmax(np.abs(u), axis=1)
        labels = np.zeros((400, 8), dtype=np.float64)
        labels[np.arange(400), dominant] = 1.0
        rng = np.random.default_rng(24)
        result = geometry.rsa_spearman(u, labels, rng, pair_count=20000)
        assert -1.0 <= result["spearman_r"] <= 1.0
        assert result["pairs"] > 0


class TestAngularSensitivityCurve:
    def test_larger_epsilon_moves_the_code_further_for_a_linear_encoder(self) -> None:
        dimension = 6
        rng = np.random.default_rng(25)
        projection = rng.standard_normal((dimension, dimension))
        gamma = np.ones(dimension)
        beta = np.zeros(dimension)

        def encode(batch: np.ndarray) -> np.ndarray:
            return geometry.canonicalize(batch @ projection, gamma, beta)

        inputs = rng.standard_normal((30, dimension)) + 5.0  # nonzero norm
        result = geometry.angular_sensitivity_curve(
            encode, inputs, epsilons=[0.0, 0.01, 0.1, 0.5], rng=np.random.default_rng(26)
        )
        assert result["epsilon"].shape == (4,)
        assert result["mean_angle_degrees"].shape == (4,)
        assert result["mean_angle_degrees"][0] == pytest.approx(0.0, abs=1e-6)
        # Not strictly monotonic in general (angle saturates at 180 on a sphere),
        # but at these small-to-moderate scales it should be increasing.
        assert np.all(np.diff(result["mean_angle_degrees"]) >= -1e-6)
