"""Agreement of the device-accelerated latent statistics with the numpy reference."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from msi_autoencoder_wrapper.analysis.autoencoder.latent import sphere_geometry
from msi_autoencoder_wrapper.analysis.autoencoder.latent.batch_geometry import (
    knn_overlap_batch,
    two_nn_intrinsic_dimension_batch,
)
from msi_autoencoder_wrapper.utils.exceptions import ValidationError

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _codes(samples: int, dimension: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.normal(size=(samples, dimension))


@pytest.mark.parametrize("device", DEVICES)
class TestTwoNearestNeighbourDimension:
    """Intrinsic dimension estimates must equal the reference."""

    @pytest.mark.parametrize("samples,dimension,seed", [(200, 6, 0), (500, 10, 1), (80, 3, 2)])
    def test_matches_the_reference(self, device, samples, dimension, seed) -> None:
        codes = _codes(samples, dimension, seed)

        expected = sphere_geometry.two_nn_intrinsic_dimension(codes)
        produced = two_nn_intrinsic_dimension_batch(codes, device=device)

        assert produced == pytest.approx(expected, rel=1e-9)

    def test_matches_on_a_low_dimensional_manifold(self, device) -> None:
        # Codes concentrated on a plane: the estimate is far from the ambient
        # dimension, so a discrepancy would be obvious rather than absorbed.
        generator = np.random.default_rng(3)
        plane = generator.normal(size=(300, 2))
        codes = np.concatenate([plane, 1e-3 * generator.normal(size=(300, 6))], axis=1)

        expected = sphere_geometry.two_nn_intrinsic_dimension(codes)
        produced = two_nn_intrinsic_dimension_batch(codes, device=device)

        assert produced == pytest.approx(expected, rel=1e-9)

    def test_too_few_samples_is_rejected(self, device) -> None:
        with pytest.raises(ValidationError):
            two_nn_intrinsic_dimension_batch(_codes(2, 4, 4), device=device)

    def test_coincident_points_are_rejected(self, device) -> None:
        codes = np.ones((10, 4))

        with pytest.raises(ValidationError):
            two_nn_intrinsic_dimension_batch(codes, device=device)


@pytest.mark.parametrize("device", DEVICES)
class TestNeighbourhoodOverlap:
    """Neighborhood agreement must equal the reference."""

    @pytest.mark.parametrize("k", [5, 10, 25])
    def test_matches_the_reference_for_unrelated_representations(self, device, k) -> None:
        first, second = _codes(300, 8, 10), _codes(300, 8, 11)

        expected = sphere_geometry.knn_overlap(first, second, k=k)
        produced = knn_overlap_batch(first, second, k=k, device=device)

        assert produced == pytest.approx(expected, abs=1e-12)

    def test_identical_representations_overlap_completely(self, device) -> None:
        codes = _codes(200, 6, 12)

        assert knn_overlap_batch(codes, codes.copy(), k=10, device=device) == pytest.approx(1.0)

    def test_matches_the_reference_for_a_rotated_representation(self, device) -> None:
        # A rotation preserves neighborhoods exactly, so the overlap must be one and
        # any indexing error in the batched form would break it.
        codes = _codes(200, 6, 13)
        rotation, _ = np.linalg.qr(np.random.default_rng(14).normal(size=(6, 6)))
        rotated = codes @ rotation

        expected = sphere_geometry.knn_overlap(codes, rotated, k=10)
        produced = knn_overlap_batch(codes, rotated, k=10, device=device)

        assert produced == pytest.approx(expected, abs=1e-12)

    def test_matches_the_reference_for_a_partially_shared_structure(self, device) -> None:
        base = _codes(250, 8, 15)
        perturbed = base + 0.35 * _codes(250, 8, 16)

        expected = sphere_geometry.knn_overlap(base, perturbed, k=10)
        produced = knn_overlap_batch(base, perturbed, k=10, device=device)

        assert 0.0 < produced < 1.0
        assert produced == pytest.approx(expected, abs=1e-12)

    def test_mismatched_row_counts_are_rejected(self, device) -> None:
        with pytest.raises(ValidationError):
            knn_overlap_batch(_codes(50, 4, 17), _codes(40, 4, 18), k=5, device=device)

    def test_neighbourhood_larger_than_the_sample_is_rejected(self, device) -> None:
        with pytest.raises(ValidationError):
            knn_overlap_batch(_codes(8, 4, 19), _codes(8, 4, 20), k=10, device=device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_and_cpu_agree() -> None:
    """The device must not change either statistic."""
    codes_a, codes_b = _codes(400, 10, 21), _codes(400, 10, 22)

    assert two_nn_intrinsic_dimension_batch(codes_a, device="cuda") == pytest.approx(
        two_nn_intrinsic_dimension_batch(codes_a, device="cpu"), rel=1e-9
    )
    assert knn_overlap_batch(codes_a, codes_b, k=10, device="cuda") == pytest.approx(
        knn_overlap_batch(codes_a, codes_b, k=10, device="cpu"), abs=1e-12
    )
