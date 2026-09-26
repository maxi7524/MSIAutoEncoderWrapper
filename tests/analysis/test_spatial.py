"""Deterministic checks of spatial image, ion-image and segmentation helpers."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from msi_autoencoder_wrapper.analysis.autoencoder.spatial import (
    MergedPixelMap,
    adjusted_rand_between,
    align_labels,
    assemble_image,
    fit_segmentation,
    ion_bin_indices,
    ion_intensities,
    presence_auc,
    segment_contingency,
    segment_marker_table,
    spatial_agreement,
)
from msi_autoencoder_wrapper.utils.exceptions import ValidationError


def test_assemble_image_places_values_by_coordinates_and_leaves_gaps_nan() -> None:
    image = assemble_image(np.array([1, 3, 2]), np.array([5, 5, 6]), np.array([1.0, 2.0, 3.0]))

    assert image.shape == (2, 3)
    np.testing.assert_array_equal(image[0], [1.0, np.nan, 2.0])
    np.testing.assert_array_equal(image[1], [np.nan, 3.0, np.nan])
    with pytest.raises(ValidationError):
        assemble_image(np.array([1, 1]), np.array([2, 2]), np.array([0.0, 1.0]))


def test_merged_pixel_map_follows_segment_start_offset_and_step(tmp_path) -> None:
    path = tmp_path / "store.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE datasets_metadata (dataset_index INTEGER, source_dataset_id TEXT, name TEXT,
                                            source_imzml_path TEXT);
            CREATE TABLE pixel_segments (dataset_index INTEGER, merged_pixel_start INTEGER,
                                         segment_length INTEGER, source_pixel_start INTEGER, source_step INTEGER);
            INSERT INTO datasets_metadata VALUES (1, 'a', 'A', '/a.imzML'), (2, 'b', 'B', '/b.imzML');
            INSERT INTO pixel_segments VALUES (1, 0, 4, 10, 1), (2, 4, 3, 0, 5);
            """
        )
    mapping = MergedPixelMap.from_store(path)
    resolved = mapping.resolve(np.array([6, 0, 3, 4]))

    assert resolved.dataset_id.tolist() == ["b", "a", "a", "b"]
    assert resolved.source_pixel.tolist() == [10, 10, 13, 0]
    assert mapping.dataset_ranges(["b"]) == [(4, 7)]
    with pytest.raises(ValueError):
        mapping.resolve(np.array([7]))


def test_ion_support_is_clipped_to_the_axis_and_empty_outside_it() -> None:
    edges = np.arange(100.0, 106.0, 1.0)  # five bins
    supports = ion_bin_indices(np.array([100.2, 102.5, 104.9, 99.0]), edges, radius=1)

    assert [support.tolist() for support in supports] == [[0, 1], [1, 2, 3], [3, 4], []]
    spectra = np.arange(10, dtype=np.float32).reshape(2, 5)
    intensities = ion_intensities(spectra, supports)
    np.testing.assert_allclose(intensities[:, :3], [[1, 6, 7], [11, 21, 17]])
    assert np.isnan(intensities[:, 3]).all()


def test_agreement_is_rank_based_and_auc_uses_average_ranks() -> None:
    reference = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert spatial_agreement(reference, np.exp(reference))["spearman"] == pytest.approx(1.0)
    assert np.isnan(spatial_agreement(reference, np.ones(5))["spearman"])

    present = np.array([False, False, True, True])
    assert presence_auc(np.array([0.1, 0.2, 0.8, 0.9]), present) == pytest.approx(1.0)
    assert presence_auc(np.array([0.9, 0.8, 0.2, 0.1]), present) == pytest.approx(0.0)
    assert presence_auc(np.ones(4), present) == pytest.approx(0.5)
    assert np.isnan(presence_auc(np.ones(4), np.zeros(4, dtype=bool)))


def test_segmentation_recovers_separated_clusters_and_their_marker_bins() -> None:
    generator = np.random.default_rng(0)
    truth = np.repeat([0, 1, 2], 60)
    centres = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
    latent = centres[truth] + generator.normal(scale=0.3, size=(truth.size, 2))  # (N, 2)
    fit = fit_segmentation(latent, 3, seed=1, sample_size=120)
    labels = fit.predict(latent)

    assert adjusted_rand_between(labels, truth) == pytest.approx(1.0)
    np.testing.assert_array_equal(align_labels(truth, labels, 3), truth)

    ## Segment s carries its intensity in bin 2 * s; all other bins share a flat floor
    spectra = np.full((truth.size, 6), 0.05)
    spectra[np.arange(truth.size), 2 * truth] = 1.0
    markers = segment_marker_table(spectra, truth, np.arange(6.0), top=1, minimum_mean=1e-3)
    assert markers.sort_values("segment").bin.tolist() == [0, 2, 4]


def test_segment_contingency_counts_overlaps_and_normalizes_per_side() -> None:
    reference = np.array([0, 0, 0, 1, 1, 2])
    other = np.array([0, 0, 1, 1, 1, 1])
    table = segment_contingency(reference, other, 3).set_index(["reference_segment", "other_segment"])

    assert table.pixels.sum() == reference.size
    assert table.loc[(0, 0), "pixels"] == 2 and table.loc[(0, 1), "pixels"] == 1
    ## Reference segment 0 has 3 pixels; compared segment 1 has 4 pixels
    assert table.loc[(0, 0), "reference_fraction"] == pytest.approx(2 / 3)
    assert table.loc[(2, 1), "other_fraction"] == pytest.approx(1 / 4)
    ## Compared segment 2 is empty, so its column share is undefined
    assert np.isnan(table.loc[(0, 2), "other_fraction"])
    with pytest.raises(Exception):
        segment_contingency(reference, np.array([0, 0, 1, 1, 1, 3]), 3)
