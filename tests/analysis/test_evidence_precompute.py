"""Equivalence and exactness tests for the evidence-threshold analysis.

The whole package rests on one claim: sweeping a threshold over a histogram of the
relative evidence reproduces exactly what
:class:`~msi_autoencoder_wrapper.data.annotation_evidence.SignalEvidencePolicy`
would decide pixel by pixel. These tests check that claim directly, against the
training-time implementation, rather than checking that the analysis code runs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from msi_autoencoder_wrapper.analysis.autoencoder.evidence.precompute import (
    EvidenceGrid,
    build_evidence_grid,
    cumulative_below,
    evidence_signals,
)
from msi_autoencoder_wrapper.analysis.autoencoder.evidence.threshold_analysis import (
    state_yield_records,
)
from msi_autoencoder_wrapper.data.annotation_evidence import (
    NEGATIVE,
    POSITIVE,
    UNLABELLED,
    IonCatalogue,
    SignalEvidencePolicy,
)
from msi_autoencoder_wrapper.utils.exceptions import ValidationError


CLASS_COUNT = 7
FEATURE_COUNT = 40
PIXEL_COUNT = 64
SEED = 20260907


@pytest.fixture
def population() -> tuple[IonCatalogue, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a deterministic sparse population with a realistic zero fraction.

    Intensities are exponentially distributed and then zeroed at random so that a
    large share of ion positions carries no signal at all, which is the regime the
    evidence rule operates in and the one where the ``r = 0`` bucket matters.
    """
    generator = torch.Generator().manual_seed(SEED)
    spectra = torch.rand((PIXEL_COUNT, FEATURE_COUNT), generator=generator).pow(4)  # (B, M)
    spectra = spectra * (torch.rand((PIXEL_COUNT, FEATURE_COUNT), generator=generator) > 0.6)  # (B, M)
    bins = tuple(
        (int(position),) for position in torch.randperm(FEATURE_COUNT, generator=generator)[:CLASS_COUNT]
    )
    catalogue = IonCatalogue(tuple(f"ion_{index}" for index in range(CLASS_COUNT)), bins, FEATURE_COUNT)
    targets = (torch.rand((PIXEL_COUNT, CLASS_COUNT), generator=generator) < 0.2).float()  # (B, C)
    mask = torch.rand((PIXEL_COUNT, CLASS_COUNT), generator=generator) < 0.9  # (B, C)
    return catalogue, spectra, targets, mask


def _padded_index(catalogue: IonCatalogue) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a catalogue into the rectangular gather index the pass uses."""
    width = max(len(bins) for bins in catalogue.bins)
    index = torch.zeros((len(catalogue.bins), width), dtype=torch.long)
    valid = torch.zeros((len(catalogue.bins), width), dtype=torch.bool)
    for row, bins in enumerate(catalogue.bins):
        index[row, : len(bins)] = torch.as_tensor(bins, dtype=torch.long)
        valid[row, : len(bins)] = True
    return index, valid


@pytest.mark.parametrize("bin_radius", [0, 1, 2])
@pytest.mark.parametrize("relative_threshold", [0.001, 0.005, 0.01, 0.05, 0.2])
def test_relative_evidence_reproduces_the_training_state_assignment(
    population, bin_radius: int, relative_threshold: float
) -> None:
    """``r <= beta`` must select exactly the entries the policy calls negative."""
    catalogue, spectra, targets, mask = population
    states = SignalEvidencePolicy(
        absolute_threshold=0.0, relative_threshold=relative_threshold, bin_radius=bin_radius
    ).classify(spectra, targets, mask, catalogue)  # (B, C)

    index, valid = _padded_index(catalogue)
    _, relative = evidence_signals(spectra, index, valid, bin_radius)  # (B, C)
    unannotated = (targets <= 0.5) & mask
    expected_negative = unannotated & (relative <= relative_threshold)  # (B, C)

    assert torch.equal(states == NEGATIVE, expected_negative)
    assert torch.equal(states == UNLABELLED, unannotated & ~expected_negative)
    assert torch.equal(states == POSITIVE, (targets > 0.5) & mask)


def test_absolute_evidence_reproduces_the_training_state_assignment(population) -> None:
    """The unnormalized evidence must drive a pure absolute threshold identically."""
    catalogue, spectra, targets, mask = population
    absolute_threshold = 0.05
    states = SignalEvidencePolicy(
        absolute_threshold=absolute_threshold, relative_threshold=0.0, bin_radius=1
    ).classify(spectra, targets, mask, catalogue)  # (B, C)

    index, valid = _padded_index(catalogue)
    signal, _ = evidence_signals(spectra, index, valid, 1)  # (B, C)
    unannotated = (targets <= 0.5) & mask
    assert torch.equal(states == NEGATIVE, unannotated & (signal <= absolute_threshold))


def test_histogram_sweep_counts_match_a_direct_entry_by_entry_count(population) -> None:
    """A cumulative sum over buckets must equal the count the rule would produce."""
    catalogue, spectra, targets, mask = population
    grid = build_evidence_grid()
    index, valid = _padded_index(catalogue)
    _, relative = evidence_signals(spectra, index, valid, 1)  # (B, C)
    unannotated = (targets <= 0.5) & mask

    edges = torch.as_tensor(grid.relative_edges, dtype=torch.float32)
    buckets = torch.searchsorted(edges, relative[unannotated].contiguous())
    counts = torch.bincount(buckets, minlength=grid.relative_buckets).numpy()

    for threshold in (0.0, 0.001, 0.005, 0.01, 0.05):
        direct = int((relative[unannotated] <= threshold).sum())
        assert int(cumulative_below(counts, grid.relative_edges, threshold)) == direct


def test_cumulative_below_refuses_a_threshold_off_the_grid() -> None:
    """An interpolated answer would be silently wrong, so it must raise instead."""
    grid = build_evidence_grid()
    counts = np.zeros(grid.relative_buckets, dtype=np.int64)
    with pytest.raises(ValidationError):
        cumulative_below(counts, grid.relative_edges, 0.00737)


def test_build_evidence_grid_carries_every_candidate_threshold() -> None:
    """Candidate thresholds must be boundaries, otherwise the sweep is not exact."""
    candidates = (0.0005, 0.005, 0.01, 0.0333)
    grid = build_evidence_grid(candidates)
    for candidate in candidates:
        assert np.isclose(grid.relative_edges, candidate).any()
    assert grid.relative_edges[0] == 0.0
    assert np.all(np.diff(grid.relative_edges) > 0)


def _statistics_from(counts_annotated: np.ndarray, counts_unannotated: np.ndarray, grid: EvidenceGrid):
    """Wrap two histograms into the minimal statistics object the sweep reads."""
    from msi_autoencoder_wrapper.analysis.autoencoder.evidence.precompute import EvidenceStatistics

    empty = np.zeros((1, 1, grid.absolute_buckets), dtype=np.int64)
    return EvidenceStatistics(
        class_names=("ion_0",),
        class_bin_counts=np.ones(1, dtype=np.int32),
        class_mz=np.asarray([500.0]),
        class_bin_centre=np.asarray([500.0]),
        bin_radii=(1,),
        annotated_relative=counts_annotated,
        unannotated_relative=counts_unannotated,
        annotated_absolute=empty,
        unannotated_absolute=empty,
        background_relative=counts_unannotated[0, 0],
        background_absolute=empty[0, 0],
        group_relative=np.zeros((1, 1, 2, grid.relative_buckets), dtype=np.int64),
        joint_relative_maximum=np.zeros((1, grid.relative_buckets, grid.maximum_buckets), dtype=np.int64),
        offset_values=np.asarray([-1, 0, 1]),
        offset_annotated=np.zeros((1, 3), dtype=np.int64),
        offset_unannotated=np.zeros(3, dtype=np.int64),
        group_names=("acquisition_0",),
        spectrum_ids=np.zeros(1, dtype=np.int64),
        spectrum_group_index=np.zeros(1, dtype=np.int32),
        spectrum_maximum=np.asarray([0.05], dtype=np.float32),
        spectrum_nonzero_bins=np.asarray([10], dtype=np.int32),
        spectrum_annotated_count=np.asarray([1], dtype=np.int32),
        spectrum_available_count=np.asarray([1], dtype=np.int32),
        spectrum_negative_count=np.zeros((1, 1, 1), dtype=np.int32),
        candidate_relative_thresholds=np.asarray([0.005]),
        grid=grid,
        metadata={"feature_count": 40},
    )


def test_state_yield_is_monotone_and_partitions_the_unannotated_population() -> None:
    """Negative and uncertain counts must partition the population at every threshold."""
    grid = build_evidence_grid()
    generator = np.random.default_rng(SEED)
    annotated = generator.integers(0, 50, size=(1, 1, grid.relative_buckets))
    unannotated = generator.integers(0, 500, size=(1, 1, grid.relative_buckets))
    # The overflow bucket holds values above one, which the relative evidence cannot
    # reach: the per-ion maximum is taken from the same spectrum as the denominator.
    annotated[..., -1] = 0
    unannotated[..., -1] = 0
    statistics = _statistics_from(annotated, unannotated, grid)

    records = state_yield_records(statistics, bin_radius=1)
    total = float(unannotated.sum())
    negatives = [record["negative_entries"] for record in records]
    assert negatives == sorted(negatives)
    for record in records:
        assert record["negative_entries"] + record["uncertain_entries"] == pytest.approx(total)
    assert records[-1]["negative_entries"] == pytest.approx(total)


def test_round_trip_persistence_preserves_every_array(tmp_path) -> None:
    """A reloaded pass must reproduce the sweep the in-memory one produces."""
    grid = build_evidence_grid()
    generator = np.random.default_rng(SEED)
    statistics = _statistics_from(
        generator.integers(0, 50, size=(1, 1, grid.relative_buckets)),
        generator.integers(0, 500, size=(1, 1, grid.relative_buckets)),
        grid,
    )
    path = statistics.save(tmp_path / "evidence.npz")
    restored = type(statistics).load(path)

    assert restored.class_names == statistics.class_names
    assert restored.bin_radii == statistics.bin_radii
    np.testing.assert_array_equal(restored.unannotated_relative, statistics.unannotated_relative)
    np.testing.assert_array_equal(restored.grid.relative_edges, statistics.grid.relative_edges)
    assert state_yield_records(restored, bin_radius=1) == state_yield_records(statistics, bin_radius=1)
