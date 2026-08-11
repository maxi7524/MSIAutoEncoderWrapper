"""Unit tests for the batched Torch primitives shared by peak-based inverse binners."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.signal import find_peaks

from msi_autoencoder_wrapper.binners.inverse_strategies import peak_ops


# local_maxima_mask
def test_local_maxima_mask_matches_scipy_on_non_tied_signal() -> None:
    values = np.asarray([[0.0, 1.0, 5.0, 2.0, 0.0, 3.0, 3.5, 1.0, 4.0, 0.5]])
    expected, _ = find_peaks(values[0], height=0.0, distance=1)
    mask = peak_ops.local_maxima_mask(torch.tensor(values), min_distance=0, threshold=0.0)
    assert np.array_equal(np.flatnonzero(mask[0].numpy()), expected)


def test_local_maxima_mask_detects_boundary_peaks() -> None:
    # scipy's own interior algorithm never reports index 0 or -1; the reference reintroduces
    # them via a threshold + single-neighbor comparison (peak_region.py:134-137).
    values = torch.tensor([[9.0, 1.0, 0.0, 1.0, 9.0]])
    mask = peak_ops.local_maxima_mask(values, min_distance=0, threshold=0.0)
    assert np.array_equal(np.flatnonzero(mask[0].numpy()), [0, 4])


def test_local_maxima_mask_applies_threshold() -> None:
    values = torch.tensor([[0.0, 1.0, 0.0, 5.0, 0.0]])
    mask = peak_ops.local_maxima_mask(values, min_distance=0, threshold=2.0)
    assert np.array_equal(np.flatnonzero(mask[0].numpy()), [3])


def _reference_peaks_with_edges(values: np.ndarray, distance: int) -> np.ndarray:
    """Reproduce peak_region.py's combination of scipy peaks with its edge-peak rule."""
    interior, _ = find_peaks(values, height=0.0, distance=distance)
    edges = [
        index
        for index in (0, values.size - 1)
        if values.size and values[index] > 0.0
        and (values.size == 1 or values[index] >= values[1 if index == 0 else -2])
    ]
    return np.unique(np.concatenate((interior, np.asarray(edges, dtype=int))))


@pytest.mark.parametrize("distance", [1, 2])
def test_local_maxima_mask_matches_scipy_distance_nms(distance: int) -> None:
    # Exact for the common distance regime (1-2); larger distances on densely-packed
    # candidates are a documented approximation (single-pass vs. iterative greedy NMS).
    rng = np.random.default_rng(11)
    values = rng.random((5, 40)) * 10
    for row in values:
        expected = _reference_peaks_with_edges(row, distance)
        mask = peak_ops.local_maxima_mask(torch.tensor(row).unsqueeze(0), min_distance=distance, threshold=0.0)
        assert np.array_equal(np.flatnonzero(mask[0].numpy()), expected)


def test_local_maxima_mask_never_selects_interior_peaks_closer_than_distance() -> None:
    # Invariant that must hold even where the single-pass NMS approximation under-selects
    # relative to scipy's exact greedy result (see peak_ops.local_maxima_mask docstring).
    # Boundary positions are excluded: like the NumPy reference, edge peaks are unioned in
    # without an additional distance check against interior peaks (by design, matching
    # peak_region.py:134-138).
    rng = np.random.default_rng(5)
    values = torch.tensor(rng.random((10, 60)) * 10)
    distance = 4
    mask = peak_ops.local_maxima_mask(values, min_distance=distance, threshold=0.0)
    depth = values.shape[1]
    for row in mask:
        positions = torch.nonzero(row, as_tuple=True)[0]
        interior = positions[(positions > 0) & (positions < depth - 1)]
        if interior.numel() > 1:
            assert torch.all(interior[1:] - interior[:-1] >= distance)


def test_local_maxima_mask_plateau_does_not_crash_and_respects_threshold() -> None:
    # Exact-tie plateaus are a documented, accepted divergence source (see docstring): this
    # only asserts the mask stays well-formed and threshold-respecting, not distance parity.
    values = torch.tensor([[0.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0]])
    mask = peak_ops.local_maxima_mask(values, min_distance=3, threshold=0.0)
    positions = torch.nonzero(mask[0], as_tuple=True)[0]
    assert positions.numel() >= 1
    assert torch.all(values[0, positions] > 0.0)


# region_stop_masks / nearest_flagged / region_bounds_valley
def test_region_stop_masks_grow_through_flat_background() -> None:
    # A long flat run of exact-zero background (common in binned MSI spectra) must not be
    # treated as a wall of "valleys" -- the reference walk grows all the way through it.
    values = torch.tensor([[0.0, 1.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0]])
    left, right = peak_ops.region_bounds_valley(torch.tensor([[2]]), values, depth=8)
    assert left.item() == 0
    assert right.item() == 7


def test_nearest_flagged_left_and_right_directions() -> None:
    flags = torch.tensor([[True, False, False, True, False, False]])
    left = peak_ops.nearest_flagged(flags, "left", default=-1)
    right = peak_ops.nearest_flagged(flags, "right", default=-1)
    assert torch.equal(left[0], torch.tensor([0, 0, 0, 3, 3, 3]))
    assert torch.equal(right[0], torch.tensor([0, 3, 3, 3, -1, -1]))


# union_of_intervals_mask
def test_union_of_intervals_mask_covers_overlapping_and_disjoint_intervals() -> None:
    left = torch.tensor([[0, 4, 8]])
    right = torch.tensor([[2, 6, 8]])
    valid = torch.tensor([[True, True, True]])
    mask = peak_ops.union_of_intervals_mask(left, right, valid, depth=10)
    expected = torch.zeros(10, dtype=torch.bool)
    expected[0:3] = True
    expected[4:7] = True
    expected[8] = True
    assert torch.equal(mask[0], expected)


def test_union_of_intervals_mask_ignores_invalid_columns() -> None:
    left = torch.tensor([[0, 4]])
    right = torch.tensor([[2, 6]])
    valid = torch.tensor([[True, False]])
    mask = peak_ops.union_of_intervals_mask(left, right, valid, depth=10)
    expected = torch.zeros(10, dtype=torch.bool)
    expected[0:3] = True
    assert torch.equal(mask[0], expected)


def test_union_of_intervals_mask_supports_three_way_overlap() -> None:
    left = torch.tensor([[0, 1, 2]])
    right = torch.tensor([[5, 5, 5]])
    valid = torch.tensor([[True, True, True]])
    mask = peak_ops.union_of_intervals_mask(left, right, valid, depth=8)
    expected = torch.zeros(8, dtype=torch.bool)
    expected[0:6] = True
    assert torch.equal(mask[0], expected)
