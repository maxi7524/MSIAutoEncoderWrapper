"""Shared batched Torch primitives for peak selection and peak regions.

REMARK: ``min_peak_distance`` (peak-detection non-max suppression) and region growth /
envelope extraction are independent pipeline stages: distance-NMS only thins the candidate
set at detection time, while region growth and the interval-union coverage mask operate on
whichever peaks survive selection, regardless of how that set was produced.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch.nn import functional as torch_functional

from ...utils.exceptions import raise_validation_error


# Peak and valley detection
## Fused local-maximum detection with distance-based non-max suppression
def local_maxima_mask(
    values: torch.Tensor,
    min_distance: int,
    threshold: torch.Tensor | float,
) -> torch.Tensor:
    """Return a boolean local-maximum mask matching ``scipy.signal.find_peaks`` closely.

    ``scipy.signal.find_peaks``'s ``distance`` is a *minimum required* separation: two local
    maxima exactly ``distance`` bins apart are both kept, only strictly-closer pairs are
    suppressed (the taller survives). This is therefore two distinct stages, not one fused
    window: (1) genuine local-maximum detection always compares only to the immediate neighbor
    on each side (radius 1, independent of ``distance``); (2) an additional suppression pass
    then keeps only candidates that are the tallest *among already-detected candidates* within
    radius ``distance - 1`` (comparing candidate heights via the same "heatmap peak extraction"
    max-pool trick, not the raw signal — the raw signal's non-candidate points must not
    interfere with this comparison). Interior detection is corrected at the array boundaries to
    match ``scipy.signal.find_peaks`` (which never reports edge indices on its own), re-adding
    them via the same rule as the NumPy reference (``peak_region.py`` lines 134-137: threshold
    plus a single-neighbor comparison).

    Exact-tie plateaus emit one deterministic center. Odd plateaus use their middle
    position; even plateaus use the lower of the two central indices.

    REMARK: stage (2)'s single-pass suppression is the standard, widely-used "windowed-max"
    approximation of true greedy tallest-first NMS (the same construction used for CenterNet-
    style heatmap peak extraction) — it can diverge from scipy's exact greedy result for chains
    of 3+ mutually-conflicting candidates spanning more than one suppression window (e.g. A-B-C
    where A and C are each within ``distance`` of B but not of each other: scipy's greedy keeps
    both A and C once the weaker B is removed, since it only re-checks distance against already-
    *kept* peaks, while this single pass compares every candidate against the original,
    un-pruned candidate set and can under-select). Empirically exact for ``min_peak_distance``
    of 1-2 (the common case, including the default); divergence only appears for larger
    ``min_peak_distance`` on densely-packed candidates, and is bounded by the invariant that no
    two surviving peaks are ever closer than ``distance`` (never over-selects).

    :param values: Dense spectrum intensities, already cleaned (finite, non-negative).
    :type values: torch.Tensor
    :param min_distance: Minimum peak separation in grid bins (``distance`` in
        ``scipy.signal.find_peaks``); non-positive values are treated as 1.
    :type min_distance: int
    :param threshold: Per-spectrum scalar height threshold, broadcastable to ``[B, 1]``.
    :type threshold: torch.Tensor | float
    :return: Boolean local-maximum mask with the same shape as ``values``.
    :rtype: torch.Tensor
    """
    depth = values.shape[-1]  # F
    distance = max(1, int(min_distance))

    threshold_row = threshold if isinstance(threshold, torch.Tensor) else torch.as_tensor(threshold, device=values.device, dtype=values.dtype)
    ## Normalize threshold to a [B, 1]-broadcastable form and a [B]/scalar flat form
    threshold_broadcast = threshold_row.view(-1, 1) if threshold_row.ndim >= 1 else threshold_row
    threshold_flat = threshold_row.view(-1) if threshold_row.ndim >= 1 else threshold_row

    ## Stage 1: local maximum vs. immediate neighbors only (radius 1, independent of distance)
    interior_pooled = torch_functional.max_pool1d(values.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)  # (B, F)
    is_peak = (values == interior_pooled) & (values > threshold_broadcast)
    if depth == 1:
        is_peak[:, 0] = values[:, 0] > threshold_flat

    ## Collapse every contiguous equal-height plateau to one deterministic center
    if depth > 1 and values.shape[0] > 0:
        previous_is_same_peak = torch.zeros_like(is_peak)
        previous_is_same_peak[:, 1:] = is_peak[:, :-1] & (values[:, 1:] == values[:, :-1])
        run_start = is_peak & ~previous_is_same_peak
        run_id = run_start.cumsum(dim=1) - 1  # (B, F)
        positions = torch.arange(depth, device=values.device).unsqueeze(0).expand_as(run_id)  # (B, F)
        group_key = torch.arange(values.shape[0], device=values.device).unsqueeze(1) * depth + run_id.clamp_min(0)  # (B, F)
        flat_size = values.shape[0] * depth
        starts = torch.full((flat_size,), depth, dtype=torch.long, device=values.device)
        ends = torch.full((flat_size,), -1, dtype=torch.long, device=values.device)
        valid_keys = group_key[is_peak]
        valid_positions = positions[is_peak]
        starts.scatter_reduce_(0, valid_keys, valid_positions, reduce="amin", include_self=True)
        ends.scatter_reduce_(0, valid_keys, valid_positions, reduce="amax", include_self=True)
        centers = torch.div(starts + ends, 2, rounding_mode="floor")
        valid_groups = ends >= 0
        centered = torch.zeros_like(is_peak)
        center_rows = torch.div(torch.arange(flat_size, device=values.device)[valid_groups], depth, rounding_mode="floor")
        centered[center_rows, centers[valid_groups]] = True
        is_peak = centered

    ## Stage 2: suppress candidates not locally dominant among candidates within distance - 1
    if distance > 1 and depth > 1:
        suppression_radius = distance - 1
        suppression_kernel = 2 * suppression_radius + 1
        candidate_source = torch.where(is_peak, values, torch.full_like(values, float("-inf")))
        suppressed_pooled = torch_functional.max_pool1d(candidate_source.unsqueeze(1), kernel_size=suppression_kernel, stride=1, padding=suppression_radius).squeeze(1)
        is_peak = is_peak & (values >= suppressed_pooled)

    return is_peak


## Direction-specific stop masks matching the sequential region-growth loop exactly
def region_stop_masks(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two directional stop masks used by valley-bounded region growth.

    A naive *symmetric* "local minimum" mask (both neighbors ``>=``) is wrong here: it flags
    every position of a flat run (e.g. a long run of exact-zero background — the common case in
    binned MSI spectra, not a rare edge case) as a stop point, so a nearest-flagged search would
    halt at the *first* zero instead of growing through the whole flat run the way the reference
    ``while`` loop does (its continue-condition, e.g. ``values[right+1] <= values[right]``, is
    non-strict and keeps walking through ties). The correct per-direction stop condition is
    strict: growing right stops only at ``i == F - 1`` or where the *next* value is strictly
    greater (``values[i+1] > values[i]``); growing left mirrors this on the other side. This
    reproduces ``peak_region.py:_region``'s ``"valley_boundaries"`` walk exactly for spectra
    without adjacent ties in the crossed comparison itself.

    :param values: Dense spectrum intensities, already cleaned.
    :type values: torch.Tensor
    :return: ``(left_stop_mask, right_stop_mask)``, each with the same shape as ``values``.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    left_stop = torch.ones_like(values, dtype=torch.bool)
    right_stop = torch.ones_like(values, dtype=torch.bool)
    left_stop[:, 1:] = values[:, :-1] > values[:, 1:]  # values[i-1] > values[i]
    right_stop[:, :-1] = values[:, 1:] > values[:, :-1]  # values[i+1] > values[i]
    return left_stop, right_stop


# Nearest-flagged-position search
## Shared cummax/cummin primitive for "nearest True position" lookups
def nearest_flagged(flags: torch.Tensor, direction: Literal["left", "right"], default: int) -> torch.Tensor:
    """Return, for every position, the nearest (inclusive) flagged position in one direction.

    Used both for locating the nearest valley bounding a peak's envelope and for locating the
    nearest surviving NMS center owning a grid position in ``TopPeaksInverseBinner``.

    :param flags: Boolean mask over the last dimension.
    :type flags: torch.Tensor
    :param direction: ``"left"`` searches positions at or before each index; ``"right"``
        searches positions at or after each index.
    :type direction: Literal["left", "right"]
    :param default: Value returned where no flagged position exists in that direction.
    :type default: int
    :return: Integer tensor of nearest flagged positions, same shape as ``flags``.
    :rtype: torch.Tensor
    """
    depth = flags.shape[-1]
    positions = torch.arange(depth, device=flags.device).unsqueeze(0).expand_as(flags)
    if direction == "left":
        candidate = torch.where(flags, positions, torch.full_like(positions, -1))
        nearest = torch.cummax(candidate, dim=-1).values
        return torch.where(nearest < 0, torch.full_like(nearest, default), nearest)
    if direction == "right":
        candidate = torch.where(flags, positions, torch.full_like(positions, depth))
        flipped = candidate.flip(dims=[-1])
        suffix_min = torch.cummin(flipped, dim=-1).values.flip(dims=[-1])
        return torch.where(suffix_min >= depth, torch.full_like(suffix_min, default), suffix_min)
    raise_validation_error("VectorizedInverseBinner", f"Unknown direction '{direction}'.")


# Region-bound construction
## Fixed-radius window around every peak, clamped to the grid
def region_bounds_fixed_window(peak_positions: torch.Tensor, width: int, depth: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``[left, right]`` bounds for a fixed-radius window around every peak.

    :param peak_positions: Padded peak positions, shape ``[B, P_max]``.
    :type peak_positions: torch.Tensor
    :param width: Number of neighboring bins included on each side.
    :type width: int
    :param depth: Number of grid bins (F).
    :type depth: int
    :return: Inclusive ``(left, right)`` bound tensors, each ``[B, P_max]``.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    left = (peak_positions - width).clamp(min=0)
    right = (peak_positions + width).clamp(max=depth - 1)
    return left, right


## Valley-to-valley envelope bounds around every peak
def region_bounds_valley(peak_positions: torch.Tensor, values: torch.Tensor, depth: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``[left, right]`` bounds as the nearest region-growth stop on each side of every peak.

    :param peak_positions: Padded peak positions, shape ``[B, P_max]``.
    :type peak_positions: torch.Tensor
    :param values: Dense, cleaned spectrum intensities, shape ``[B, F]`` (see ``region_stop_masks``).
    :type values: torch.Tensor
    :param depth: Number of grid bins (F).
    :type depth: int
    :return: Inclusive ``(left, right)`` bound tensors, each ``[B, P_max]``.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    left_stop_mask, right_stop_mask = region_stop_masks(values)
    left_bound_per_position = nearest_flagged(left_stop_mask, "left", default=0)  # (B, F)
    right_bound_per_position = nearest_flagged(right_stop_mask, "right", default=depth - 1)  # (B, F)
    safe_positions = peak_positions.clamp(0, depth - 1)
    left = left_bound_per_position.gather(1, safe_positions)
    right = right_bound_per_position.gather(1, safe_positions)
    return left, right


## Relative-height threshold-crossing bounds around every peak
def region_bounds_relative_height(peak_positions: torch.Tensor, values: torch.Tensor, relative_height: float, depth: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``[left, right]`` bounds where the signal first drops below a per-peak threshold.

    Each peak has its own boundary (``relative_height * values[peak]``), so this cannot reuse a
    single shared position-level mask across peaks (unlike ``region_bounds_valley``) — it
    broadcasts over an explicit peak axis instead.

    :param peak_positions: Padded peak positions, shape ``[B, P_max]``.
    :type peak_positions: torch.Tensor
    :param values: Dense, cleaned spectrum intensities, shape ``[B, F]``.
    :type values: torch.Tensor
    :param relative_height: Fraction of peak height defining the boundary.
    :type relative_height: float
    :param depth: Number of grid bins (F).
    :type depth: int
    :return: Inclusive ``(left, right)`` bound tensors, each ``[B, P_max]``.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    device = values.device
    safe_positions = peak_positions.clamp(0, depth - 1)
    heights = values.gather(1, safe_positions)  # (B, P_max)
    boundary = relative_height * heights  # (B, P_max)
    crossed = values.unsqueeze(1) < boundary.unsqueeze(2)  # (B, P_max, F) -- below this peak's own boundary
    positions = torch.arange(depth, device=device).view(1, 1, depth).expand_as(crossed)
    peak_index = safe_positions.unsqueeze(-1)  # (B, P_max, 1)

    ## Left bound: one past the nearest crossing strictly before the peak, else 0
    left_candidate = torch.where(crossed, positions, torch.full_like(positions, -1))
    left_running_max = torch.cummax(left_candidate, dim=-1).values
    before_peak_index = (peak_index - 1).clamp(min=0)
    last_crossed_before_peak = left_running_max.gather(-1, before_peak_index).squeeze(-1)
    left = torch.where(last_crossed_before_peak >= 0, last_crossed_before_peak + 1, torch.zeros_like(last_crossed_before_peak))
    left = torch.where(safe_positions == 0, torch.zeros_like(left), left)

    ## Right bound: one before the nearest crossing strictly after the peak, else depth - 1
    right_candidate = torch.where(crossed, positions, torch.full_like(positions, depth))
    flipped = right_candidate.flip(dims=[-1])
    right_running_min = torch.cummin(flipped, dim=-1).values.flip(dims=[-1])
    after_peak_index = (peak_index + 1).clamp(max=depth - 1)
    first_crossed_after_peak = right_running_min.gather(-1, after_peak_index).squeeze(-1)
    right = torch.where(first_crossed_after_peak < depth, first_crossed_after_peak - 1, torch.full_like(first_crossed_after_peak, depth - 1))
    right = torch.where(safe_positions == depth - 1, torch.full_like(right, depth - 1), right)

    return left, right


# Envelope union
## Sweep-line difference-array coverage union — replaces per-position ownership attribution
def union_of_intervals_mask(left: torch.Tensor, right: torch.Tensor, peak_valid_mask: torch.Tensor, depth: int) -> torch.Tensor:
    """Return the boolean union of every valid peak's ``[left, right]`` interval.

    Computes coverage via a sweep-line difference array (``+1`` at each interval start, ``-1``
    just past each interval end, then a cumulative sum) rather than resolving per-position
    ownership: since only ``"keep_region"``/``"maximum"`` reductions are supported by the
    batched path, coverage — not attribution — is all that is ever needed, at ``O(B * P_max)``
    scatter plus ``O(B * F)`` cumsum, exact for any number of overlapping intervals.

    :param left: Inclusive interval starts, shape ``[B, P_max]``.
    :type left: torch.Tensor
    :param right: Inclusive interval ends, shape ``[B, P_max]``.
    :type right: torch.Tensor
    :param peak_valid_mask: Validity mask selecting which padded columns are real peaks.
    :type peak_valid_mask: torch.Tensor
    :param depth: Number of grid bins (F).
    :type depth: int
    :return: Boolean coverage mask, shape ``[B, F]``.
    :rtype: torch.Tensor
    """
    batch_size = left.shape[0]
    diff = torch.zeros((batch_size, depth + 1), device=left.device, dtype=torch.long)
    starts = left.clamp(0, depth)
    ends = (right + 1).clamp(0, depth)
    increments = peak_valid_mask.long()  # REMARK: invalid columns contribute a 0 increment,
    diff.scatter_add_(1, starts, increments)  # so their (possibly arbitrary) padded start/end
    diff.scatter_add_(1, ends, -increments)  # positions are harmless no-ops regardless of value
    coverage = diff.cumsum(dim=1)[:, :depth]
    return coverage > 0
