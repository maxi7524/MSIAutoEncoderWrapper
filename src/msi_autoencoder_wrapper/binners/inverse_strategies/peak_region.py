"""Local-maximum and peak-region inverse binning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
from scipy.signal import find_peaks

from ..base_binner import MSIBaseBinner
from ..base_inverse import MSIBaseInverseBinner
from ..binners_manager import BinnerManager
from .selection_utils import base_diagnostics, rank_candidates, resolve_threshold, valid_candidate_mask, validate_budget, validate_input


@BinnerManager.register_inverse_binner("PeakRegionInverseBinner")
class PeakRegionInverseBinner(MSIBaseInverseBinner):
    """Select local maxima and reduce non-overlapping peak regions.

    Overlaps are assigned pointwise to the stronger maximum; equal-height peaks
    use the lower grid index. Consequently every input bin contributes at most once.
    """

    def __init__(self, binner: Optional[MSIBaseBinner] = None, peak_threshold_strategy: str = "absolute", peak_threshold_options: Mapping[str, Any] | None = None, region_strategy: str = "fixed_window", region_options: Mapping[str, Any] | None = None, reduction_strategy: str = "centroid", min_peaks: int = 0, max_peaks: int | None = None, min_peak_distance: int = 0, retained_fraction: float | None = None, active_context: Optional[Any] = None) -> None:
        super().__init__(binner=binner, active_context=active_context)
        self.peak_threshold_strategy, self.peak_threshold_options = peak_threshold_strategy, dict(peak_threshold_options or {})
        self.region_strategy, self.region_options, self.reduction_strategy = region_strategy, dict(region_options or {}), reduction_strategy
        self.min_peaks, self.max_peaks = validate_budget(min_peaks, max_peaks)
        self.min_peak_distance = int(min_peak_distance)
        self.retained_fraction = retained_fraction
        if self.min_peak_distance < 0 or (retained_fraction is not None and not 0 < retained_fraction <= 1):
            from ...utils.exceptions import raise_validation_error
            raise_validation_error("PeakRegionInverseBinner", "Distance must be non-negative and retained_fraction must belong to (0, 1].")
        if region_strategy not in {"fixed_window", "valley_boundaries", "relative_height"} or reduction_strategy not in {"keep_region", "maximum", "centroid"}:
            from ...utils.exceptions import raise_validation_error
            raise_validation_error("PeakRegionInverseBinner", "Unknown region or reduction strategy.")
        self._config = {"peak_threshold_strategy": peak_threshold_strategy, "peak_threshold_options": self.peak_threshold_options, "region_strategy": region_strategy, "region_options": self.region_options, "reduction_strategy": reduction_strategy, "min_peaks": self.min_peaks, "max_peaks": self.max_peaks, "min_peak_distance": self.min_peak_distance, "retained_fraction": retained_fraction}
        self.last_diagnostics: dict[str, Any] = {}

    def _region(self, peak: int, values: np.ndarray) -> np.ndarray:
        if self.region_strategy == "fixed_window":
            width = int(self.region_options.get("window_size", 1))
            if width < 0:
                from ...utils.exceptions import raise_validation_error
                raise_validation_error("PeakRegionInverseBinner", "window_size must be non-negative.")
            return np.arange(max(0, peak - width), min(values.size, peak + width + 1))
        left = right = peak
        if self.region_strategy == "relative_height":
            boundary = float(self.region_options.get("relative_height", 0.5)) * values[peak]
            while left > 0 and values[left - 1] >= boundary: left -= 1
            while right + 1 < values.size and values[right + 1] >= boundary: right += 1
        else:
            while left > 0 and values[left - 1] <= values[left]: left -= 1
            while right + 1 < values.size and values[right + 1] <= values[right]: right += 1
        return np.arange(left, right + 1)

    def __call__(self, grid_ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values, axis = validate_input(grid_ys, self._Binner.GetXAxis())
        cleaned = np.where(valid_candidate_mask(values), values, 0.0)
        threshold = resolve_threshold(cleaned, self.peak_threshold_strategy, self.peak_threshold_options)
        if np.asarray(threshold).ndim:
            eligible = cleaned > threshold
            peak_source = np.where(eligible, cleaned, 0.0)
            minimum_height = None
        else:
            peak_source, minimum_height = cleaned, float(threshold)
        peaks, _ = find_peaks(peak_source, height=minimum_height, distance=max(1, self.min_peak_distance))
        if np.asarray(threshold).ndim:
            edge_peaks = [index for index in (0, values.size - 1) if values.size and cleaned[index] > np.asarray(threshold)[index]]
        else:
            edge_peaks = [index for index in (0, values.size - 1) if values.size and cleaned[index] > float(threshold) and (values.size == 1 or cleaned[index] >= cleaned[1 if index == 0 else -2])]
        peaks = np.unique(np.concatenate((peaks, np.asarray(edge_peaks, dtype=int))))
        ranked = rank_candidates(peaks, cleaned)
        target_count = ranked.size if self.max_peaks is None else min(ranked.size, self.max_peaks)
        if self.retained_fraction is not None and ranked.size:
            cumulative = np.cumsum(cleaned[ranked]); total = float(np.sum(cleaned[ranked]))
            target_count = min(target_count, int(np.searchsorted(cumulative, self.retained_fraction * total) + 1))
        target_count = min(ranked.size, max(self.min_peaks, target_count))
        selected_peaks = ranked[:target_count]
        owners: dict[int, int] = {}
        regions = {int(peak): self._region(int(peak), cleaned) for peak in selected_peaks}
        for peak in selected_peaks:
            for index in regions[int(peak)]:
                current = owners.get(int(index))
                if current is None or (cleaned[peak], -peak) > (cleaned[current], -current): owners[int(index)] = int(peak)
        assigned = {int(peak): np.asarray(sorted(index for index, owner in owners.items() if owner == peak), dtype=int) for peak in selected_peaks}
        output_x: list[float] = []; output_y: list[float] = []
        for peak in np.sort(selected_peaks):
            region = assigned[int(peak)]
            if self.reduction_strategy == "keep_region": output_x.extend(axis[region]); output_y.extend(values[region])
            elif self.reduction_strategy == "maximum": output_x.append(float(axis[peak])); output_y.append(float(values[peak]))
            else:
                mass = float(np.sum(values[region])); output_x.append(float(np.sum(axis[region] * values[region]) / mass)); output_y.append(mass)
        selected_bins = np.asarray(sorted(owners), dtype=int)
        widths = [len(region) for region in assigned.values()]
        self.last_diagnostics = {**base_diagnostics(values, selected_bins, len(output_x)), "threshold_value": threshold, "target_fraction": self.retained_fraction, "target_fraction_reached": None if self.retained_fraction is None else float(np.sum(values[selected_bins]) / np.sum(values[valid_candidate_mask(values)])) >= self.retained_fraction, "detected_peak_count": int(peaks.size), "selected_region_count": int(len(assigned)), "merged_region_count": int(sum(len(region) for region in regions.values()) > len(owners)), "mean_region_width": float(np.mean(widths)) if widths else 0.0, "max_region_width": int(max(widths, default=0))}
        return np.asarray(output_x), np.asarray(output_y)

    def get_last_diagnostics(self) -> dict[str, Any]:
        """Return an independent copy of diagnostics produced by the last call."""
        return dict(self.last_diagnostics)
