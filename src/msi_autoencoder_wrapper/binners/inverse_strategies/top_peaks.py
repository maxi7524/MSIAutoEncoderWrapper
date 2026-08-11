"""Vectorized peak and peak-neighbourhood inverse binners."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import torch

from ..base_binner import MSIBaseBinner
from ..base_inverse import MSIBaseInverseBinner
from ..binners_manager import BinnerManager
from ...data import InverseSpectrumBatch, SpectrumBatch
from ...utils.exceptions import raise_validation_error
from . import peak_ops
from .reconstruction import pack_point_projection, pack_region_projection


class _PeakSelectionBase(MSIBaseInverseBinner):
    """Share vectorized local-maximum detection and optional peak limiting."""

    def __init__(
        self,
        binner: Optional[MSIBaseBinner] = None,
        max_peaks: int | None = None,
        min_peak_distance: int = 1,
        reconstruction_mass_axis: torch.Tensor | None = None,
        active_context: Optional[Any] = None,
    ) -> None:
        super().__init__(binner, reconstruction_mass_axis, active_context)
        self.max_peaks = None if max_peaks is None else int(max_peaks)
        self.min_peak_distance = int(min_peak_distance)
        if self.max_peaks is not None and self.max_peaks <= 0:
            raise_validation_error(type(self).__name__, "max_peaks must be positive or None.")
        if self.min_peak_distance < 1:
            raise_validation_error(type(self).__name__, "min_peak_distance must be positive.")

    def _peak_mask(self, values: torch.Tensor) -> torch.Tensor:
        """Return all or the strongest limited local maxima for every row."""
        cleaned = torch.where(torch.isfinite(values) & (values > 0), values, torch.zeros_like(values))  # (B, F)
        detected = peak_ops.local_maxima_mask(cleaned, self.min_peak_distance, 0.0)  # (B, F)
        if self.max_peaks is None or self.max_peaks >= values.shape[1]:
            return detected
        ranked_values = torch.where(detected, cleaned, torch.full_like(cleaned, float("-inf")))
        indices = torch.topk(ranked_values, k=self.max_peaks, dim=1).indices  # (B, K)
        valid_count = detected.sum(dim=1).clamp(max=self.max_peaks)  # (B,)
        valid = torch.arange(self.max_peaks, device=values.device).unsqueeze(0) < valid_count.unsqueeze(1)  # (B, K)
        selected = torch.zeros_like(detected)
        selected.scatter_(1, indices, valid)
        return selected


@BinnerManager.register_inverse_binner("TopPeaksInverseBinner")
class TopPeaksInverseBinner(_PeakSelectionBase):
    """Project detected local maxima onto a shared reconstruction axis."""

    def __init__(
        self,
        binner: Optional[MSIBaseBinner] = None,
        max_peaks: int | None = None,
        min_peak_distance: int = 1,
        reconstruction_mass_axis: torch.Tensor | None = None,
        active_context: Optional[Any] = None,
    ) -> None:
        super().__init__(binner, max_peaks, min_peak_distance, reconstruction_mass_axis, active_context)
        self._config = {"max_peaks": self.max_peaks, "min_peak_distance": self.min_peak_distance}

    def transform(self, batch: SpectrumBatch) -> InverseSpectrumBatch:
        """Detect and project local maxima for the complete batch."""
        return pack_point_projection(batch, self._peak_mask(batch.spectra), self.reconstruction_mass_axis)


@BinnerManager.register_inverse_binner("TopPeaksNeighbourhoodInverseBinner")
class TopPeaksNeighbourhoodInverseBinner(_PeakSelectionBase):
    """Project the union of regions surrounding selected local maxima."""

    def __init__(
        self,
        binner: Optional[MSIBaseBinner] = None,
        max_peaks: int | None = None,
        min_peak_distance: int = 1,
        region_strategy: str = "fixed_window",
        region_options: Mapping[str, Any] | None = None,
        reconstruction_mass_axis: torch.Tensor | None = None,
        active_context: Optional[Any] = None,
    ) -> None:
        super().__init__(binner, max_peaks, min_peak_distance, reconstruction_mass_axis, active_context)
        if region_strategy not in {"fixed_window", "valley_boundaries", "relative_height"}:
            raise_validation_error(type(self).__name__, "Unknown region_strategy.")
        self.region_strategy = region_strategy
        self.region_options = dict(region_options or {})
        if region_strategy == "fixed_window" and int(self.region_options.get("window_size", 1)) < 0:
            raise_validation_error(type(self).__name__, "window_size must be non-negative.")
        self._config = {
            "max_peaks": self.max_peaks,
            "min_peak_distance": self.min_peak_distance,
            "region_strategy": self.region_strategy,
            "region_options": self.region_options,
        }

    def transform(self, batch: SpectrumBatch) -> InverseSpectrumBatch:
        """Select peak envelopes and expand them over the reconstruction axis."""
        values = torch.where(torch.isfinite(batch.spectra) & (batch.spectra > 0), batch.spectra, torch.zeros_like(batch.spectra))  # (B, F)
        peak_mask = self._peak_mask(values)  # (B, F)
        peak_count = peak_mask.sum(dim=1)  # (B,)
        max_count = int(peak_count.max().item()) if peak_count.numel() else 0
        rank_values = torch.where(peak_mask, values, torch.full_like(values, float("-inf")))
        positions = torch.sort(rank_values, dim=1, descending=True, stable=True).indices[:, :max_count]  # (B, P)
        valid = torch.arange(max_count, device=values.device).unsqueeze(0) < peak_count.unsqueeze(1)  # (B, P)
        depth = values.shape[1]
        if self.region_strategy == "fixed_window":
            left, right = peak_ops.region_bounds_fixed_window(positions, int(self.region_options.get("window_size", 1)), depth)
        elif self.region_strategy == "valley_boundaries":
            left, right = peak_ops.region_bounds_valley(positions, values, depth)
        else:
            left, right = peak_ops.region_bounds_relative_height(positions, values, float(self.region_options.get("relative_height", 0.5)), depth)
        keep_mask = peak_ops.union_of_intervals_mask(left, right, valid, depth)  # (B, F)
        return pack_region_projection(batch, keep_mask, self.reconstruction_mass_axis)
