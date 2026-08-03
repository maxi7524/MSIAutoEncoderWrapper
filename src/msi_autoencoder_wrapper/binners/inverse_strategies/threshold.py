"""Threshold-based inverse binning."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Optional

import numpy as np

from ..base_binner import MSIBaseBinner
from ..base_inverse import MSIBaseInverseBinner
from ..binners_manager import BinnerManager
from .selection_utils import base_diagnostics, rank_candidates, resolve_threshold, valid_candidate_mask, validate_input


@BinnerManager.register_inverse_binner("ThresholdInverseBinner")
class ThresholdInverseBinner(MSIBaseInverseBinner):
    """Keep finite positive grid points strictly above a spectrum-dependent threshold."""

    def __init__(self, binner: Optional[MSIBaseBinner] = None, threshold_strategy: str | Callable[..., Any] = "absolute", threshold_options: Mapping[str, Any] | None = None, absolute_minimum: float | None = None, max_bins: int | None = None, active_context: Optional[Any] = None) -> None:
        super().__init__(binner=binner, active_context=active_context)
        self.threshold_strategy = threshold_strategy
        self.threshold_options = dict(threshold_options or {})
        self.absolute_minimum = absolute_minimum
        self.max_bins = None if max_bins is None else int(max_bins)
        if self.max_bins is not None and self.max_bins <= 0:
            from ...utils.exceptions import raise_validation_error
            raise_validation_error("ThresholdInverseBinner", "max_bins must be positive.")
        strategy_name = threshold_strategy if isinstance(threshold_strategy, str) else "callable"
        self._config = {"threshold_strategy": strategy_name, "threshold_options": self.threshold_options, "absolute_minimum": absolute_minimum, "max_bins": max_bins}
        if callable(threshold_strategy):
            self._config["callable_name"] = getattr(threshold_strategy, "__qualname__", None)
            self._config["callable_serializable"] = False
        self.last_diagnostics: dict[str, Any] = {}

    def __call__(self, grid_ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values, axis = validate_input(grid_ys, self._Binner.GetXAxis())
        threshold = resolve_threshold(values, self.threshold_strategy, self.threshold_options)
        if self.absolute_minimum is not None:
            threshold = np.maximum(threshold, float(self.absolute_minimum))
        indices = np.flatnonzero(valid_candidate_mask(values) & (values > threshold))
        if self.max_bins is not None and indices.size > self.max_bins:
            indices = np.sort(rank_candidates(indices, values)[: self.max_bins])
        self.last_diagnostics = {**base_diagnostics(values, indices, indices.size), "threshold_value": threshold, "target_fraction": None, "target_fraction_reached": None}
        return axis[indices], values[indices]

    def get_last_diagnostics(self) -> dict[str, Any]:
        """Return an independent copy of diagnostics produced by the last call."""
        return dict(self.last_diagnostics)
