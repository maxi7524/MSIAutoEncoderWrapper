"""Cumulative-mass inverse binning."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Optional

import numpy as np

from ..base_binner import MSIBaseBinner
from ..base_inverse import MSIBaseInverseBinner
from ..binners_manager import BinnerManager
from .selection_utils import base_diagnostics, select_by_cumulative_mass, valid_candidate_mask, validate_budget, validate_input


@BinnerManager.register_inverse_binner("CumulativeMassInverseBinner")
class CumulativeMassInverseBinner(MSIBaseInverseBinner):
    """Keep the smallest high-weight set reaching a requested mass fraction."""

    def __init__(self, binner: Optional[MSIBaseBinner] = None, retained_fraction: float = 0.95, min_bins: int = 0, max_bins: int | None = None, mass_strategy: str | Callable[..., Any] = "intensity", mass_options: Mapping[str, Any] | None = None, active_context: Optional[Any] = None) -> None:
        super().__init__(binner=binner, active_context=active_context)
        self.retained_fraction = float(retained_fraction)
        self.min_bins, self.max_bins = validate_budget(min_bins, max_bins)
        if not 0.0 < self.retained_fraction <= 1.0:
            from ...utils.exceptions import raise_validation_error
            raise_validation_error("CumulativeMassInverseBinner", "retained_fraction must belong to (0, 1].")
        self.mass_strategy, self.mass_options = mass_strategy, dict(mass_options or {})
        name = mass_strategy if isinstance(mass_strategy, str) else "custom"
        self._config = {"retained_fraction": self.retained_fraction, "min_bins": self.min_bins, "max_bins": self.max_bins, "mass_strategy": name, "mass_options": self.mass_options}
        self.last_diagnostics: dict[str, Any] = {}

    def _weights(self, values: np.ndarray) -> np.ndarray:
        if callable(self.mass_strategy):
            weights = self.mass_strategy(values, **self.mass_options)
        elif self.mass_strategy in {"intensity", "normalized_intensity"}:
            weights = np.clip(values, 0.0, None)
            if self.mass_strategy == "normalized_intensity" and np.sum(weights) > 0:
                weights = weights / np.sum(weights)
        elif self.mass_strategy == "squared_intensity":
            weights = np.clip(values, 0.0, None) ** 2
        else:
            from ...utils.exceptions import raise_validation_error
            raise_validation_error("CumulativeMassInverseBinner", f"Unknown mass strategy '{self.mass_strategy}'.")
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != values.shape or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            from ...utils.exceptions import raise_validation_error
            raise_validation_error("CumulativeMassInverseBinner", "Mass weights must be finite, non-negative, and match the spectrum.")
        return weights

    def __call__(self, grid_ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values, axis = validate_input(grid_ys, self._Binner.GetXAxis())
        weights = self._weights(values)
        candidates = np.flatnonzero(valid_candidate_mask(values) & (weights > 0.0))
        selected, reached, fraction = select_by_cumulative_mass(candidates, weights, self.retained_fraction, self.min_bins, self.max_bins)
        selected = np.sort(selected)
        self.last_diagnostics = {**base_diagnostics(values, selected, selected.size, weights), "target_fraction": self.retained_fraction, "target_fraction_reached": reached, "retained_mass_fraction": fraction, "threshold_value": None}
        return axis[selected], values[selected]

    def get_last_diagnostics(self) -> dict[str, Any]:
        """Return an independent copy of diagnostics produced by the last call."""
        return dict(self.last_diagnostics)
