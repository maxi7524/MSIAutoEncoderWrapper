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
    """Keep the smallest high-weight set reaching a requested mass fraction.

    Candidates are ranked by the configured mass weights. Returned coordinates retain
    their original grid order, and returned intensities are not replaced by weights.
    """

    # Strategy configuration
    ## Initialize mass weighting, selection limits, and diagnostics
    def __init__(self, binner: Optional[MSIBaseBinner] = None, retained_fraction: float = 0.95, min_bins: int = 0, max_bins: int | None = None, mass_strategy: str | Callable[..., Any] = "intensity", mass_options: Mapping[str, Any] | None = None, track_diagnostics: bool = False, active_context: Optional[Any] = None) -> None:
        """Configure cumulative-mass inverse binning.

        :param binner: Forward binner defining the shared m/z axis.
        :type binner: Optional[MSIBaseBinner]
        :param retained_fraction: Fraction of candidate mass the selected set should
            retain.
        :type retained_fraction: float
        :param min_bins: Minimum number of candidates to emit when available.
        :type min_bins: int
        :param max_bins: Optional maximum number of emitted candidates.
        :type max_bins: int | None
        :param mass_strategy: ``"intensity"``, ``"normalized_intensity"``,
            ``"squared_intensity"``, or a callable producing selection weights.
        :type mass_strategy: str | Callable[..., Any]
        :param mass_options: Keyword arguments passed to a custom mass strategy.
        :type mass_options: Mapping[str, Any] | None
        :param track_diagnostics: Whether calls populate ``last_diagnostics``.
        :type track_diagnostics: bool
        :param active_context: Optional context used to resolve ``binner``.
        :type active_context: Optional[Any]
        :raises ValidationError: If the retained fraction or selection limits are
            invalid.
        """
        super().__init__(binner=binner, active_context=active_context)
        self.retained_fraction = float(retained_fraction)
        self.min_bins, self.max_bins = validate_budget(min_bins, max_bins)
        self.track_diagnostics = bool(track_diagnostics)
        if not 0.0 < self.retained_fraction <= 1.0:
            from ...utils.exceptions import raise_validation_error
            raise_validation_error("CumulativeMassInverseBinner", "retained_fraction must belong to (0, 1].")
        self.mass_strategy, self.mass_options = mass_strategy, dict(mass_options or {})
        name = mass_strategy if isinstance(mass_strategy, str) else "custom"
        self._config = {"retained_fraction": self.retained_fraction, "min_bins": self.min_bins, "max_bins": self.max_bins, "mass_strategy": name, "mass_options": self.mass_options, "track_diagnostics": self.track_diagnostics}
        self.last_diagnostics: dict[str, Any] = {}

    # Mass weighting
    ## Resolve non-negative weights used only for candidate ranking
    def _weights(self, values: np.ndarray) -> np.ndarray:
        """Return validated selection weights for one dense spectrum.

        :param values: Validated dense spectrum intensities.
        :type values: numpy.ndarray
        :return: Finite non-negative weights matching ``values``.
        :rtype: numpy.ndarray
        :raises ValidationError: If the strategy is unknown or returns invalid weights.
        """
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

    # Spectrum reconstruction
    ## Select the smallest deterministic prefix satisfying the mass target
    def __call__(self, grid_ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Select grid points according to cumulative configured mass.

        :param grid_ys: Dense intensities aligned with the forward binner axis.
        :type grid_ys: numpy.ndarray
        :return: Selected m/z coordinates and their original intensities.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        :raises ValidationError: If the input or mass weights are invalid.
        """
        ### Validate input and compute ranking weights
        values, axis = validate_input(grid_ys, self._Binner.GetXAxis())
        weights = self._weights(values)

        ### Select finite positive candidates within the configured budget
        candidates = np.flatnonzero(valid_candidate_mask(values) & (weights > 0.0))
        selected, reached, fraction = select_by_cumulative_mass(candidates, weights, self.retained_fraction, self.min_bins, self.max_bins)

        ### Restore physical axis order before constructing sparse output
        selected = np.sort(selected)

        ### Record optional selection diagnostics without changing the output
        if self.track_diagnostics:
            self.last_diagnostics = {**base_diagnostics(values, selected, selected.size, weights), "target_fraction": self.retained_fraction, "target_fraction_reached": reached, "retained_mass_fraction": fraction, "threshold_value": None}
        else:
            self.last_diagnostics = {}
        return axis[selected], values[selected]

    # Diagnostics interface
    ## Return diagnostics without exposing mutable internal state
    def get_last_diagnostics(self) -> dict[str, Any]:
        """Return an independent copy of diagnostics produced by the last call.

        :return: Selection diagnostics, or an empty mapping when tracking is disabled.
        :rtype: dict[str, Any]
        """
        return dict(self.last_diagnostics)
