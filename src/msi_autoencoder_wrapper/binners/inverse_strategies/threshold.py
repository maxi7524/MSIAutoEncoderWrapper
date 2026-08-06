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
    """Keep finite positive grid points above a configurable threshold.

    The strategy preserves coordinates and intensities from the forward grid. When
    ``max_bins`` limits the result, candidates are ranked by descending intensity
    with the grid index used as the deterministic tie breaker.
    """

    # Strategy configuration
    ## Initialize threshold resolution, selection budget, and diagnostics
    def __init__(self, binner: Optional[MSIBaseBinner] = None, threshold_strategy: str | Callable[..., Any] = "absolute", threshold_options: Mapping[str, Any] | None = None, absolute_minimum: float | None = None, max_bins: int | None = None, track_diagnostics: bool = False, active_context: Optional[Any] = None) -> None:
        """Configure threshold-based inverse binning.

        :param binner: Forward binner defining the shared m/z axis.
        :type binner: Optional[MSIBaseBinner]
        :param threshold_strategy: Registered threshold name or callable returning a
            scalar or one threshold per grid position.
        :type threshold_strategy: str | Callable[..., Any]
        :param threshold_options: Keyword arguments passed to the threshold strategy.
        :type threshold_options: Mapping[str, Any] | None
        :param absolute_minimum: Optional lower bound applied after resolving the
            strategy threshold.
        :type absolute_minimum: float | None
        :param max_bins: Optional maximum number of emitted grid points.
        :type max_bins: int | None
        :param track_diagnostics: Whether calls populate ``last_diagnostics``.
        :type track_diagnostics: bool
        :param active_context: Optional context used to resolve ``binner``.
        :type active_context: Optional[Any]
        :raises ValidationError: If ``max_bins`` is not positive.
        """
        super().__init__(binner=binner, active_context=active_context)
        self.threshold_strategy = threshold_strategy
        self.threshold_options = dict(threshold_options or {})
        self.absolute_minimum = absolute_minimum
        self.max_bins = None if max_bins is None else int(max_bins)
        self.track_diagnostics = bool(track_diagnostics)
        if self.max_bins is not None and self.max_bins <= 0:
            from ...utils.exceptions import raise_validation_error
            raise_validation_error("ThresholdInverseBinner", "max_bins must be positive.")
        strategy_name = threshold_strategy if isinstance(threshold_strategy, str) else "callable"
        self._config = {"threshold_strategy": strategy_name, "threshold_options": self.threshold_options, "absolute_minimum": absolute_minimum, "max_bins": max_bins, "track_diagnostics": self.track_diagnostics}
        if callable(threshold_strategy):
            self._config["callable_name"] = getattr(threshold_strategy, "__qualname__", None)
            self._config["callable_serializable"] = False
        self.last_diagnostics: dict[str, Any] = {}

    # Spectrum reconstruction
    ## Filter and optionally rank one dense spectrum
    def __call__(self, grid_ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return grid points that pass the resolved threshold.

        :param grid_ys: Dense intensities aligned with the forward binner axis.
        :type grid_ys: numpy.ndarray
        :return: Selected m/z coordinates and their unchanged intensities.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        :raises ValidationError: If the input shape or threshold result is invalid.
        """
        ### Validate the dense spectrum and resolve its threshold
        values, axis = validate_input(grid_ys, self._Binner.GetXAxis())
        threshold = resolve_threshold(values, self.threshold_strategy, self.threshold_options)
        if self.absolute_minimum is not None:
            threshold = np.maximum(threshold, float(self.absolute_minimum))
        ### Select finite positive candidates above the effective threshold
        indices = np.flatnonzero(valid_candidate_mask(values) & (values > threshold))

        ### Enforce the output budget while restoring physical axis order
        if self.max_bins is not None and indices.size > self.max_bins:
            indices = np.sort(rank_candidates(indices, values)[: self.max_bins])

        ### Record optional selection diagnostics without changing the output
        if self.track_diagnostics:
            self.last_diagnostics = {**base_diagnostics(values, indices, indices.size), "threshold_value": threshold, "target_fraction": None, "target_fraction_reached": None}
        else:
            self.last_diagnostics = {}
        return axis[indices], values[indices]

    # Diagnostics interface
    ## Return diagnostics without exposing mutable internal state
    def get_last_diagnostics(self) -> dict[str, Any]:
        """Return an independent copy of diagnostics produced by the last call.

        :return: Selection diagnostics, or an empty mapping when tracking is disabled.
        :rtype: dict[str, Any]
        """
        return dict(self.last_diagnostics)
