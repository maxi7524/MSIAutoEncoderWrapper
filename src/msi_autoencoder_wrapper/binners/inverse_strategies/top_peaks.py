"""Budgeted neighborhood selection around the strongest grid points."""

import numpy as np
from ..base_inverse import MSIBaseInverseBinner
from ..base_binner import MSIBaseBinner
from ..binners_manager import BinnerManager
from .selection_utils import base_diagnostics
from typing import Any, Optional


@BinnerManager.register_inverse_binner("TopPeaksInverseBinner")
class TopPeaksInverseBinner(MSIBaseInverseBinner):
    """Select non-overlapping neighborhoods around the strongest grid points.

    Candidates are processed in descending intensity order. A complete neighborhood
    is accepted only when it fits within ``max_bins``; accepted neighborhoods retain
    their original grid coordinates and intensities.
    """

    # Strategy configuration
    ## Initialize the output budget, neighborhood width, and diagnostics
    def __init__(self, binner: Optional[MSIBaseBinner] = None, max_bins: int = 500, window_size: int = 3, track_diagnostics: bool = False, active_context: Optional[Any] = None) -> None:
        """Configure neighborhood selection around high-intensity bins.

        :param binner: Forward binner defining the shared m/z axis.
        :type binner: Optional[MSIBaseBinner]
        :param max_bins: Maximum total number of grid positions in the output.
        :type max_bins: int
        :param window_size: Number of neighboring bins included on each side of a
            selected center.
        :type window_size: int
        :param track_diagnostics: Whether calls populate ``last_diagnostics``.
        :type track_diagnostics: bool
        :param active_context: Optional context used to resolve ``binner``.
        :type active_context: Optional[Any]
        """
        super().__init__(binner=binner, active_context=active_context)
        self.track_diagnostics = bool(track_diagnostics)
        self._config = {"max_bins": max_bins, "window_size": window_size, "track_diagnostics": self.track_diagnostics}
        self._max_bins = int(max_bins)
        self._window_size = int(window_size)
        self.last_diagnostics: dict[str, Any] = {}

    # Spectrum reconstruction
    ## Grow non-overlapping neighborhoods under the global bin budget
    def __call__(self, grid_ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return accepted peak neighborhoods in physical grid order.

        :param grid_ys: Dense intensities aligned with the forward binner axis.
        :type grid_ys: numpy.ndarray
        :return: Selected m/z coordinates and their unchanged intensities.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        ### Track grid positions already assigned to an accepted neighborhood
        depth = len(grid_ys)
        keep_mask = np.zeros(depth, dtype=bool)

        ### Visit candidate centers from strongest to weakest
        sorted_indices = np.argsort(grid_ys)[::-1]
        count = 0

        ### Accept only complete neighborhoods that fit within the remaining budget
        for idx in sorted_indices:
            if count >= self._max_bins:
                break

            if not keep_mask[idx]:
                start = max(0, idx - self._window_size)
                end = min(depth, idx + self._window_size + 1)

                #### Count only positions not already owned by a stronger neighborhood
                added_points = ~keep_mask[start:end]
                added_count = np.sum(added_points)

                if count + added_count <= self._max_bins:
                    keep_mask[start:end] = True
                    count += added_count
                else:
                    break

        ### Construct sparse output and optional diagnostics
        grid_xs = self._Binner.GetXAxis()
        selected = np.flatnonzero(keep_mask)
        if self.track_diagnostics:
            self.last_diagnostics = {**base_diagnostics(grid_ys, selected, selected.size), "threshold_value": None, "target_fraction": None, "target_fraction_reached": None}
        else:
            self.last_diagnostics = {}
        return grid_xs[keep_mask], grid_ys[keep_mask]

    # Diagnostics interface
    ## Return diagnostics without exposing mutable internal state
    def get_last_diagnostics(self) -> dict[str, Any]:
        """Return an independent copy of diagnostics produced by the last call.

        :return: Selection diagnostics, or an empty mapping when tracking is disabled.
        :rtype: dict[str, Any]
        """
        return dict(self.last_diagnostics)
