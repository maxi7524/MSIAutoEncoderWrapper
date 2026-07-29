"""Public forward and inverse binning diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

from ....utils.exceptions import raise_validation_error
from .metrics import summarize_forward, summarize_inverse


class BinningAnalysis:
    """Expose preprocessing reports independently of model reconstruction."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def forward_report(
        self,
        spectrum_ids: Optional[Sequence[int]] = None,
    ) -> Mapping[str, float]:
        """Report finite output and TIC behavior of forward binning."""
        runtime = self.owner.models[self.owner.default_model_name]
        selected = self.owner.selected_ids(spectrum_ids, runtime.dataset)
        finite: list[float] = []
        ratios: list[float] = []
        for spectrum_id in selected:
            mz_axis, intensities = self.owner.reader.GetSpectrum(int(spectrum_id))
            binned = np.asarray(self.owner.binner(xs=mz_axis, ys=intensities))
            finite.append(float(np.mean(np.isfinite(binned))))
            raw_tic = float(np.sum(intensities))
            ratios.append(float(np.sum(binned)) / raw_tic if raw_tic else np.nan)
        return summarize_forward(finite, ratios)

    def inverse_report(
        self,
        spectrum_ids: Optional[Sequence[int]] = None,
    ) -> Mapping[str, float]:
        """Report loss introduced by inverse/forward binning round trip."""
        inverse_binner = getattr(self.owner.context, "inverse_binner", None)
        if inverse_binner is None:
            raise_validation_error(
                "BinningAnalysis", "The active context has no inverse binner."
            )
        runtime = self.owner.models[self.owner.default_model_name]
        selected = self.owner.selected_ids(spectrum_ids, runtime.dataset)
        mse_values: list[float] = []
        mae_values: list[float] = []
        ratios: list[float] = []
        for spectrum_id in selected:
            binned = np.asarray(runtime.dataset[int(spectrum_id)][1], dtype=np.float64)
            mz_axis, intensities = inverse_binner(binned)
            round_trip = np.asarray(
                self.owner.binner(xs=mz_axis, ys=intensities), dtype=np.float64
            )
            residual = binned - round_trip
            mse_values.append(float(np.mean(residual**2)))
            mae_values.append(float(np.mean(np.abs(residual))))
            tic = float(np.sum(binned))
            ratios.append(float(np.sum(round_trip)) / tic if tic else np.nan)
        return summarize_inverse(mse_values, mae_values, ratios)

    def overview(
        self,
        spectrum_ids: Optional[Sequence[int]] = None,
    ) -> Mapping[str, Mapping[str, float]]:
        """Return forward and inverse diagnostics as one logical group."""
        return {
            "forward": self.forward_report(spectrum_ids),
            "inverse": self.inverse_report(spectrum_ids),
        }
