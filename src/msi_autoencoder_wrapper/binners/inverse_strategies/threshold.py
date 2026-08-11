"""Quantile-based inverse binning."""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..base_binner import MSIBaseBinner
from ..base_inverse import MSIBaseInverseBinner
from ..binners_manager import BinnerManager
from ...data import InverseSpectrumBatch, SpectrumBatch
from ...utils.exceptions import raise_validation_error
from .reconstruction import pack_point_projection


@BinnerManager.register_inverse_binner("QuantileInverseBinner")
class QuantileInverseBinner(MSIBaseInverseBinner):
    """Select positive grid values above a per-spectrum intensity quantile."""

    def __init__(
        self,
        binner: Optional[MSIBaseBinner] = None,
        quantile: float = 0.95,
        reconstruction_mass_axis: torch.Tensor | None = None,
        active_context: Optional[Any] = None,
    ) -> None:
        super().__init__(binner, reconstruction_mass_axis, active_context)
        self.quantile = float(quantile)
        if not 0.0 <= self.quantile <= 1.0:
            raise_validation_error("QuantileInverseBinner", "quantile must belong to [0, 1].")
        self._config = {"quantile": self.quantile}

    def transform(self, batch: SpectrumBatch) -> InverseSpectrumBatch:
        """Select values above the row quantile and project them to the target axis."""
        cleaned = torch.where(torch.isfinite(batch.spectra) & (batch.spectra > 0), batch.spectra, torch.zeros_like(batch.spectra))  # (B, F)
        if batch.batch_size == 0:
            keep_mask = torch.zeros_like(cleaned, dtype=torch.bool)
        else:
            threshold = torch.quantile(cleaned, self.quantile, dim=1, keepdim=True)  # (B, 1)
            keep_mask = (cleaned > 0) & (cleaned >= threshold)  # (B, F)
        return pack_point_projection(batch, keep_mask, self.reconstruction_mass_axis)
