"""Statistical and lossless inverse-binner strategies."""

from __future__ import annotations


from ...data import InverseSpectrumBatch, SpectrumBatch
from ...utils.logger import get_custom_logger
from ..base_inverse import MSIBaseInverseBinner
from ..binners_manager import BinnerManager
from .reconstruction import pack_dense_projection


logger = get_custom_logger(__name__)


@BinnerManager.register_inverse_binner("PassthroughInverseBinner")
class PassthroughInverseBinner(MSIBaseInverseBinner):
    """Return the binned input without selection or reconstruction."""

    def transform(self, batch: SpectrumBatch) -> InverseSpectrumBatch:
        """Pack the input values on their existing shared mass axis.

        :param batch: Dense binned spectra with shape ``(B, F)``.
        :type batch: SpectrumBatch
        :return: Packed spectra with exactly the input axis and values.
        :rtype: InverseSpectrumBatch
        """
        axis = batch.space.mass_axis.to(device=batch.device, dtype=batch.spectra.dtype)
        return pack_dense_projection(batch, axis, batch.spectra)
