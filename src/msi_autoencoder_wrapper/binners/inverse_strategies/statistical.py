"""Statistical and lossless inverse-binner strategies."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from ...data import InverseSpectrumBatch, SpectrumBatch
from ...utils.exceptions import raise_validation_error
from ...utils.logger import get_custom_logger
from ..base_binner import MSIBaseBinner
from ..base_inverse import MSIBaseInverseBinner
from ..binners_manager import BinnerManager
from .reconstruction import pack_dense_projection, pack_top_projection


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


@BinnerManager.register_inverse_binner("StatisticalInverseBinner")
class StatisticalInverseBinner(MSIBaseInverseBinner):
    """Learn one sparse reconstruction axis from a deterministic spectrum sample.

    Raw points are quantized to ``mz_resolution`` only while estimating the
    shared reconstruction representation. For each forward bin, the estimator
    stores the relative positive intensity support of its retained m/z values.
    Reconstruction is therefore a single matrix multiplication, not a lookup of
    original per-pixel axes.
    """

    def __init__(
        self,
        binner: Optional[MSIBaseBinner] = None,
        reconstruction_mass_axis: torch.Tensor | np.ndarray | None = None,
        active_context: Optional[Any] = None,
        reader: Optional[Any] = None,
        sample_fraction: float = 0.1,
        max_sample_spectra: int | None = None,
        mz_resolution: float = 0.5,
        candidate_axis_size: int = 10_000,
        reconstruction_axis_size: int = 2_000,
        max_candidates_per_bin: int = 3,
        random_seed: int = 0,
    ) -> None:
        """Build the statistical sparse inverse map once.

        :param binner: Forward regular binner defining source bins.
        :type binner: Optional[MSIBaseBinner]
        :param reconstruction_mass_axis: Explicit axis. When supplied, only an
            identity bin-to-axis projection is available; use the estimator path
            for irregular source spectra.
        :type reconstruction_mass_axis: torch.Tensor | numpy.ndarray | None
        :param active_context: Context supplying the reader when ``reader`` is omitted.
        :type active_context: Optional[Any]
        :param reader: Reader of original irregular spectra.
        :type reader: Optional[Any]
        :param sample_fraction: Fraction of spectra sampled without replacement.
        :type sample_fraction: float
        :param max_sample_spectra: Optional upper bound on sampled spectra.
        :type max_sample_spectra: int | None
        :param mz_resolution: Quantization precision of learned m/z coordinates.
        :type mz_resolution: float
        :param candidate_axis_size: Number of supported coordinates considered.
        :type candidate_axis_size: int
        :param reconstruction_axis_size: Maximum retained reconstruction coordinates
            per transformed spectrum.
        :type reconstruction_axis_size: int
        :param max_candidates_per_bin: Maximum retained m/z directions for one
            forward bin in the learned projection.
        :type max_candidates_per_bin: int
        :param random_seed: Deterministic sample seed.
        :type random_seed: int
        """
        self.sample_fraction = float(sample_fraction)
        self.max_sample_spectra = max_sample_spectra
        self.mz_resolution = float(mz_resolution)
        self.candidate_axis_size = int(candidate_axis_size)
        self.reconstruction_axis_size = int(reconstruction_axis_size)
        self.max_candidates_per_bin = int(max_candidates_per_bin)
        self.random_seed = int(random_seed)
        if not 0.0 < self.sample_fraction <= 1.0:
            raise_validation_error("StatisticalInverseBinner", "sample_fraction must belong to (0, 1].")
        if not np.isfinite(self.mz_resolution) or self.mz_resolution <= 0:
            raise_validation_error("StatisticalInverseBinner", "mz_resolution must be finite and positive.")
        if self.max_sample_spectra is not None and int(self.max_sample_spectra) <= 0:
            raise_validation_error("StatisticalInverseBinner", "max_sample_spectra must be positive or None.")
        if (
            self.candidate_axis_size <= 0
            or self.reconstruction_axis_size <= 0
            or self.max_candidates_per_bin <= 0
        ):
            raise_validation_error("StatisticalInverseBinner", "axis sizes must be positive.")

        super().__init__(binner, reconstruction_mass_axis, active_context)
        self.reader = reader or getattr(self.active_context, "reader", None)
        if reconstruction_mass_axis is None:
            if self.reader is None:
                raise_validation_error(
                    "StatisticalInverseBinner",
                    "A reader is required to estimate a sparse reconstruction axis.",
                )
            axis, mapping = self._estimate_sparse_projection(self.reader)
            self.reconstruction_mass_axis = self._validate_axis(axis).to(self.dtype)
            self._projection = mapping  # (F, G)
        else:
            self._projection = self._identity_projection()
        self._config = {
            "sample_fraction": self.sample_fraction,
            "max_sample_spectra": self.max_sample_spectra,
            "mz_resolution": self.mz_resolution,
            "candidate_axis_size": self.candidate_axis_size,
            "reconstruction_axis_size": self.reconstruction_axis_size,
            "max_candidates_per_bin": self.max_candidates_per_bin,
            "random_seed": self.random_seed,
        }

    def _identity_projection(self) -> torch.Tensor:
        """Map a supplied axis only when it equals the forward binner axis."""
        source_axis = torch.as_tensor(self._Binner.GetXAxis(), dtype=self.dtype)
        if not torch.equal(self.reconstruction_mass_axis, source_axis):
            raise_validation_error(
                "StatisticalInverseBinner",
                "An explicit reconstruction_mass_axis must equal the binner axis; "
                "otherwise provide a reader to estimate the sparse projection.",
            )
        return torch.eye(source_axis.numel(), dtype=self.dtype)

    def _estimate_sparse_projection(self, reader: Any) -> tuple[np.ndarray, torch.Tensor]:
        """Estimate ``forward bin -> shared sparse m/z`` proportions from raw spectra."""
        spectrum_count = int(reader.GetNumberOfSpectra())
        if spectrum_count <= 0:
            raise_validation_error("StatisticalInverseBinner", "Cannot estimate an axis from an empty reader.")
        requested = max(1, int(np.ceil(spectrum_count * self.sample_fraction)))
        sample_count = min(spectrum_count, requested)
        if self.max_sample_spectra is not None:
            sample_count = min(sample_count, int(self.max_sample_spectra))
        sampled_indices = np.random.default_rng(self.random_seed).choice(
            spectrum_count, size=sample_count, replace=False
        )
        source_depth = self._Binner.GetXAxisDepth()
        support: dict[tuple[int, int], float] = {}
        candidate_support: dict[int, float] = {}

        logger.info(
            "Estimating sparse inverse axis from %s of %s spectra at %.6g m/z precision.",
            sample_count,
            spectrum_count,
            self.mz_resolution,
        )

        # Sample original irregular spectra and aggregate their within-bin proportions.
        for spectrum_index in sampled_indices:
            mz, intensities = reader.GetSpectrum(int(spectrum_index))
            mz = np.asarray(mz, dtype=np.float32)
            intensities = np.asarray(intensities, dtype=np.float32)
            bin_indices = np.floor((mz - self._Binner.GetXMin()) / self._Binner.bin_step).astype(np.int64)
            valid = (
                np.isfinite(mz)
                & np.isfinite(intensities)
                & (intensities > 0)
                & (mz >= self._Binner.GetXMin())
                & (mz <= self._Binner.GetXMax())
                & (bin_indices >= 0)
                & (bin_indices < source_depth)
            )
            if not np.any(valid):
                continue
            mz = mz[valid]
            intensities = intensities[valid]
            bin_indices = bin_indices[valid]
            bin_sum = np.bincount(bin_indices, weights=intensities, minlength=source_depth)
            proportions = intensities / bin_sum[bin_indices]
            ticks = np.rint(mz / self.mz_resolution).astype(np.int64)
            for bin_index, tick, proportion in zip(bin_indices, ticks, proportions):
                key = (int(bin_index), int(tick))
                support[key] = support.get(key, 0.0) + float(proportion)
                candidate_support[int(tick)] = candidate_support.get(int(tick), 0.0) + float(proportion)

        if not candidate_support:
            raise_validation_error("StatisticalInverseBinner", "The sampled spectra contain no positive finite points in the binner range.")

        # Retain a fixed global candidate axis; each output spectrum is reduced later.
        ranked_ticks = sorted(candidate_support, key=lambda tick: (-candidate_support[tick], tick))
        retained_ticks = ranked_ticks[: min(self.candidate_axis_size, len(ranked_ticks))]
        retained_ticks.sort()
        axis = np.asarray(retained_ticks, dtype=np.float32) * self.mz_resolution
        target_index = {tick: index for index, tick in enumerate(retained_ticks)}
        projection = torch.zeros((source_depth, len(retained_ticks)), dtype=self.dtype)
        for (bin_index, tick), value in support.items():
            column = target_index.get(tick)
            if column is not None:
                projection[bin_index, column] += value

        # Keep a small number of statistically dominant destinations per source bin.
        top_count = min(self.max_candidates_per_bin, projection.shape[1])
        if top_count < projection.shape[1]:
            top_indices = torch.topk(projection, k=top_count, dim=1).indices  # (F, K)
            retained = torch.zeros_like(projection, dtype=torch.bool)
            retained.scatter_(1, top_indices, True)
            projection = torch.where(retained, projection, torch.zeros_like(projection))
        row_sum = projection.sum(dim=1, keepdim=True)  # (F, 1)
        projection = torch.where(row_sum > 0, projection / row_sum.clamp_min(torch.finfo(projection.dtype).eps), projection)  # (F, G)
        return axis, projection

    def transform(self, batch: SpectrumBatch) -> InverseSpectrumBatch:
        """Distribute each binned value using the learned sparse projection.

        :param batch: Binned spectra with shape ``(B, F)``.
        :type batch: SpectrumBatch
        :return: Packed reconstructed spectra on the learned shared axis.
        :rtype: InverseSpectrumBatch
        """
        if batch.spectra.shape[1] != self._projection.shape[0]:
            raise_validation_error("StatisticalInverseBinner", "Input feature count does not match the source binner axis.")
        projection = self._projection.to(device=batch.device)
        values = batch.spectra.to(projection.dtype) @ projection  # (B, G)
        axis = self.reconstruction_mass_axis.to(device=batch.device, dtype=projection.dtype)
        return pack_top_projection(batch, axis, values, self.reconstruction_axis_size)
