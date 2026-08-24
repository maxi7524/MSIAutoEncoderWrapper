import numpy as np
import torch
from ..base_binner import MSIBaseBinner
from ..binners_manager import BinnerManager
from typing import Any, Optional
from ...utils.exceptions import raise_validation_error
from ...data import RawSpectrumBatch, SharedAxisRawBatch, SpectrumBatch, SpectrumSpace


@BinnerManager.register_binner("LinearBinning")
class LinearBinning(MSIBaseBinner):
    """Accumulate irregular MSI points into equidistant mass-to-charge bins.

    Bin centers form the shared output axis. A point maps to its unique interval
    ``[x_min + k * bin_step, x_min + (k + 1) * bin_step)``; the final interval
    also includes ``x_max``. There is no nearest-centre tolerance. Points below
    ``x_min`` or above ``x_max`` are discarded.
    """

    # Binner configuration and axis construction
    ## __init__ setup and validation
    def __init__(self, bin_step: float, x_min: Optional[float] = None, x_max: Optional[float] = None, aggregation: str = "sum", active_context: Optional[Any] = None) -> None:
        """Initialize the binning interval and construct the shared output axis.

        Explicit boundaries take precedence over boundaries exposed by the active
        context reader.

        :param bin_step: Width of each regular m/z interval.
        :type bin_step: float
        :param x_min: Lower binning boundary. Resolved from ``active_context`` when
            omitted.
        :type x_min: Optional[float]
        :param x_max: Upper binning boundary. Resolved from ``active_context`` when
            omitted.
        :type x_max: Optional[float]
        :param active_context: Optional context providing a reader and default mass
            boundaries.
        :type active_context: Optional[Any]
        :raises ValidationError: If neither explicit boundaries nor reader boundaries
            provide a complete mass range.
        """
        super().__init__(active_context=active_context)

        ### Resolve the mass range

        #### Explicit arguments have priority over values obtained from the active context
        resolved_x_min = x_min
        resolved_x_max = x_max

        #### Missing boundaries are read from the currently active MSI reader
        if active_context and getattr(active_context, "reader", None) is not None:
            if resolved_x_min is None:
                resolved_x_min = active_context.reader.GetXMin()
            if resolved_x_max is None:
                resolved_x_max = active_context.reader.GetXMax()

        #### A complete mass range is required before the binning axis can be constructed
        if resolved_x_min is None or resolved_x_max is None:
            raise_validation_error(
                context_name="LinearBinning",
                message=(
                    "Explicit 'x_min' and 'x_max' values or an active context reader "
                    "session are required."
                ),
            )

        ### Store the resolved binning parameters

        #### Convert external numerical values to a consistent internal representation
        self.x_min = float(resolved_x_min)
        self.x_max = float(resolved_x_max)
        self.bin_step = float(bin_step)
        if not np.isfinite(self.bin_step) or self.bin_step <= 0:
            raise_validation_error("LinearBinning", "bin_step must be finite and positive.")
        if not np.isfinite(self.x_min) or not np.isfinite(self.x_max) or self.x_min >= self.x_max:
            raise_validation_error("LinearBinning", "x_min and x_max must be finite and x_min < x_max.")
        if aggregation not in {"sum", "mean"}:
            raise_validation_error("LinearBinning", "aggregation must be 'sum' or 'mean'.")
        self.aggregation = aggregation

        #### Preserve the effective configuration for reconstruction or serialization
        self._config = {"x_min": self.x_min, "x_max": self.x_max, "bin_step": self.bin_step, "aggregation": self.aggregation}

        ### Construct the regular binning axis

        #### Bin edges define the intervals into which original mass values are accumulated
        self.bin_edges = np.arange(self.x_min, self.x_max + self.bin_step, self.bin_step, dtype=np.float32)

        #### Bin centers form the physical mass axis of the dense output spectrum
        self.grid = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

    # Spectrum transformation
    ## Transform a batch of spectra
    def transform(
        self, batch: RawSpectrumBatch | SharedAxisRawBatch
    ) -> SpectrumBatch:
        """Bin every packed raw point with one vectorized Torch aggregation.

        The operation runs on the device already storing ``batch``. A regular
        axis permits direct arithmetic bin lookup and avoids a binary search.

        :param batch: Packed raw spectra on CPU or CUDA.
        :type batch: RawSpectrumBatch
        :return: Dense spectra and their shared physical axis.
        :rtype: SpectrumBatch
        """
        ### Select the algorithm appropriate for the batch representation

        #### A shared mass axis can be mapped once and reused for all spectra
        if isinstance(batch, SharedAxisRawBatch):
            return self._transform_shared_axis_batch(batch)

        ### Read the packed spectral points

        #### Raw points from all spectra are stored in common one-dimensional tensors
        mz = batch.mass_values.to(self.dtype)
        intensity = batch.intensities.to(self.dtype)

        ### Map each mass value to a bin

        #### Regular spacing permits direct arithmetic computation of the bin index
        bin_indices = torch.floor((mz - self.bin_edges[0]) / self.bin_step).to(
            torch.long
        )

        #### The rightmost boundary is explicitly included in the final bin
        bin_indices = torch.where(
            mz == self.bin_edges[-1],
            torch.full_like(bin_indices, self.GetXAxisDepth() - 1),
            bin_indices,
        )

        ### Remove points that cannot contribute to the output

        #### A point is valid only when its coordinate and intensity are finite
        #### and its computed bin lies inside the configured mass range
        valid = (
            torch.isfinite(mz)
            & torch.isfinite(intensity)
            & (mz >= self.x_min)
            & (mz <= self.x_max)
            & (bin_indices >= 0)
            & (bin_indices < self.GetXAxisDepth())
        )

        ### Convert sample-bin coordinates into flat destination indices

        #### Each pair `(sample index, bin index)` identifies one element
        #### of the flattened dense output matrix
        flat_indices = (
            batch.sample_indices[valid] * self.GetXAxisDepth() + bin_indices[valid]
        )

        ### Aggregate intensities into the dense batch representation

        #### The batch is initially allocated as one contiguous output vector
        dense = torch.zeros(
            batch.batch_size * self.GetXAxisDepth(),
            device=intensity.device,
            dtype=intensity.dtype,
        )

        #### Intensities assigned to the same sample and bin are summed in place
        dense.scatter_add_(0, flat_indices, intensity[valid])

        #### Mean aggregation divides the accumulated signal by the contributing count
        if self.aggregation == "mean":
            counts = torch.zeros_like(dense)
            counts.scatter_add_(0, flat_indices, torch.ones_like(intensity[valid]))
            dense = torch.where(counts > 0, dense / counts.clamp_min(1), dense)

        #### Restore the matrix layout `[number of spectra, number of bins]`
        spectra = dense.view(batch.batch_size, self.GetXAxisDepth())

        ### Construct the transformed batch

        #### Convert the NumPy bin centers to a tensor compatible with the input device
        axis = torch.as_tensor(self.grid, device=mz.device, dtype=self.dtype)

        #### Preserve sample identifiers and supervised targets from the raw batch
        return SpectrumBatch(
            sample_ids=batch.sample_ids,
            spectra=spectra,
            space=SpectrumSpace(mass_axis=axis),
            targets=batch.targets,
        )

    ## Transform a batch whose spectra share one mass axis
    def _transform_shared_axis_batch(
        self,
        batch: SharedAxisRawBatch,
    ) -> SpectrumBatch:
        """Bin a native ``[B, P]`` batch while mapping its shared axis once.

        :param batch: Spectra whose rows share one physical m/z axis.
        :type batch: SharedAxisRawBatch
        :return: Dense spectra on the same device with the configured bin centers.
        :rtype: SpectrumBatch
        """

        ### Map the shared physical axis to output bins

        #### The same mass coordinate vector describes every row of the intensity matrix
        mz = batch.mass_axis.to(self.dtype)
        intensities = batch.intensities.to(self.dtype)

        #### Bin indices are therefore computed only once for all spectra
        bin_indices = torch.floor((mz - self.bin_edges[0]) / self.bin_step).to(
            torch.long
        )

        #### The final right boundary belongs to the final output bin
        bin_indices = torch.where(
            mz == self.bin_edges[-1],
            torch.full_like(bin_indices, self.GetXAxisDepth() - 1),
            bin_indices,
        )

        ### Identify usable positions on the shared axis

        #### Axis positions outside the configured range or containing non-finite
        #### coordinates must not contribute to any spectrum
        valid = (
            torch.isfinite(mz)
            & (mz >= self.x_min)
            & (mz <= self.x_max)
            & (bin_indices >= 0)
            & (bin_indices < self.GetXAxisDepth())
        )

        #### Scatter operations still require an in-range index at every position
        #### even when the corresponding source value will later be zeroed
        safe_indices = bin_indices.clamp(0, self.GetXAxisDepth() - 1)

        ### Remove invalid spectral contributions

        #### Invalid axis positions and non-finite intensities are replaced with zero
        source = torch.where(
            valid.unsqueeze(0) & torch.isfinite(intensities),
            intensities,
            torch.zeros_like(intensities),
        )

        ### Aggregate all spectra using the common bin mapping

        #### Allocate one dense output row for every spectrum
        spectra = torch.zeros(
            (batch.batch_size, self.GetXAxisDepth()),
            device=intensities.device,
            dtype=self.dtype,
        )

        #### Expand the shared bin indices across rows and sum intensities per bin
        spectra.scatter_add_(1, safe_indices.unsqueeze(0).expand_as(source), source)

        #### Mean aggregation uses one validity count per row and destination bin
        if self.aggregation == "mean":
            counts = torch.zeros_like(spectra)
            valid_source = (valid.unsqueeze(0) & torch.isfinite(intensities)).to(spectra.dtype)
            counts.scatter_add_(1, safe_indices.unsqueeze(0).expand_as(valid_source), valid_source)
            spectra = torch.where(counts > 0, spectra / counts.clamp_min(1), spectra)

        ### Construct the transformed batch

        #### Bin centers define the shared physical axis of the dense spectra
        axis = torch.as_tensor(self.grid, device=mz.device, dtype=self.dtype)

        #### Preserve identifiers and targets associated with the input spectra
        return SpectrumBatch(
            sample_ids=batch.sample_ids,
            spectra=spectra,
            space=SpectrumSpace(mass_axis=axis),
            targets=batch.targets,
        )

    # Binning-axis interface
    ## Return the lower mass-range boundary
    def GetXMin(self) -> float:
        """Return the configured lower m/z boundary."""
        return self.x_min

    ## Return the upper mass-range boundary
    def GetXMax(self) -> float:
        """Return the configured upper m/z boundary."""
        return self.x_max

    ## Return the representative mass coordinate of each bin
    def GetXAxis(self) -> np.ndarray:
        """Return the representative center coordinate of every bin."""
        return self.grid

    ## Return the dimensionality of a binned spectrum
    def GetXAxisDepth(self) -> int:
        """Return the number of bins in the regular output spectrum."""
        return len(self.grid)

    def map_mass_values_to_bins(self, mass_values: np.ndarray) -> np.ndarray:
        """Map mass values using exactly the intervals used by :meth:`transform`.

        :param mass_values: Raw m/z values with arbitrary NumPy shape.
        :type mass_values: numpy.ndarray
        :return: Dense bin indices, or ``-1`` for non-finite and out-of-range
            values.
        :rtype: numpy.ndarray
        """
        mass_array = np.asarray(mass_values, dtype=np.float64)
        indices = np.full(mass_array.shape, -1, dtype=np.int32)
        valid = np.isfinite(mass_array) & (mass_array >= self.x_min) & (mass_array <= self.x_max)
        if not np.any(valid):
            return indices
        mapped = np.floor((mass_array[valid] - self.bin_edges[0]) / self.bin_step).astype(
            np.int64
        )
        # REMARK: Match the existing Torch forward path exactly. When x_max is
        # not an integer number of bin steps from x_min, the trailing axis bin
        # has no source interval ending at x_max and must not receive x_max.
        mapped[mass_array[valid] == self.bin_edges[-1]] = self.GetXAxisDepth() - 1
        in_axis = (mapped >= 0) & (mapped < self.GetXAxisDepth())
        valid_positions = np.flatnonzero(valid)
        indices.flat[valid_positions[in_axis]] = mapped[in_axis].astype(np.int32)
        return indices
