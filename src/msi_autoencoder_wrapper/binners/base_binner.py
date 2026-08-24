from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Any, Optional

from ..configuration import ConfigurableComponent
from ..utils.exceptions import raise_validation_error


class MSIBaseBinner(ConfigurableComponent, ABC):
    """
    Abstract Base Class establishing structural interface benchmarks for forward spectral binning algorithms.
    """

    def __init__(self, active_context: Optional[Any] = None) -> None:
        """
        Initializes the abstract binner base with an optional active context bridge.

        :param active_context: Active execution session proxy tracking live datasets.
        :type active_context: Optional[Any]
        """
        self._config: dict[str, Any] = {}
        self.active_context = active_context
        self.dtype = getattr(
            getattr(active_context, "_wrapper", None), "dtype", torch.float32
        )

    def __call__(self, batch: Any) -> Any:
        """Apply the canonical batched Torch transformation."""
        return self.transform(batch)

    @abstractmethod
    def transform(self, batch: Any) -> Any:
        """Transform a Torch batch without a separate single-spectrum backend."""
        pass

    def transform_spectrum(
        self,
        mass_values: torch.Tensor | np.ndarray,
        intensities: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Transform one spectrum through the canonical ``B=1`` batch path."""
        from ..data import RawSpectrumBatch

        mass_tensor = torch.as_tensor(mass_values, dtype=self.dtype)
        intensity_tensor = torch.as_tensor(intensities, dtype=self.dtype)
        point_count = int(mass_tensor.numel())
        batch = RawSpectrumBatch(
            sample_ids=torch.zeros(1, dtype=torch.long, device=intensity_tensor.device),
            mass_values=mass_tensor.to(device=intensity_tensor.device),
            intensities=intensity_tensor,
            offsets=torch.tensor([0, point_count], dtype=torch.long, device=intensity_tensor.device),
            sample_indices=torch.zeros(point_count, dtype=torch.long, device=intensity_tensor.device),
        )
        return self.transform(batch).spectra[0]

    @abstractmethod
    def GetXMin(self) -> float:
        """Retrieves absolute starting floor mass boundary threshold configured across the shared grid."""
        pass

    @abstractmethod
    def GetXMax(self) -> float:
        """Retrieves absolute terminal ceiling mass boundary threshold configured across the shared grid."""
        pass

    @abstractmethod
    def GetXAxis(self) -> np.ndarray:
        """Retrieves the unified master grid alignment reference array matrix containing m/z points."""
        pass

    @abstractmethod
    def GetXAxisDepth(self) -> int:
        """Computes total absolute feature dimension channels length available within binned spaces."""
        pass

    def GetMzRangeIndices(self, mz_min: float, mz_max: float) -> np.ndarray:
        """Return grid indices whose centers lie in an inclusive m/z range.

        :param mz_min: Inclusive lower mass bound.
        :type mz_min: float
        :param mz_max: Inclusive upper mass bound.
        :type mz_max: float
        :return: One-dimensional integer grid indices.
        :rtype: numpy.ndarray
        :raises ValidationError: If the range is reversed.
        """
        if mz_min > mz_max:
            raise_validation_error("Binner", "mz_min cannot be greater than mz_max.")
        axis = np.asarray(self.GetXAxis())
        return np.flatnonzero((axis >= float(mz_min)) & (axis <= float(mz_max)))

    def GetBinIndices(self, mz: float, tolerance: float = 0.0) -> np.ndarray:
        """Return grid indices in an absolute tolerance around one m/z value.

        :param mz: Center mass value.
        :type mz: float
        :param tolerance: Non-negative absolute tolerance.
        :type tolerance: float
        :return: One-dimensional integer grid indices.
        :rtype: numpy.ndarray
        :raises ValidationError: If tolerance is negative.
        """
        if tolerance < 0:
            raise_validation_error("Binner", "tolerance cannot be negative.")
        return self.GetMzRangeIndices(float(mz) - tolerance, float(mz) + tolerance)

    def map_mass_values_to_bins(self, mass_values: np.ndarray) -> np.ndarray:
        """Map raw m/z values to dense bins when the strategy supports it.

        :param mass_values: Raw m/z coordinates with arbitrary NumPy shape.
        :type mass_values: numpy.ndarray
        :return: Integer bin indices, with ``-1`` for values outside the output
            coordinate system.
        :rtype: numpy.ndarray
        :raises NotImplementedError: If the binning strategy cannot expose its
            exact forward coordinate mapping.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose an exact m/z-to-bin mapping."
        )
