from abc import ABC, abstractmethod
import numpy as np
import torch
from .base_binner import MSIBaseBinner
from typing import Any, Optional
from ..utils.exceptions import raise_validation_error
from ..configuration import ConfigurableComponent

class MSIBaseInverseBinner(ConfigurableComponent, ABC):
    """
    Abstract Base Class establishing architectural interfaces for reverse signal reconstructors.
    """

    def __init__(
        self,
        binner: Optional[MSIBaseBinner] = None,
        reconstruction_mass_axis: torch.Tensor | np.ndarray | None = None,
        active_context: Optional[Any] = None,
    ) -> None:
        """
        Binds the structural forward master grid tracking object to the inverse processing pipeline.

        :param binner: Active forward binner strategy. Falls back to active_context lookup if None.
        :type binner: Optional[MSIBaseBinner]
        :param active_context: Active execution session proxy tracking live datasets.
        :type active_context: Optional[Any]
        :raises ValueError: If no binner can be resolved directly or from the context.
        """
        self._config: dict[str, Any] = {}
        self.active_context = active_context or getattr(binner, "active_context", None)
        
        # Resolve binner instance via direct injection or session context proxy
        if binner is not None:
            self._Binner = binner
        elif active_context and getattr(active_context, "binner", None) is not None:
            self._Binner = active_context.binner
        else:
            raise_validation_error(
                context_name="InverseBinner",
                message=(
                    "An explicit binner instance or an active context with a registered "
                    "binner is required."
                ),
            )

        self.reconstruction_mass_axis = self._resolve_reconstruction_axis(
            reconstruction_mass_axis
        )

    def __call__(self, batch: Any) -> Any:
        """Apply the canonical batched Torch inverse transformation."""
        return self.transform(batch)

    @abstractmethod
    def transform(self, batch: Any) -> Any:
        """Transform a dense Torch batch into a packed reconstruction."""
        pass

    def transform_spectra(
        self,
        spectra: torch.Tensor | np.ndarray,
    ) -> Any:
        """Transform one or more dense spectra through the canonical batch path."""
        from ..data import SpectrumBatch, SpectrumSpace

        values = spectra if isinstance(spectra, torch.Tensor) else torch.tensor(np.asarray(spectra))
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2:
            raise_validation_error("InverseBinner", "spectra must have shape [F] or [B, F].")
        axis = torch.as_tensor(self._Binner.GetXAxis(), device=values.device, dtype=values.dtype)
        return self.transform(SpectrumBatch(
            sample_ids=torch.arange(values.shape[0], device=values.device),
            spectra=values,
            space=SpectrumSpace(mass_axis=axis),
        ))

    def _resolve_reconstruction_axis(
        self,
        explicit_axis: torch.Tensor | np.ndarray | None,
    ) -> torch.Tensor:
        """Resolve one shared reconstruction axis without retaining per-pixel axes."""
        if explicit_axis is not None:
            return self._validate_axis(explicit_axis)

        reader = getattr(self.active_context, "reader", None)
        if reader is not None:
            capabilities = reader.capabilities
            if (
                getattr(capabilities, "reconstruction_mass_axis", False)
                or getattr(capabilities, "shared_mass_axis", False)
            ):
                return self._validate_axis(reader.GetXAxis())
        if reader is not None:
            cached = getattr(reader, "_inverse_reconstruction_axis", None)
            if cached is None:
                count = int(reader.GetNumberOfSpectra())
                minimum = float("inf")
                maximum = float("-inf")
                total_length = 0
                for index in range(count):
                    axis, _ = reader.GetSpectrum(index)
                    axis = np.asarray(axis)
                    if axis.size:
                        minimum = min(minimum, float(axis[0]))
                        maximum = max(maximum, float(axis[-1]))
                        total_length += int(axis.size)
                if count and total_length and np.isfinite(minimum) and np.isfinite(maximum):
                    mean_length = max(1, int(round(total_length / count)))
                    cached = np.linspace(minimum, maximum, mean_length, dtype=np.float64)
                    setattr(reader, "_inverse_reconstruction_axis", cached)
            if cached is not None:
                return self._validate_axis(cached)
        return self._validate_axis(self._Binner.GetXAxis())

    @staticmethod
    def _validate_axis(axis: torch.Tensor | np.ndarray) -> torch.Tensor:
        resolved = torch.as_tensor(axis, dtype=torch.float64)
        if resolved.ndim != 1 or resolved.numel() == 0:
            raise_validation_error("InverseBinner", "reconstruction_mass_axis must be a non-empty one-dimensional array.")
        if not bool(torch.isfinite(resolved).all()) or (resolved.numel() > 1 and not bool(torch.all(resolved[1:] > resolved[:-1]))):
            raise_validation_error("InverseBinner", "reconstruction_mass_axis must be finite and strictly increasing.")
        return resolved
