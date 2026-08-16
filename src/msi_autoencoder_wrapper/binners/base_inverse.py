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
        self.dtype = getattr(
            getattr(self.active_context, "_wrapper", None),
            "dtype",
            getattr(binner, "dtype", torch.float32),
        )
        
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

        values = torch.as_tensor(spectra, dtype=self.dtype)
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
        """Resolve an explicit, real shared, or binner mass axis.

        A reader is consulted only when it declares an actual shared mass axis.
        In particular, this method must not estimate an axis by scanning spectra:
        that is the responsibility of ``StatisticalInverseBinner``.
        """
        if explicit_axis is not None:
            return self._validate_axis(explicit_axis)

        reader = getattr(self.active_context, "reader", None)
        if reader is not None:
            capabilities = reader.capabilities
            if getattr(capabilities, "shared_mass_axis", False):
                return self._validate_axis(reader.GetXAxis())
        return self._validate_axis(self._Binner.GetXAxis())

    @staticmethod
    def _validate_axis(axis: torch.Tensor | np.ndarray) -> torch.Tensor:
        resolved = torch.as_tensor(axis, dtype=torch.float32)
        if resolved.ndim != 1 or resolved.numel() == 0:
            raise_validation_error("InverseBinner", "reconstruction_mass_axis must be a non-empty one-dimensional array.")
        if not bool(torch.isfinite(resolved).all()) or (resolved.numel() > 1 and not bool(torch.all(resolved[1:] > resolved[:-1]))):
            raise_validation_error("InverseBinner", "reconstruction_mass_axis must be finite and strictly increasing.")
        return resolved
