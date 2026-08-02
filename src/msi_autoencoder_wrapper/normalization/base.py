"""Mathematical contracts for reversible MSI intensity normalization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import torch

NormalizationStage = Literal["raw", "binned"]
DenormalizationStage = Literal["after_decode", "after_inverse_binning"]
OutputSpace = Literal["normalized", "source"]


@dataclass(frozen=True)
class NormalizationCapabilities:
    """Describe properties preserved by one normalization step."""

    invertible: bool
    preserves_nonnegativity: bool
    preserves_linear_intensity: bool
    samplewise_scalar: bool
    can_inverse_after_binning: bool
    can_inverse_after_inverse_binning: bool


@dataclass(frozen=True)
class NormalizationTrace:
    """Carry reversible per-batch normalization state through the model graph."""

    pipeline_name: str
    stage: NormalizationStage
    states: tuple[Mapping[str, torch.Tensor], ...] = field(default_factory=tuple)
    capabilities: NormalizationCapabilities = field(
        default_factory=lambda: NormalizationCapabilities(
            invertible=True,
            preserves_nonnegativity=True,
            preserves_linear_intensity=True,
            samplewise_scalar=True,
            can_inverse_after_binning=True,
            can_inverse_after_inverse_binning=True,
        )
    )

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "NormalizationTrace":
        """Move tensor state to ``device`` while retaining immutable semantics."""
        resolved = torch.device(device)
        return NormalizationTrace(
            pipeline_name=self.pipeline_name,
            stage=self.stage,
            states=tuple(
                {
                    name: value.to(resolved, non_blocking=non_blocking)
                    for name, value in state.items()
                }
                for state in self.states
            ),
            capabilities=self.capabilities,
        )


class BaseNormalization(ABC):
    """Define one differentiable and optionally reversible normalization step."""

    capabilities: NormalizationCapabilities

    def __init__(self, epsilon: float = 1e-12) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be greater than zero.")
        self.epsilon = float(epsilon)

    def fit(self, batches: Any) -> "BaseNormalization":
        """Fit dataset-level state; samplewise implementations need no pass."""
        del batches
        return self

    @abstractmethod
    def transform(
        self,
        values: torch.Tensor,
        *,
        sample_indices: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Normalize dense ``[B, F]`` or packed ``[N]`` intensities."""
        raise NotImplementedError

    @abstractmethod
    def inverse(
        self,
        values: torch.Tensor,
        state: Mapping[str, torch.Tensor],
        *,
        sample_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Restore values using state returned by :meth:`transform`."""
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        """Return reproducible constructor parameters."""
        return {"epsilon": self.epsilon}
