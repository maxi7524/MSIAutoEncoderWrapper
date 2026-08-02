"""Abstract contracts for metrics defined on distinct object spaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

MetricDirection = Literal["minimize", "maximize", "absolute_minimize"]
MetricScope = Literal["sample", "feature", "class", "dataset"]


@dataclass(frozen=True)
class MetricRequirements:
    """Declare mathematical input properties required by a metric."""

    requires_nonnegative: bool = False
    requires_linear_intensity: bool = False
    accepts_samplewise_scalar: bool = True
    required_space: Literal["normalized", "source"] | None = None


class BaseMetric(nn.Module, ABC):
    """Base contract shared by all model-independent metrics."""

    direction: MetricDirection
    scope: MetricScope
    requirements = MetricRequirements()


class SpectrumMetric(BaseMetric, ABC):
    """Metric whose compared objects are spectra on a feature or m/z axis."""

    @abstractmethod
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        mass_axis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one metric value per input spectrum."""
        raise NotImplementedError


class ClassificationMetric(BaseMetric, ABC):
    """Metric evaluating sample-level classifier decisions."""

    @abstractmethod
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Accumulate one classification batch."""
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> Any:
        """Return a dataset-level classification result."""
        raise NotImplementedError


class ClassMetric(BaseMetric, ABC):
    """Metric returning one value for every target class."""

    scope: MetricScope = "class"

    @abstractmethod
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Accumulate predictions used for per-class results."""
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> torch.Tensor:
        """Return a tensor with shape ``[classes]``."""
        raise NotImplementedError
