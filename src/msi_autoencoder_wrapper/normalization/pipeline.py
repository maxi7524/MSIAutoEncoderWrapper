"""Ordered normalization pipeline and active reconstruction policy."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch

from ..utils.exceptions import raise_validation_error
from ..utils.logger import get_custom_logger
from .base import (
    BaseNormalization,
    DenormalizationStage,
    NormalizationCapabilities,
    NormalizationStage,
    NormalizationTrace,
    OutputSpace,
)
from .strategies import ScalarNormalization

logger = get_custom_logger(__name__)


@dataclass(frozen=True)
class ReconstructionPolicy:
    """Control output space and the explicit inverse-normalization location."""

    output_space: OutputSpace | None = None
    denormalization_stage: DenormalizationStage = "after_inverse_binning"


class NormalizationPipeline:
    """Execute ordered normalization steps and retain enough state to invert them."""

    def __init__(
        self,
        steps: Mapping[str, BaseNormalization] | None = None,
        *,
        stage: NormalizationStage = "binned",
        reconstruction: ReconstructionPolicy | None = None,
    ) -> None:
        if stage not in {"raw", "binned"}:
            raise_validation_error("Normalization", "stage must be 'raw' or 'binned'.")
        self.steps = OrderedDict(steps or {})
        self.stage = stage
        self.reconstruction = reconstruction or ReconstructionPolicy()
        self._validate_reconstruction_policy()

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "NormalizationPipeline":
        """Construct a pipeline from the public ordered dictionary configuration."""
        raw = dict(config or {})
        stage = raw.pop("stage", "binned")
        reconstruction_raw = dict(raw.pop("reconstruction", {}))
        if "steps" in raw:
            configured_steps = raw.pop("steps")
        else:
            configured_steps = dict(raw)
            raw.clear()
        if raw:
            raise_validation_error("Normalization", f"Unknown settings: {sorted(raw)}.")
        steps: OrderedDict[str, BaseNormalization] = OrderedDict()
        for name, specification in configured_steps.items():
            if isinstance(specification, BaseNormalization):
                steps[str(name)] = specification
                continue
            params = {} if specification is None else dict(specification)
            strategy = params.pop("type", name)
            if strategy in {"tic", "max", "l2"}:
                params.setdefault("kind", strategy)
                strategy = "scalar"
            if strategy != "scalar":
                raise_validation_error("Normalization", f"Unknown strategy '{strategy}'.")
            steps[str(name)] = ScalarNormalization(**params)
        return cls(
            steps,
            stage=stage,
            reconstruction=ReconstructionPolicy(**reconstruction_raw),
        )

    @property
    def capabilities(self) -> NormalizationCapabilities:
        """Return the conservative intersection of all step capabilities."""
        values = [step.capabilities for step in self.steps.values()]
        if not values:
            return NormalizationCapabilities(True, True, True, True, True, True)
        return NormalizationCapabilities(
            **{
                field: all(getattr(value, field) for value in values)
                for field in NormalizationCapabilities.__dataclass_fields__
            }
        )

    def set_reconstruction(
        self,
        *,
        output_space: OutputSpace | None = None,
        denormalization_stage: DenormalizationStage = "after_inverse_binning",
    ) -> None:
        """Set the active inference reconstruction behavior explicitly."""
        self.reconstruction = ReconstructionPolicy(output_space, denormalization_stage)
        self._validate_reconstruction_policy()

    def set_denormalization(self, stage: DenormalizationStage) -> None:
        """Change only the inverse-normalization location for local inference.

        :param stage: ``after_decode`` or ``after_inverse_binning``.
        :type stage: DenormalizationStage
        """
        self.set_reconstruction(
            output_space=self.reconstruction.output_space,
            denormalization_stage=stage,
        )

    def set_output_space(self, output_space: OutputSpace | None) -> None:
        """Change only the active reconstruction output-space preference.

        :param output_space: ``normalized``, ``source``, or ``None`` for the
            stage-derived default.
        :type output_space: OutputSpace | None
        """
        self.set_reconstruction(
            output_space=output_space,
            denormalization_stage=self.reconstruction.denormalization_stage,
        )

    def fit(self, batches: Iterable[Any]) -> "NormalizationPipeline":
        """Fit every dataset-level step on training batches only."""
        materialized = batches
        for step in self.steps.values():
            step.fit(materialized)
        return self

    def transform(
        self,
        values: torch.Tensor,
        *,
        sample_indices: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, NormalizationTrace]:
        """Apply ordered steps and return a trace carried by downstream batches."""
        states = []
        transformed = values
        for step in self.steps.values():
            transformed, state = step.transform(
                transformed,
                sample_indices=sample_indices,
                batch_size=batch_size,
            )
            states.append(state)
        return transformed, NormalizationTrace(
            pipeline_name="+".join(self.steps) or "none",
            stage=self.stage,
            states=tuple(states),
            capabilities=self.capabilities,
        )

    def inverse(
        self,
        values: torch.Tensor,
        trace: NormalizationTrace,
        *,
        sample_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply inverse steps in reverse order using the recorded state."""
        if len(trace.states) != len(self.steps):
            raise_validation_error("Normalization", "Trace does not match the active pipeline.")
        restored = values
        for step, state in zip(reversed(tuple(self.steps.values())), reversed(trace.states)):
            restored = step.inverse(restored, state, sample_indices=sample_indices)
        return restored

    def get_config(self) -> dict[str, Any]:
        """Return the complete reproducible pipeline and reconstruction policy."""
        return {
            "stage": self.stage,
            "steps": {
                name: {"type": "scalar", **step.get_config()}
                for name, step in self.steps.items()
            },
            "reconstruction": {
                "output_space": self.reconstruction.output_space,
                "denormalization_stage": self.reconstruction.denormalization_stage,
            },
        }

    def _validate_reconstruction_policy(self) -> None:
        """Reject inverse placement unsupported by any configured step."""
        if (
            self.reconstruction.denormalization_stage == "after_inverse_binning"
            and not self.capabilities.can_inverse_after_inverse_binning
        ):
            raise_validation_error(
                "Normalization",
                "The configured pipeline cannot be inverted after inverse binning.",
            )
