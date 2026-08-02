"""Portable configuration for model-dataset partitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from ....utils.exceptions import raise_validation_error


_SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitConfig:
    """Describe one deterministic train, validation, and test partition."""

    strategy: str = "random"
    fractions: Mapping[str, float] = field(
        default_factory=lambda: {"train": 0.8, "validation": 0.0, "test": 0.2}
    )
    seed: int = 0
    parameters: Mapping[str, Any] = field(default_factory=dict)
    assignments: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.strategy not in {
            "random",
            "grouped",
            "target_stratified",
            "mask_stratified",
            "predefined",
        }:
            raise_validation_error(
                "DatasetSplit", f"Unsupported split strategy '{self.strategy}'."
            )
        if set(self.fractions) != set(_SPLIT_NAMES):
            raise_validation_error(
                "DatasetSplit",
                "fractions must contain train, validation, and test.",
            )
        resolved = {name: float(self.fractions[name]) for name in _SPLIT_NAMES}
        if any(value < 0 or value > 1 for value in resolved.values()):
            raise_validation_error("DatasetSplit", "Split fractions must be in [0, 1].")
        if abs(sum(resolved.values()) - 1.0) > 1e-9:
            raise_validation_error("DatasetSplit", "Split fractions must sum to one.")
        if resolved["train"] <= 0:
            raise_validation_error("DatasetSplit", "The train fraction must be positive.")
        if self.strategy == "predefined" and self.assignments is None:
            raise_validation_error(
                "DatasetSplit", "predefined splitting requires assignments."
            )
        object.__setattr__(self, "fractions", resolved)
        object.__setattr__(self, "parameters", dict(self.parameters))

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SplitConfig":
        """Build a validated split configuration from a portable mapping."""
        return cls(
            strategy=str(config.get("strategy", "random")),
            fractions=dict(
                config.get(
                    "fractions",
                    {"train": 0.8, "validation": 0.0, "test": 0.2},
                )
            ),
            seed=int(config.get("seed", 0)),
            parameters=dict(config.get("parameters", {})),
            assignments=config.get("assignments"),
        )

    def get_config(self) -> Dict[str, Any]:
        """Return the portable split definition."""
        result: Dict[str, Any] = {
            "strategy": self.strategy,
            "fractions": dict(self.fractions),
            "seed": self.seed,
            "parameters": dict(self.parameters),
        }
        if self.assignments is not None:
            result["assignments"] = dict(self.assignments)
        return result

