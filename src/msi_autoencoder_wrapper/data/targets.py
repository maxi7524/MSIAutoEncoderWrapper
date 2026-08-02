"""Target schemas and batched annotation tensors."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Literal

import torch


@dataclass(frozen=True)
class TargetSchema:
    """Describe one classification target without duplicating it per sample."""

    name: str
    target_type: Literal["single_label", "multi_label"]
    class_names: tuple[str, ...]

    @property
    def class_count(self) -> int:
        """Return the number of target classes."""
        return len(self.class_names)


@dataclass(frozen=True)
class TargetSample:
    """Store target values and availability masks for one sample."""

    values: Mapping[str, torch.Tensor]
    masks: Mapping[str, torch.Tensor]

    @classmethod
    def empty(cls) -> "TargetSample":
        """Return an empty target sample."""
        return cls(values=MappingProxyType({}), masks=MappingProxyType({}))


@dataclass(frozen=True)
class TargetBatch:
    """Store collated target values, masks, and shared schemas."""

    values: Mapping[str, torch.Tensor]
    masks: Mapping[str, torch.Tensor]
    schemas: Mapping[str, TargetSchema]

    @classmethod
    def empty(cls) -> "TargetBatch":
        """Return an empty target batch."""
        empty = MappingProxyType({})
        return cls(values=empty, masks=empty, schemas=empty)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "TargetBatch":
        """Move target tensors and masks while preserving CPU schemas."""
        resolved = torch.device(device)
        return TargetBatch(
            values={
                name: value.to(resolved, non_blocking=non_blocking)
                for name, value in self.values.items()
            },
            masks={
                name: mask.to(resolved, non_blocking=non_blocking)
                for name, mask in self.masks.items()
            },
            schemas=self.schemas,
        )
