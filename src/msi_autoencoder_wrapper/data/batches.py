"""Batched MSI records with explicit shapes, devices, and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from ..utils.exceptions import raise_validation_error
from .spaces import SpectrumSpace
from .targets import TargetBatch


@dataclass(frozen=True)
class RawSpectrumBatch:
    """Store variable-length raw spectra in packed contiguous tensors."""

    sample_ids: torch.Tensor
    mass_values: torch.Tensor
    intensities: torch.Tensor
    offsets: torch.Tensor
    sample_indices: torch.Tensor
    targets: TargetBatch = field(default_factory=TargetBatch.empty)

    def __post_init__(self) -> None:
        batch_size = int(self.sample_ids.numel())
        if self.sample_ids.ndim != 1 or self.offsets.shape != (batch_size + 1,):
            raise_validation_error(
                "RawSpectrumBatch", "sample_ids and offsets have incompatible shapes."
            )
        point_count = int(self.mass_values.numel())
        if (
            self.mass_values.ndim != 1
            or self.intensities.shape != (point_count,)
            or self.sample_indices.shape != (point_count,)
        ):
            raise_validation_error(
                "RawSpectrumBatch", "Packed raw tensors must share one point dimension."
            )
        if self.offsets.device.type == "cpu" and (
            int(self.offsets[0]) != 0 or int(self.offsets[-1]) != point_count
        ):
            raise_validation_error(
                "RawSpectrumBatch", "offsets must span every packed raw point."
            )

    @property
    def batch_size(self) -> int:
        """Return the number of spectra represented by the batch."""
        return int(self.sample_ids.numel())

    @property
    def device(self) -> torch.device:
        """Return the device storing packed spectral values."""
        return self.intensities.device

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "RawSpectrumBatch":
        """Move all numerical batch tensors to one device exactly once."""
        resolved = torch.device(device)
        return RawSpectrumBatch(
            sample_ids=self.sample_ids.to(resolved, non_blocking=non_blocking),
            mass_values=self.mass_values.to(resolved, non_blocking=non_blocking),
            intensities=self.intensities.to(resolved, non_blocking=non_blocking),
            offsets=self.offsets.to(resolved, non_blocking=non_blocking),
            sample_indices=self.sample_indices.to(resolved, non_blocking=non_blocking),
            targets=self.targets.to(resolved, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "RawSpectrumBatch":
        """Pin CPU tensors for asynchronous host-to-device transfer."""
        return RawSpectrumBatch(
            sample_ids=self.sample_ids.pin_memory(),
            mass_values=self.mass_values.pin_memory(),
            intensities=self.intensities.pin_memory(),
            offsets=self.offsets.pin_memory(),
            sample_indices=self.sample_indices.pin_memory(),
            targets=TargetBatch(
                values={
                    name: value.pin_memory()
                    for name, value in self.targets.values.items()
                },
                masks={
                    name: value.pin_memory()
                    for name, value in self.targets.masks.items()
                },
                schemas=self.targets.schemas,
            ),
        )


@dataclass(frozen=True)
class SpectrumBatch:
    """Store dense spectra, annotations, and optional named augmented views."""

    sample_ids: torch.Tensor
    spectra: torch.Tensor
    space: SpectrumSpace
    targets: TargetBatch = field(default_factory=TargetBatch.empty)
    views: Mapping[str, torch.Tensor] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.spectra.ndim != 2:
            raise_validation_error("SpectrumBatch", "spectra must have shape [B, F].")
        if self.sample_ids.shape != (self.spectra.shape[0],):
            raise_validation_error(
                "SpectrumBatch", "sample_ids must contain one identifier per spectrum."
            )
        if self.spectra.shape[1] != self.space.feature_count:
            raise_validation_error(
                "SpectrumBatch", "spectra feature count must match the mass axis."
            )
        for name, view in self.views.items():
            if view.shape != self.spectra.shape:
                raise_validation_error(
                    "SpectrumBatch", f"View '{name}' must match spectra shape."
                )

    @property
    def batch_size(self) -> int:
        """Return the logical number of source samples."""
        return int(self.spectra.shape[0])

    @property
    def device(self) -> torch.device:
        """Return the device storing dense spectra."""
        return self.spectra.device

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "SpectrumBatch":
        """Move spectra, annotations, views, and axis to one device."""
        resolved = torch.device(device)
        return SpectrumBatch(
            sample_ids=self.sample_ids.to(resolved, non_blocking=non_blocking),
            spectra=self.spectra.to(resolved, non_blocking=non_blocking),
            space=self.space.to(resolved, non_blocking=non_blocking),
            targets=self.targets.to(resolved, non_blocking=non_blocking),
            views={
                name: value.to(resolved, non_blocking=non_blocking)
                for name, value in self.views.items()
            },
            metadata=self.metadata,
        )

    def with_view(self, name: str, values: torch.Tensor) -> "SpectrumBatch":
        """Return a batch with one additional view without mutating source data."""
        views = dict(self.views)
        views[name] = values
        return SpectrumBatch(
            sample_ids=self.sample_ids,
            spectra=self.spectra,
            space=self.space,
            targets=self.targets,
            views=views,
            metadata=self.metadata,
        )

    def pin_memory(self) -> "SpectrumBatch":
        """Pin dense CPU tensors before an asynchronous CUDA transfer."""
        return SpectrumBatch(
            sample_ids=self.sample_ids.pin_memory(),
            spectra=self.spectra.pin_memory(),
            space=SpectrumSpace(
                mass_axis=self.space.mass_axis.pin_memory(),
                representation=self.space.representation,
                normalization=self.space.normalization,
                axis_unit=self.space.axis_unit,
            ),
            targets=TargetBatch(
                values={
                    name: value.pin_memory()
                    for name, value in self.targets.values.items()
                },
                masks={
                    name: value.pin_memory()
                    for name, value in self.targets.masks.items()
                },
                schemas=self.targets.schemas,
            ),
            views={name: value.pin_memory() for name, value in self.views.items()},
            metadata=self.metadata,
        )

    def model_input(self) -> torch.Tensor:
        """Return original spectra followed by named views for one model forward."""
        if not self.views:
            return self.spectra
        return torch.cat([self.spectra, *self.views.values()], dim=0)

    def __len__(self) -> int:
        """Expose the legacy tuple length during the migration period."""
        return 4 if self.targets.values else 2

    def __getitem__(self, index: int) -> Any:
        """Expose legacy positional fields without copying their tensors."""
        legacy = (
            self.sample_ids,
            self.spectra,
            self.targets.values,
            self.targets.masks,
        )
        if index >= len(self):
            raise IndexError(index)
        return legacy[index]


@dataclass(frozen=True)
class LatentBatch:
    """Store latent embeddings and the spectral space reconstructed from them."""

    sample_ids: torch.Tensor
    embeddings: torch.Tensor
    reconstruction_space: SpectrumSpace
    targets: TargetBatch = field(default_factory=TargetBatch.empty)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise_validation_error("LatentBatch", "embeddings must have shape [B, L].")
        if self.sample_ids.shape != (self.embeddings.shape[0],):
            raise_validation_error(
                "LatentBatch", "sample_ids must contain one identifier per embedding."
            )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "LatentBatch":
        """Move latent tensors and their reconstruction-space descriptor."""
        resolved = torch.device(device)
        return LatentBatch(
            sample_ids=self.sample_ids.to(resolved, non_blocking=non_blocking),
            embeddings=self.embeddings.to(resolved, non_blocking=non_blocking),
            reconstruction_space=self.reconstruction_space.to(
                resolved, non_blocking=non_blocking
            ),
            targets=self.targets.to(resolved, non_blocking=non_blocking),
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class InverseSpectrumBatch:
    """Store variable-length inverse-binned spectra without padding."""

    sample_ids: torch.Tensor
    mass_values: torch.Tensor
    intensities: torch.Tensor
    offsets: torch.Tensor
    source_space: SpectrumSpace
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        point_count = self.mass_values.numel()
        if self.mass_values.ndim != 1 or self.intensities.shape != (point_count,):
            raise_validation_error(
                "InverseSpectrumBatch", "Packed inverse tensors must share [N]."
            )
        if self.offsets.shape != (self.sample_ids.numel() + 1,):
            raise_validation_error(
                "InverseSpectrumBatch", "offsets must contain B + 1 boundaries."
            )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "InverseSpectrumBatch":
        """Move inverse spectra while retaining their dense source space."""
        resolved = torch.device(device)
        return InverseSpectrumBatch(
            sample_ids=self.sample_ids.to(resolved, non_blocking=non_blocking),
            mass_values=self.mass_values.to(resolved, non_blocking=non_blocking),
            intensities=self.intensities.to(resolved, non_blocking=non_blocking),
            offsets=self.offsets.to(resolved, non_blocking=non_blocking),
            source_space=self.source_space.to(resolved, non_blocking=non_blocking),
            metadata=self.metadata,
        )
