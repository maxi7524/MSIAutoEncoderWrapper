"""Immutable descriptions of numerical spaces used by MSI tensors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch

from ..utils.exceptions import raise_validation_error


@dataclass(frozen=True)
class SpectrumSpace:
    """Describe a shared spectral axis and representation semantics.

    :param mass_axis: Strictly increasing one-dimensional m/z coordinates.
    :type mass_axis: torch.Tensor
    :param representation: Spectral representation associated with the axis.
    :type representation: Literal["binned", "reconstruction"]
    :param normalization: Intensity normalization applied to every spectrum.
    :type normalization: Literal["none", "tic", "max", "l2"]
    :param axis_unit: Unit assigned to the mass axis.
    :type axis_unit: str
    """

    mass_axis: torch.Tensor
    representation: Literal["binned", "reconstruction"] = "binned"
    normalization: str = "none"
    axis_unit: str = "m/z"

    def __post_init__(self) -> None:
        axis = self.mass_axis
        if not isinstance(axis, torch.Tensor) or axis.ndim != 1 or axis.numel() < 1:
            raise_validation_error(
                "SpectrumSpace", "mass_axis must be a non-empty one-dimensional tensor."
            )
        if axis.is_floating_point() and not bool(torch.isfinite(axis).all()):
            raise_validation_error("SpectrumSpace", "mass_axis must contain finite values.")
        if axis.numel() > 1 and not bool(torch.all(axis[1:] > axis[:-1])):
            raise_validation_error(
                "SpectrumSpace", "mass_axis values must be strictly increasing."
            )

    @property
    def feature_count(self) -> int:
        """Return the number of coordinates in the spectral space."""
        return int(self.mass_axis.numel())

    @property
    def device(self) -> torch.device:
        """Return the device storing the mass axis."""
        return self.mass_axis.device

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "SpectrumSpace":
        """Return a space whose axis resides on ``device``.

        :param device: Destination device.
        :type device: torch.device | str
        :param non_blocking: Request an asynchronous transfer when supported.
        :type non_blocking: bool
        :return: Original space when already colocated, otherwise a new descriptor.
        :rtype: SpectrumSpace
        """
        resolved = torch.device(device)
        if self.mass_axis.device == resolved:
            return self
        return replace(
            self,
            mass_axis=self.mass_axis.to(resolved, non_blocking=non_blocking),
        )

    def as_reconstruction(self) -> "SpectrumSpace":
        """Return the same axis marked as a reconstruction representation."""
        if self.representation == "reconstruction":
            return self
        return replace(self, representation="reconstruction")
