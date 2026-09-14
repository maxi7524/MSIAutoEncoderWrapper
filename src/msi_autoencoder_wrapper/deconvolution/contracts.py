"""Typed Torch contracts shared by deconvolution data and models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import torch

if TYPE_CHECKING:
    from .data.dictionary import GlobalCandidateDictionary


@dataclass(frozen=True)
class DeconvolutionBatch:
    """One batch of spectra generated from or evaluated against one dictionary.

    :param spectra: Dense input spectra with shape ``(B, M)``.
    :param dictionary: Shared global candidate dictionary with matrix shape
        ``(M, C)``.
    :param abundances: Optional exact candidate abundances with shape ``(B, C)``.
    :param presence: Optional exact candidate support with shape ``(B, C)``.
    :param metadata: Reproducibility metadata for the generated population.
    :type spectra: torch.Tensor
    :type dictionary: GlobalCandidateDictionary
    :type abundances: torch.Tensor | None
    :type presence: torch.Tensor | None
    :type metadata: collections.abc.Mapping[str, object]
    :raises ValueError: If the tensors are incompatible with the dictionary.
    """

    spectra: torch.Tensor
    dictionary: "GlobalCandidateDictionary"
    abundances: torch.Tensor | None = None
    presence: torch.Tensor | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.spectra.ndim != 2:
            raise ValueError("spectra must have shape (B, M).")
        batch_size, feature_count = self.spectra.shape
        if feature_count != self.dictionary.feature_count:
            raise ValueError("spectra and dictionary feature counts disagree.")
        expected_targets = (batch_size, self.dictionary.candidate_count)
        if self.abundances is not None and self.abundances.shape != expected_targets:
            raise ValueError("abundances must have shape (B, C).")
        if self.presence is not None and self.presence.shape != expected_targets:
            raise ValueError("presence must have shape (B, C).")
        if self.abundances is not None and bool((self.abundances < 0).any()):
            raise ValueError("abundances must be non-negative.")

    @property
    def batch_size(self) -> int:
        """Return the number of spectra in the batch."""
        return int(self.spectra.shape[0])

    def to(self, device: torch.device | str) -> "DeconvolutionBatch":
        """Move all tensors, including the global dictionary, to one device.

        :param device: Target Torch device.
        :type device: torch.device | str
        :return: Batch whose numerical tensors are stored on ``device``.
        :rtype: DeconvolutionBatch
        """
        resolved_device = torch.device(device)
        return DeconvolutionBatch(
            spectra=self.spectra.to(resolved_device),  # (B, M)
            dictionary=self.dictionary.to(resolved_device),
            abundances=(
                self.abundances.to(resolved_device)  # (B, C)
                if self.abundances is not None
                else None
            ),
            presence=(
                self.presence.to(resolved_device)  # (B, C)
                if self.presence is not None
                else None
            ),
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class DeconvolutionResult:
    """Outputs of one non-negative decomposition against a global dictionary.

    :param abundances: Estimated non-negative candidate abundances, shape ``(B, C)``.
    :param reconstruction: Dictionary reconstruction, shape ``(B, M)``.
    :param residual: ``spectra - reconstruction``, shape ``(B, M)``.
    :param objective: Per-spectrum squared reconstruction objective, shape ``(B,)``.
    :type abundances: torch.Tensor
    :type reconstruction: torch.Tensor
    :type residual: torch.Tensor
    :type objective: torch.Tensor
    """

    abundances: torch.Tensor
    reconstruction: torch.Tensor
    residual: torch.Tensor
    objective: torch.Tensor
