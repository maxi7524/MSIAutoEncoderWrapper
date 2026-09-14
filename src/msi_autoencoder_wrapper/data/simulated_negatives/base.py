"""Abstract contract for deriving reliable-negative masks from dataset data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import torch


class SimulatedNegativeStrategy(ABC):
    """Derive one reliable-negative mask aligned with annotation targets.

    A strategy receives canonical, pre-training-mask annotation values. It must
    never infer a reliable negative for an annotated positive column.
    """

    @abstractmethod
    def build_mask(
        self,
        dataset: Any,
        target_field: str,
        source_indices: Sequence[int],
        targets: torch.Tensor,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        """Return a Boolean ``N_sim`` mask with shape ``(N, C)``.

        :param dataset: Dataset exposing model-input spectra and target schemas.
        :type dataset: Any
        :param target_field: Binary annotation-derived target name.
        :type target_field: str
        :param source_indices: Stable reader-level spectrum identifiers.
        :type source_indices: Sequence[int]
        :param targets: Canonical binary annotations with shape ``(N, C)``.
        :type targets: torch.Tensor
        :param availability: Candidate availability with shape ``(N, C)``.
        :type availability: torch.Tensor
        :return: Reliable-negative mask with shape ``(N, C)``.
        :rtype: torch.Tensor
        """
