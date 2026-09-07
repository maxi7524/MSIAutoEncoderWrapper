"""Signal-based evidence for annotation-derived ion targets."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

NEGATIVE, POSITIVE, UNLABELLED, UNAVAILABLE = 0, 1, 2, -1


@dataclass(frozen=True)
class IonCatalogue:
    """Store stable ion identities and their possible coordinates on one axis.

    :param class_names: Formula/adduct identities in target order.
    :param bins: Nonempty coordinate tuples for each identity.
    :param feature_count: Width of the binned spectra.
    """

    class_names: tuple[str, ...]
    bins: tuple[tuple[int, ...], ...]
    feature_count: int

    def __post_init__(self) -> None:
        if isinstance(self.feature_count, bool) or not isinstance(self.feature_count, int) or self.feature_count < 1:
            raise ValueError("feature_count must be a positive integer.")
        if len(self.class_names) != len(self.bins) or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("Ion identities must be unique and aligned with coordinate groups.")
        if any(not bins or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= self.feature_count
                               for i in bins) for bins in self.bins):
            raise ValueError("Every ion needs nonempty coordinates inside the spectral axis.")

    @classmethod
    def from_dataset(cls, dataset: Any, target_field: str = "molecule") -> "IonCatalogue":
        """Resolve the existing dataset annotation map without reading spectra.

        :param dataset: Dataset exposing mapped annotations and target schemas.
        :param target_field: Ion target field.
        :return: Catalogue aligned exactly with the target columns.
        :raises ValueError: If an identity has no mapped spectral coordinate.
        """
        index = dataset.get_mapped_annotation_index()
        if index.coordinate_system != "binner":
            raise ValueError("Ion evidence requires annotation mapping to binner coordinates.")
        names = dataset.get_target_schemas()[target_field].class_names
        coordinates = {}
        for i, identity in enumerate(index.annotation_identities):
            coordinates["|".join(identity)] = tuple(
                int(v) for v in np.unique(index.coordinate_indices[index.annotation_indices == i])
            )
        bins = tuple(coordinates.get(name, ()) for name in names)
        if any(not values for values in bins):
            raise ValueError("Every ion target must have at least one mapped coordinate.")
        logger.info("Resolved signal evidence catalogue for %s ions.", len(names))
        return cls(tuple(names), bins, len(index.coordinate_axis))


@dataclass(frozen=True)
class SignalEvidencePolicy:
    """Classify missing annotations using a peak threshold after normalization.

    :param absolute_threshold: Minimum intensity in the model input scale.
    :param relative_threshold: Minimum fraction of the spectrum maximum.
    :param bin_radius: Number of neighbouring bins on either side.
    :raises ValueError: If thresholds or the radius are invalid.
    """

    absolute_threshold: float = 0.0
    relative_threshold: float = 0.01
    bin_radius: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.absolute_threshold) or self.absolute_threshold < 0:
            raise ValueError("absolute_threshold must be finite and nonnegative.")
        if not math.isfinite(self.relative_threshold) or not 0 <= self.relative_threshold <= 1:
            raise ValueError("relative_threshold must be in [0, 1].")
        if isinstance(self.bin_radius, bool) or not isinstance(self.bin_radius, int) or self.bin_radius < 0:
            raise ValueError("bin_radius must be a nonnegative integer.")

    def classify(
        self, spectra: torch.Tensor, targets: torch.Tensor,
        mask: torch.Tensor, catalogue: IonCatalogue,
    ) -> torch.Tensor:
        """Return P/N/U states, retaining positives and excluding unavailable labels.

        :param spectra: Nonnegative model inputs, shape ``(B, M)``.
        :param targets: Binary annotation indicators, shape ``(B, C)``.
        :param mask: Target availability, shape ``(B, C)``.
        :param catalogue: Ion positions in the same target order.
        :return: Integer states, shape ``(B, C)``; unavailable entries are -1.
        :raises ValueError: If tensors or coordinate dimensions disagree.
        """
        if spectra.ndim != 2 or spectra.shape[1] != catalogue.feature_count:
            raise ValueError("Spectra do not match the ion catalogue axis.")
        if targets.shape != (len(spectra), len(catalogue.bins)) or mask.shape != targets.shape:
            raise ValueError("Evidence targets and availability must have shape (B, C).")
        if not bool(torch.isfinite(spectra).all()) or bool((spectra < 0).any()):
            raise ValueError("Evidence requires finite nonnegative intensities.")
        # Measure local peak evidence without changing the original annotations
        pooled = F.max_pool1d(
            spectra.unsqueeze(1), 2 * self.bin_radius + 1,
            stride=1, padding=self.bin_radius,
        ).squeeze(1)  # (B, M)
        threshold = torch.maximum(
            spectra.amax(dim=1, keepdim=True) * self.relative_threshold,
            spectra.new_tensor(self.absolute_threshold),
        )  # (B, 1)
        signal = torch.stack([
            pooled[:, list(bins)].amax(dim=1) for bins in catalogue.bins
        ], dim=1) if catalogue.bins else spectra.new_empty((len(spectra), 0))  # (B, C)
        states = torch.where(signal > threshold, UNLABELLED, NEGATIVE)  # (B, C)
        states = torch.where(targets > 0.5, POSITIVE, states)  # (B, C)
        return states.masked_fill(~mask.bool(), UNAVAILABLE)
