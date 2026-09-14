"""Build global Torch candidate dictionaries from dataset candidate catalogues."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from ...utils.logger import get_custom_logger


logger = get_custom_logger(__name__)


@dataclass(frozen=True)
class GlobalCandidateDictionary:
    """One dataset-wide candidate dictionary aligned with a binner axis.

    Each candidate-ion record in the dataset-filtered catalogue becomes one
    dictionary column. The initial baseline renders its theoretical ``m/z`` as
    a unit impulse at the bin supplied by the active binner. The candidate
    records retain formulas, adducts, source provenance, and catalogue IDs so
    later profile models can replace the rendering without changing identity.

    :param matrix: Dictionary matrix with shape ``(M, C)``.
    :param candidate_ions: Catalogue records in exact column order.
    :param mass_axis: Active binner mass axis with shape ``(M,)``.
    :type matrix: torch.Tensor
    :type candidate_ions: tuple[collections.abc.Mapping[str, object], ...]
    :type mass_axis: torch.Tensor
    :raises ValueError: If matrix, axis, or provenance dimensions disagree.
    """

    matrix: torch.Tensor
    candidate_ions: tuple[Mapping[str, object], ...]
    mass_axis: torch.Tensor

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            raise ValueError("matrix must have shape (M, C).")
        if self.mass_axis.ndim != 1 or self.mass_axis.shape[0] != self.matrix.shape[0]:
            raise ValueError("mass_axis must have shape (M,) matching matrix.")
        if self.matrix.shape[1] != len(self.candidate_ions):
            raise ValueError("candidate metadata must contain one record per column.")
        if self.matrix.shape[1] == 0:
            raise ValueError("dictionary must contain at least one candidate ion.")
        if not bool(torch.isfinite(self.matrix).all()) or bool((self.matrix < 0).any()):
            raise ValueError("dictionary matrix must be finite and non-negative.")
        if bool((self.matrix.sum(dim=0) <= 0).any()):
            raise ValueError("every dictionary column must contain positive signal.")

    @property
    def feature_count(self) -> int:
        """Return the binner feature count ``M``."""
        return int(self.matrix.shape[0])

    @property
    def candidate_count(self) -> int:
        """Return the global candidate count ``C``."""
        return int(self.matrix.shape[1])

    @classmethod
    def from_candidate_catalog(
        cls,
        candidate_catalog: Any,
        binner: Any,
        *,
        filters: Mapping[str, Any] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "GlobalCandidateDictionary":
        """Build a global dictionary from all catalogue ions on one binner axis.

        :param candidate_catalog: Reader exposing ``get_candidate_ions(filters=...)``.
        :param binner: Active binner exposing ``GetXAxis`` and
            ``map_mass_values_to_bins``.
        :param filters: Optional catalogue filters applied before dictionary
            construction.
        :param dtype: Floating-point dtype of the dictionary matrix.
        :type candidate_catalog: Any
        :type binner: Any
        :type filters: collections.abc.Mapping[str, typing.Any] | None
        :type dtype: torch.dtype
        :return: Dataset-wide candidate dictionary.
        :rtype: GlobalCandidateDictionary
        :raises ValueError: If no candidate ions map onto the active axis.
        """
        get_ions = getattr(candidate_catalog, "get_candidate_ions", None)
        map_masses = getattr(binner, "map_mass_values_to_bins", None)
        get_axis = getattr(binner, "GetXAxis", None)
        if not callable(get_ions) or not callable(map_masses) or not callable(get_axis):
            raise ValueError("candidate catalog and binner do not expose the required API.")

        records = tuple(get_ions(filters=filters))
        if not records:
            raise ValueError("Candidate catalogue query returned no ions.")
        mass_axis = torch.as_tensor(np.asarray(get_axis()), dtype=dtype)  # (M,)
        masses = np.asarray(
            [float(record["theoretical_mz"]) for record in records],
            dtype=np.float64,
        )  # (C_all,)
        mapped_bins = np.asarray(map_masses(masses), dtype=np.int64)  # (C_all,)
        valid = (mapped_bins >= 0) & (mapped_bins < mass_axis.numel())
        if not valid.any():
            raise ValueError("No candidate ions map onto the active binner axis.")

        selected_records = tuple(
            dict(record) for record, is_valid in zip(records, valid, strict=True) if is_valid
        )
        selected_bins = torch.as_tensor(mapped_bins[valid], dtype=torch.long)  # (C,)
        matrix = torch.zeros(
            (mass_axis.numel(), selected_bins.numel()), dtype=dtype
        )  # (M, C)
        matrix[selected_bins, torch.arange(selected_bins.numel())] = 1.0
        logger.info(
            "Built global candidate dictionary with %s features and %s candidate ions.",
            matrix.shape[0],
            matrix.shape[1],
        )
        return cls(matrix=matrix, candidate_ions=selected_records, mass_axis=mass_axis)

    def to(self, device: torch.device | str) -> "GlobalCandidateDictionary":
        """Move the dictionary matrix and mass axis without changing provenance.

        :param device: Target Torch device.
        :type device: torch.device | str
        :return: Dictionary stored on ``device``.
        :rtype: GlobalCandidateDictionary
        """
        resolved_device = torch.device(device)
        return GlobalCandidateDictionary(
            matrix=self.matrix.to(resolved_device),  # (M, C)
            candidate_ions=self.candidate_ions,
            mass_axis=self.mass_axis.to(resolved_device),  # (M,)
        )

    def select_columns(self, indices: torch.Tensor) -> "GlobalCandidateDictionary":
        """Create a candidate subdictionary while preserving column provenance.

        :param indices: Unique global candidate-column indices with shape ``(S,)``.
        :type indices: torch.Tensor
        :return: Dictionary with matrix shape ``(M, S)``.
        :rtype: GlobalCandidateDictionary
        :raises ValueError: If indices are not a non-empty unique valid selection.
        """
        if indices.ndim != 1 or indices.numel() == 0:
            raise ValueError("indices must be a non-empty one-dimensional tensor.")
        indices = indices.to(device=self.matrix.device, dtype=torch.long)  # (S,)
        if bool((indices < 0).any()) or bool((indices >= self.candidate_count).any()):
            raise ValueError("indices contain an invalid global candidate column.")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("indices must not contain duplicate candidate columns.")
        selected_metadata = tuple(self.candidate_ions[index] for index in indices.cpu().tolist())
        return GlobalCandidateDictionary(
            matrix=self.matrix.index_select(1, indices),  # (M, S)
            candidate_ions=selected_metadata,
            mass_axis=self.mass_axis,
        )
