"""Build global Torch candidate dictionaries from dataset candidate catalogues."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any, Mapping

import numpy as np
import torch

from ......utils.logger import get_custom_logger


logger = get_custom_logger(__name__)


def _candidate_ion_query(filters: Mapping[str, Any] | None) -> tuple[str, list[object]]:
    """Build the public candidate-reader filter query for local SQLite optimization."""
    filters = dict(filters or {})
    clauses: list[str] = []
    values: list[object] = []
    mapping = {
        "provider": "compound.provider",
        "adduct": "ion.adduct",
        "polarity": "ion.polarity",
        "formula": "ion.formula",
    }
    for key, column in mapping.items():
        if filters.get(key) is not None:
            clauses.append(f"{column} = ?")
            values.append(str(filters[key]))
    if filters.get("mz_min") is not None:
        clauses.append("ion.theoretical_mz >= ?")
        values.append(float(filters["mz_min"]))
    if filters.get("mz_max") is not None:
        clauses.append("ion.theoretical_mz <= ?")
        values.append(float(filters["mz_max"]))
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, values


def _sample_sqlite_candidate_ions(
    path: Path,
    *,
    filters: Mapping[str, Any] | None,
    binner: Any,
    candidate_limit: int,
    selection_seed: int,
) -> tuple[tuple[Mapping[str, object], ...], torch.Tensor] | None:
    """Sample valid catalogue ions without deserializing every database record.

    This is an implementation optimization for the local ``CandidateCatalogReader``
    backend. It preserves its public filtering and ordering semantics, while moving
    only ``candidate_ion_key`` and theoretical mass for the global population.
    """
    where, values = _candidate_ion_query(filters)
    identifier_query = (
        "SELECT ion.candidate_ion_key, ion.theoretical_mz "
        "FROM candidate_ions AS ion JOIN candidate_compounds AS compound USING (compound_key)"
        f"{where} ORDER BY ion.theoretical_mz, ion.candidate_ion_key"
    )
    try:
        with sqlite3.connect(path) as connection:
            identifier_rows = connection.execute(identifier_query, values).fetchall()
    except sqlite3.Error:
        return None
    if not identifier_rows:
        raise ValueError("Candidate catalogue query returned no ions.")
    masses = np.asarray([float(row[1]) for row in identifier_rows], dtype=np.float64)  # (C_all,)
    mapped_bins = np.asarray(binner.map_mass_values_to_bins(masses), dtype=np.int64)  # (C_all,)
    axis_size = len(binner.GetXAxis())
    valid_indices = torch.as_tensor(
        np.flatnonzero((mapped_bins >= 0) & (mapped_bins < axis_size)),
        dtype=torch.long,
    )  # (C_valid,)
    if valid_indices.numel() == 0:
        raise ValueError("No candidate ions map onto the active binner axis.")
    if candidate_limit > valid_indices.numel():
        raise ValueError(
            "candidate_limit must be between one and the count of ions mapped to the axis."
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(selection_seed)
    sampled_positions = torch.randperm(
        valid_indices.numel(), generator=generator, device="cpu"
    )[:candidate_limit]  # (C_subset,)
    selected_indices = valid_indices.index_select(0, sampled_positions)  # (C_subset,)
    selected_keys = [str(identifier_rows[index][0]) for index in selected_indices.tolist()]
    placeholders = ", ".join("?" for _ in selected_keys)
    record_query = f"""
        SELECT ion.*, compound.provider, compound.provider_version,
               compound.identifier, compound.name
        FROM candidate_ions AS ion
        JOIN candidate_compounds AS compound USING (compound_key)
        WHERE ion.candidate_ion_key IN ({placeholders})
    """
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        records_by_key = {
            str(row["candidate_ion_key"]): dict(row)
            for row in connection.execute(record_query, selected_keys).fetchall()
        }
    selected_records = tuple(records_by_key[key] for key in selected_keys)
    selected_bins = torch.as_tensor(mapped_bins[selected_indices.numpy()], dtype=torch.long)  # (C_subset,)
    return selected_records, selected_bins


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
        candidate_limit: int | None = None,
        selection_seed: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "GlobalCandidateDictionary":
        """Build a global dictionary from all catalogue ions on one binner axis.

        :param candidate_catalog: Reader exposing ``get_candidate_ions(filters=...)``.
        :param binner: Active binner exposing ``GetXAxis`` and
            ``map_mass_values_to_bins``.
        :param filters: Optional catalogue filters applied before dictionary
            construction.
        :param candidate_limit: Optional deterministic limit applied after the
            catalogue query and binner-axis filtering. It prevents exploratory
            dense dictionaries from allocating the complete global universe.
        :param selection_seed: Required local Torch seed when ``candidate_limit``
            is set. It makes the selected candidate-ion records reproducible.
        :param dtype: Floating-point dtype of the dictionary matrix.
        :type candidate_catalog: Any
        :type binner: Any
        :type filters: collections.abc.Mapping[str, typing.Any] | None
        :type dtype: torch.dtype
        :return: Dataset-wide candidate dictionary, or a reproducible subset for
            an explicitly bounded exploratory analysis.
        :rtype: GlobalCandidateDictionary
        :raises ValueError: If no candidate ions map onto the active axis.
        """
        get_ions = getattr(candidate_catalog, "get_candidate_ions", None)
        map_masses = getattr(binner, "map_mass_values_to_bins", None)
        get_axis = getattr(binner, "GetXAxis", None)
        if not callable(get_ions) or not callable(map_masses) or not callable(get_axis):
            raise ValueError("candidate catalog and binner do not expose the required API.")
        if candidate_limit is not None and selection_seed is None:
            raise ValueError("selection_seed is required when candidate_limit is set.")

        mass_axis = torch.as_tensor(np.asarray(get_axis()), dtype=dtype)  # (M,)
        catalogue_path = getattr(candidate_catalog, "path", None)
        sqlite_sample = None
        if candidate_limit is not None and selection_seed is not None and catalogue_path is not None:
            path = Path(catalogue_path)
            if path.is_file():
                # REMARK: The global catalogue may contain nearly one million ions.
                # The exploratory baseline must not deserialize all records just to
                # render a seeded hundred-column dense Torch matrix.
                sqlite_sample = _sample_sqlite_candidate_ions(
                    path,
                    filters=filters,
                    binner=binner,
                    candidate_limit=candidate_limit,
                    selection_seed=selection_seed,
                )
        if sqlite_sample is not None:
            selected_records, selected_bins = sqlite_sample
        else:
            records = tuple(get_ions(filters=filters))
            if not records:
                raise ValueError("Candidate catalogue query returned no ions.")
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
        if candidate_limit is not None and sqlite_sample is None:
            if candidate_limit < 1 or candidate_limit > selected_bins.numel():
                raise ValueError(
                    "candidate_limit must be between one and the count of ions mapped to the axis."
                )
            selection_generator = torch.Generator(device="cpu")
            selection_generator.manual_seed(selection_seed)
            selected_indices = torch.randperm(
                selected_bins.numel(),
                generator=selection_generator,
                device="cpu",
            )[:candidate_limit]  # (C_subset,)
            selected_bins = selected_bins.index_select(0, selected_indices)  # (C_subset,)
            selected_records = tuple(
                selected_records[index] for index in selected_indices.tolist()
            )
        matrix = torch.zeros(
            (mass_axis.numel(), selected_bins.numel()), dtype=dtype
        )  # (M, C)
        matrix[selected_bins, torch.arange(selected_bins.numel())] = 1.0
        logger.info(
            "Built candidate dictionary with %s features and %s candidate ions.",
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
