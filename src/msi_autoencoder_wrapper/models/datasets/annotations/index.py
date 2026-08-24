"""Sparse dataset-owned annotation indices aligned to one coordinate system."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MappedSpectrumAnnotationIndex:
    """Store annotation identities and mapped coordinates in compact CSR arrays."""

    spectrum_ids: np.ndarray
    spectrum_offsets: np.ndarray
    annotation_indices: np.ndarray
    coordinate_indices: np.ndarray
    annotation_identities: tuple[tuple[str, str], ...]
    coordinate_axis: np.ndarray
    coordinate_system: str

    def __post_init__(self) -> None:
        row_count = int(self.spectrum_ids.size)
        entry_count = int(self.annotation_indices.size)
        if self.spectrum_offsets.shape != (row_count + 1,):
            raise ValueError("MappedSpectrumAnnotationIndex has invalid offsets.")
        if self.coordinate_indices.shape != (entry_count,):
            raise ValueError("MappedSpectrumAnnotationIndex coordinates are misaligned.")
        if self.coordinate_system not in {"binner", "annotation"}:
            raise ValueError("MappedSpectrumAnnotationIndex has an unknown coordinate system.")

    def entry_slice(self, spectrum_id: int) -> slice:
        """Return one CSR slice, or an empty slice for an unannotated spectrum."""
        position = int(np.searchsorted(self.spectrum_ids, int(spectrum_id)))
        if position >= self.spectrum_ids.size or int(self.spectrum_ids[position]) != int(spectrum_id):
            return slice(0, 0)
        return slice(int(self.spectrum_offsets[position]), int(self.spectrum_offsets[position + 1]))

    def has_annotations(self, spectrum_id: int) -> bool:
        """Return whether the spectrum retains at least one mapped annotation."""
        entry_slice = self.entry_slice(spectrum_id)
        return entry_slice.stop > entry_slice.start

    def identities_for_spectrum(self, spectrum_id: int) -> tuple[tuple[str, str], ...]:
        """Return retained annotation identities for one spectrum."""
        entry_slice = self.entry_slice(spectrum_id)
        return tuple(
            self.annotation_identities[int(index)]
            for index in self.annotation_indices[entry_slice]
        )

    def coordinates_for_spectrum(self, spectrum_id: int) -> np.ndarray:
        """Return mapped coordinate indices for one spectrum."""
        return self.coordinate_indices[self.entry_slice(spectrum_id)]
