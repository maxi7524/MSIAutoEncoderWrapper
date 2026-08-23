"""Compact spectrum-to-annotation index shared by annotation readers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectrumAnnotationIndex:
    """Store source-specific molecular identities and m/z values in CSR form.

    :param spectrum_ids: Sorted spectrum identifiers represented by the rows.
    :type spectrum_ids: numpy.ndarray
    :param spectrum_offsets: CSR offsets with one terminal value.
    :type spectrum_offsets: numpy.ndarray
    :param annotation_indices: Indices into ``annotation_identities``.
    :type annotation_indices: numpy.ndarray
    :param mz_values: Source-specific m/z value aligned with every entry. Missing
        reference masses are represented by ``NaN``.
    :type mz_values: numpy.ndarray
    :param annotation_identities: Canonical ``(formula, adduct)`` identities.
    :type annotation_identities: tuple[tuple[str, str], ...]
    """

    spectrum_ids: np.ndarray
    spectrum_offsets: np.ndarray
    annotation_indices: np.ndarray
    mz_values: np.ndarray
    annotation_identities: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        row_count = int(self.spectrum_ids.size)
        entry_count = int(self.annotation_indices.size)
        if self.spectrum_ids.ndim != 1 or self.spectrum_offsets.shape != (row_count + 1,):
            raise ValueError("SpectrumAnnotationIndex has incompatible row arrays.")
        if self.mz_values.shape != (entry_count,):
            raise ValueError("SpectrumAnnotationIndex entry arrays must be aligned.")
        if row_count and not bool(np.all(self.spectrum_ids[1:] > self.spectrum_ids[:-1])):
            raise ValueError("SpectrumAnnotationIndex spectrum_ids must be unique and sorted.")
        if int(self.spectrum_offsets[0]) != 0 or int(self.spectrum_offsets[-1]) != entry_count:
            raise ValueError("SpectrumAnnotationIndex offsets must span all entries.")

    def entry_slice(self, spectrum_id: int) -> slice:
        """Return the CSR entry slice for one spectrum identifier.

        :param spectrum_id: Stable source or merged spectrum identifier.
        :type spectrum_id: int
        :return: Entry slice, empty when the identifier is not represented.
        :rtype: slice
        """
        position = int(np.searchsorted(self.spectrum_ids, int(spectrum_id)))
        if position >= self.spectrum_ids.size or int(self.spectrum_ids[position]) != int(
            spectrum_id
        ):
            return slice(0, 0)
        return slice(
            int(self.spectrum_offsets[position]),
            int(self.spectrum_offsets[position + 1]),
        )

    def identities_for_spectrum(self, spectrum_id: int) -> tuple[tuple[str, str], ...]:
        """Return molecular identities present in one spectrum.

        :param spectrum_id: Stable spectrum identifier.
        :type spectrum_id: int
        :return: Formula/adduct identities in deterministic source order.
        :rtype: tuple[tuple[str, str], ...]
        """
        entry_slice = self.entry_slice(spectrum_id)
        return tuple(
            self.annotation_identities[int(index)]
            for index in self.annotation_indices[entry_slice]
        )

    def mz_for_spectrum(self, spectrum_id: int) -> np.ndarray:
        """Return the aligned source-specific m/z values for one spectrum.

        :param spectrum_id: Stable spectrum identifier.
        :type spectrum_id: int
        :return: Read-only view of aligned m/z values.
        :rtype: numpy.ndarray
        """
        return self.mz_values[self.entry_slice(spectrum_id)]


def build_annotation_index(
    spectrum_ids: list[int],
    entries: dict[int, list[tuple[tuple[str, str], float]]],
) -> SpectrumAnnotationIndex:
    """Build deterministic CSR arrays from already filtered reader entries.

    :param spectrum_ids: Spectrum rows to include, including unannotated rows.
    :type spectrum_ids: list[int]
    :param entries: Per-spectrum molecular identity and m/z records.
    :type entries: dict[int, list[tuple[tuple[str, str], float]]]
    :return: Compact annotation index.
    :rtype: SpectrumAnnotationIndex
    """
    ordered_spectra = np.asarray(sorted({int(value) for value in spectrum_ids}), dtype=np.int64)
    identities = tuple(
        sorted(
            {
                identity
                for spectrum_entries in entries.values()
                for identity, _ in spectrum_entries
            }
        )
    )
    identity_indices = {identity: index for index, identity in enumerate(identities)}
    offsets = np.zeros(ordered_spectra.size + 1, dtype=np.int64)
    for position, spectrum_id in enumerate(ordered_spectra):
        offsets[position + 1] = offsets[position] + len(entries.get(int(spectrum_id), ()))
    annotation_indices = np.empty(int(offsets[-1]), dtype=np.int32)
    mz_values = np.empty(int(offsets[-1]), dtype=np.float64)
    cursor = 0
    for spectrum_id in ordered_spectra:
        for identity, mz in entries.get(int(spectrum_id), ()):
            annotation_indices[cursor] = identity_indices[identity]
            mz_values[cursor] = float(mz)
            cursor += 1
    for array in (ordered_spectra, offsets, annotation_indices, mz_values):
        array.setflags(write=False)
    return SpectrumAnnotationIndex(
        spectrum_ids=ordered_spectra,
        spectrum_offsets=offsets,
        annotation_indices=annotation_indices,
        mz_values=mz_values,
        annotation_identities=identities,
    )
