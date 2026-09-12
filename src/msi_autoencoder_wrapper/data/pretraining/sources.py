"""Abstract sources of labelled peak coordinates for synthetic spectra."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from ..annotation_evidence import IonCatalogue


class CandidateCatalogPeakSourceError(ValueError):
    """Raised when a candidate catalogue cannot define synthetic coordinates."""


class SyntheticPeakSource(ABC):
    """Provide labelled peak coordinates on one already binned spectral axis.

    Implementations may derive these coordinates from observed annotations,
    external candidate ions, or another curated source.  The synthetic sampler
    intentionally depends only on this contract, not on a dataset annotation
    index.
    """

    @property
    @abstractmethod
    def feature_count(self) -> int:
        """Return the number of bins on the synthetic spectral axis."""

    @property
    @abstractmethod
    def class_names(self) -> tuple[str, ...]:
        """Return target names in the exact model target-column order."""

    @property
    @abstractmethod
    def bins(self) -> tuple[tuple[int, ...], ...]:
        """Return nonempty candidate bin coordinates for every target class."""

    def get_label_metadata(self, label_index: int) -> Mapping[str, Any]:
        """Return optional immutable provenance for one generated label.

        :param label_index: Position in :attr:`class_names`.
        :type label_index: int
        :returns: Empty metadata when the source has no external provenance.
        :rtype: collections.abc.Mapping[str, typing.Any]
        """
        if label_index < 0 or label_index >= len(self.class_names):
            raise IndexError("Synthetic label index is outside the peak source.")
        return {}


class CataloguePeakSource(SyntheticPeakSource):
    """Adapt the current annotation-derived :class:`IonCatalogue` contract."""

    def __init__(self, catalogue: IonCatalogue) -> None:
        self.catalogue = catalogue

    @property
    def feature_count(self) -> int:
        """Return the binned feature count of the wrapped catalogue."""
        return self.catalogue.feature_count

    @property
    def class_names(self) -> tuple[str, ...]:
        """Return catalogue identities in their target order."""
        return self.catalogue.class_names

    @property
    def bins(self) -> tuple[tuple[int, ...], ...]:
        """Return catalogue bin coordinates in their target order."""
        return self.catalogue.bins


class CandidateCatalogPeakSource(SyntheticPeakSource):
    """Map filtered database candidate ions onto one active binner axis.

    Candidate labels preserve every distinct theoretical ``m/z``. Equivalent
    records from multiple providers are merged only when formula, adduct, and
    theoretical mass are identical; their source and class provenance remains
    attached to the resulting label.

    :param candidate_catalog: Open candidate SQLite reader.
    :type candidate_catalog: Any
    :param binner: Active binner exposing ``GetXAxis`` and
        ``map_mass_values_to_bins``.
    :type binner: Any
    :param filters: Candidate-ion filters accepted by the catalogue reader.
    :type filters: collections.abc.Mapping[str, typing.Any] | None
    :param allowed_labels: Optional formula/adduct labels allowed by the active
        model target schema.
    :type allowed_labels: collections.abc.Sequence[str] | None
    :param chemical_classes: Optional class names; a candidate must have at
        least one matching provider class.
    :type chemical_classes: collections.abc.Sequence[str] | None
    """

    def __init__(
        self,
        candidate_catalog: Any,
        binner: Any,
        *,
        filters: Mapping[str, Any] | None = None,
        allowed_labels: Sequence[str] | None = None,
        chemical_classes: Sequence[str] | None = None,
    ) -> None:
        mapper = getattr(binner, "map_mass_values_to_bins", None)
        if not callable(mapper):
            raise CandidateCatalogPeakSourceError(
                "Candidate synthesis requires binner.map_mass_values_to_bins()."
            )
        candidate_ions = candidate_catalog.get_candidate_ions(filters=filters)
        if not candidate_ions:
            raise CandidateCatalogPeakSourceError("Candidate catalogue query returned no ions.")
        classes_by_compound = {
            str(record["compound_key"]): {
                str(item["name"])
                for item in record.get("classes", ())
                if item.get("name")
            }
            for record in candidate_catalog.get_candidates()
        }
        permitted_labels = set(allowed_labels) if allowed_labels is not None else None
        requested_classes = set(chemical_classes or ())

        # Candidate normalization and exact binner mapping
        ## The m/z-to-bin mapping is delegated to the active binner so synthetic
        ## coordinates use precisely the same bin intervals as real spectra.
        mapped_bins = np.asarray(
            mapper(
                np.asarray(
                    [float(record["theoretical_mz"]) for record in candidate_ions],
                    dtype=np.float64,
                )
            ),
            dtype=np.int64,
        )
        grouped: dict[tuple[str, str, float], dict[str, Any]] = {}
        for record, bin_index in zip(candidate_ions, mapped_bins, strict=True):
            if bin_index < 0:
                continue
            formula = str(record["formula"])
            adduct = str(record["adduct"])
            base_label = f"{formula}|{adduct}"
            if permitted_labels is not None and base_label not in permitted_labels:
                continue
            compound_key = str(record["compound_key"])
            classes = classes_by_compound.get(compound_key, set())
            if requested_classes and not classes.intersection(requested_classes):
                continue
            theoretical_mz = float(record["theoretical_mz"])
            key = (formula, adduct, theoretical_mz)
            group = grouped.setdefault(
                key,
                {
                    "base_label": base_label,
                    "bin": int(bin_index),
                    "formula": formula,
                    "adduct": adduct,
                    "theoretical_mz": theoretical_mz,
                    "charge": int(record["charge"]),
                    "candidate_ion_keys": set(),
                    "compound_keys": set(),
                    "providers": set(),
                    "chemical_classes": set(),
                },
            )
            group["candidate_ion_keys"].add(str(record["candidate_ion_key"]))
            group["compound_keys"].add(compound_key)
            group["providers"].add(str(record["provider"]))
            group["chemical_classes"].update(classes)
        if not grouped:
            raise CandidateCatalogPeakSourceError(
                "No candidate ion remains after axis, label, and class filtering."
            )

        # Stable label creation
        ## Formula/adduct is retained for compatibility with current ion heads.
        ## A mass suffix is added only if one formula/adduct unexpectedly maps
        ## to multiple theoretical masses, preventing accidental label merging.
        groups = sorted(
            grouped.values(),
            key=lambda value: (value["theoretical_mz"], value["base_label"]),
        )
        multiplicity: dict[str, int] = defaultdict(int)
        for group in groups:
            multiplicity[group["base_label"]] += 1
        self._metadata = tuple(
            {
                "label": (
                    group["base_label"]
                    if multiplicity[group["base_label"]] == 1
                    else f"{group['base_label']}|mz={group['theoretical_mz']:.8f}"
                ),
                "formula": group["formula"],
                "adduct": group["adduct"],
                "charge": group["charge"],
                "theoretical_mz": group["theoretical_mz"],
                "candidate_ion_keys": tuple(sorted(group["candidate_ion_keys"])),
                "compound_keys": tuple(sorted(group["compound_keys"])),
                "providers": tuple(sorted(group["providers"])),
                "chemical_classes": tuple(sorted(group["chemical_classes"])),
            }
            for group in groups
        )
        self._class_names = tuple(item["label"] for item in self._metadata)
        self._bins = tuple((int(group["bin"]),) for group in groups)
        self._feature_count = int(len(np.asarray(binner.GetXAxis())))

    @property
    def feature_count(self) -> int:
        """Return the active binner feature count."""
        return self._feature_count

    @property
    def class_names(self) -> tuple[str, ...]:
        """Return stable formula/adduct candidate labels."""
        return self._class_names

    @property
    def bins(self) -> tuple[tuple[int, ...], ...]:
        """Return exact candidate m/z bins in candidate-label order."""
        return self._bins

    def get_label_metadata(self, label_index: int) -> Mapping[str, Any]:
        """Return provider and chemistry provenance for one candidate label."""
        if label_index < 0 or label_index >= len(self._metadata):
            raise IndexError("Synthetic candidate label index is outside the source.")
        return self._metadata[label_index]
