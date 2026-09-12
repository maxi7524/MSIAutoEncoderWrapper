"""Abstract sources of labelled peak coordinates for synthetic spectra."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..annotation_evidence import IonCatalogue


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
