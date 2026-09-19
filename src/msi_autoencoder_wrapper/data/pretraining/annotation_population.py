"""Train-split annotation records used by synthetic annotation strategies."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from torch.utils.data import Subset


@dataclass(frozen=True)
class AnnotationPeakRecord:
    """One binned peak with the complete annotation set of one source spectrum.

    :param spectrum_id: Immutable identifier of the source pixel or spectrum.
    :type spectrum_id: int
    :param bin_index: Coordinate on the active binned mass axis.
    :type bin_index: int
    :param label_indices: Positive molecular target columns at this coordinate.
    :type label_indices: tuple[int, ...]
    """

    spectrum_id: int
    bin_index: int
    label_indices: tuple[int, ...]


class AnnotationPopulation:
    """Immutable train-only pixel/bin annotation population.

    The population keeps the source-pixel context that is intentionally lost by
    :class:`IonCatalogue`. A bin without an entry for a selected spectrum has an
    empty label tuple and is therefore a complete synthetic negative.
    """

    def __init__(
        self,
        *,
        feature_count: int,
        spectrum_ids: Iterable[int],
        records: Iterable[AnnotationPeakRecord],
    ) -> None:
        if isinstance(feature_count, bool) or not isinstance(feature_count, int) or feature_count < 1:
            raise ValueError("feature_count must be a positive integer.")
        normalized_spectrum_ids = tuple(sorted({int(value) for value in spectrum_ids}))
        if not normalized_spectrum_ids:
            raise ValueError("Annotation population requires at least one train spectrum.")
        records_by_coordinate: dict[tuple[int, int], AnnotationPeakRecord] = {}
        records_by_label: dict[int, list[AnnotationPeakRecord]] = defaultdict(list)
        overlap_records: list[AnnotationPeakRecord] = []
        for record in records:
            if record.spectrum_id not in normalized_spectrum_ids:
                raise ValueError("Annotation record is outside the configured train population.")
            if record.bin_index < 0 or record.bin_index >= feature_count:
                raise ValueError("Annotation record bin is outside the active axis.")
            labels = tuple(sorted({int(value) for value in record.label_indices}))
            if not labels:
                raise ValueError("Stored annotation records must contain at least one positive label.")
            normalized = AnnotationPeakRecord(record.spectrum_id, record.bin_index, labels)
            key = (normalized.spectrum_id, normalized.bin_index)
            if key in records_by_coordinate:
                raise ValueError("Annotation population contains duplicate pixel/bin records.")
            records_by_coordinate[key] = normalized
            for label in labels:
                records_by_label[label].append(normalized)
            if len(labels) > 1:
                overlap_records.append(normalized)
        self.feature_count = feature_count
        self.spectrum_ids = normalized_spectrum_ids
        self._records_by_coordinate = records_by_coordinate
        self._records_by_label = {
            label: tuple(values) for label, values in records_by_label.items()
        }
        self._overlap_records = tuple(overlap_records)

    @classmethod
    def from_dataset(cls, dataset: Any, target_field: str = "molecule") -> "AnnotationPopulation":
        """Build one population from annotations belonging only to the train split.

        :param dataset: Real dataset exposing partitions, target schemas, and a
            mapped annotation index.
        :type dataset: Any
        :param target_field: Multi-label target schema used for molecular labels.
        :type target_field: str
        :return: Pixel/bin records grouped from the train split only.
        :rtype: AnnotationPopulation
        """
        train_partition = dataset.create_partitions().train
        spectrum_ids = _source_indices(train_partition)
        index = dataset.get_mapped_annotation_index()
        if index.coordinate_system != "binner":
            raise ValueError("Annotation synthesis requires binner-mapped annotations.")
        target_names = dataset.get_target_schemas()[target_field].class_names
        target_indices = {name: position for position, name in enumerate(target_names)}
        records: list[AnnotationPeakRecord] = []
        for spectrum_id in spectrum_ids:
            entry_slice = index.entry_slice(spectrum_id)
            grouped: dict[int, set[int]] = defaultdict(set)
            for annotation_index, coordinate in zip(
                index.annotation_indices[entry_slice],
                index.coordinate_indices[entry_slice],
                strict=True,
            ):
                identity = "|".join(index.annotation_identities[int(annotation_index)])
                target_index = target_indices.get(identity)
                if target_index is not None:
                    grouped[int(coordinate)].add(target_index)
            records.extend(
                AnnotationPeakRecord(spectrum_id, coordinate, tuple(sorted(labels)))
                for coordinate, labels in grouped.items()
            )
        return cls(
            feature_count=len(index.coordinate_axis),
            spectrum_ids=spectrum_ids,
            records=records,
        )

    @property
    def positive_labels(self) -> tuple[int, ...]:
        """Return train-observed molecular columns in deterministic order."""
        return tuple(sorted(self._records_by_label))

    @property
    def overlap_records(self) -> tuple[AnnotationPeakRecord, ...]:
        """Return records whose source pixel/bin has more than one label."""
        return self._overlap_records

    def labels_for(self, spectrum_id: int, bin_index: int) -> tuple[int, ...]:
        """Return complete labels for one pixel/bin, or empty labels when absent."""
        record = self._records_by_coordinate.get((int(spectrum_id), int(bin_index)))
        return () if record is None else record.label_indices

    def records_for_label(self, label_index: int) -> tuple[AnnotationPeakRecord, ...]:
        """Return train records containing one requested positive class."""
        return self._records_by_label.get(int(label_index), ())

    def rare_labels(self, fraction: float) -> tuple[int, ...]:
        """Return the least frequent positive classes under a deterministic cutoff."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError("rare_fraction must belong to (0, 1].")
        count = max(1, int(np.ceil(len(self._records_by_label) * fraction)))
        ordered = sorted(
            self._records_by_label,
            key=lambda label: (len(self._records_by_label[label]), label),
        )
        return tuple(ordered[:count])


def _source_indices(partition: Any) -> tuple[int, ...]:
    """Resolve original dataset identifiers from nested Torch subsets."""
    if isinstance(partition, Subset):
        parent = _source_indices(partition.dataset)
        return tuple(parent[int(index)] for index in partition.indices)
    return tuple(range(len(partition)))
