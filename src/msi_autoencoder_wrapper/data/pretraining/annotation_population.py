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
        member_getter = getattr(dataset, "get_annotation_member_datasets", None)
        if callable(member_getter):
            return cls._from_cohort(dataset, target_field)
        train_partition = dataset.create_partitions().train
        spectrum_ids = _source_indices(train_partition)
        index = dataset.get_mapped_annotation_index()
        if index.coordinate_system != "binner":
            raise ValueError("Annotation synthesis requires binner-mapped annotations.")
        target_names = dataset.get_target_schemas()[target_field].class_names
        target_indices = {name: position for position, name in enumerate(target_names)}
        records = _records_from_sparse_index(
            index=index,
            selected_spectrum_ids=spectrum_ids,
            target_indices=target_indices,
        )
        return cls(
            feature_count=len(index.coordinate_axis),
            spectrum_ids=spectrum_ids,
            records=records,
        )

    @classmethod
    def _from_cohort(cls, dataset: Any, target_field: str) -> "AnnotationPopulation":
        """Merge train-only pixel/bin records while retaining global source IDs."""
        train_partition = dataset.create_partitions().train
        source_ids = set(_source_indices(train_partition))
        target_names = dataset.get_target_schemas()[target_field].class_names
        target_indices = {name: position for position, name in enumerate(target_names)}
        records: list[AnnotationPeakRecord] = []
        feature_count = len(dataset.get_synthetic_context().binner.GetXAxis())
        for offset, member_dataset in dataset.get_annotation_member_datasets():
            index = member_dataset.get_mapped_annotation_index()
            if len(index.coordinate_axis) != feature_count:
                raise ValueError("Cohort annotation axes do not match the shared binner.")
            local_source_ids = {
                source_id - offset
                for source_id in source_ids
                if offset <= source_id < offset + member_dataset._source_length()
            }
            local_records = _records_from_sparse_index(
                index=index,
                selected_spectrum_ids=local_source_ids,
                target_indices=target_indices,
            )
            records.extend(
                AnnotationPeakRecord(
                    spectrum_id=offset + record.spectrum_id,
                    bin_index=record.bin_index,
                    label_indices=record.label_indices,
                )
                for record in local_records
            )
        return cls(
            feature_count=feature_count,
            spectrum_ids=source_ids,
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


def _records_from_sparse_index(
    *,
    index: Any,
    selected_spectrum_ids: Iterable[int],
    target_indices: Mapping[str, int],
) -> list[AnnotationPeakRecord]:
    """Extract selected records by intersecting CSR rows before decoding entries.

    :param index: Mapped sparse annotation index with sorted spectrum IDs.
    :type index: Any
    :param selected_spectrum_ids: Source IDs belonging to the train split.
    :type selected_spectrum_ids: collections.abc.Iterable[int]
    :param target_indices: Molecular identity to target-column mapping.
    :type target_indices: collections.abc.Mapping[str, int]
    :return: Complete positive records from selected sparse index rows.
    :rtype: list[AnnotationPeakRecord]

    REMARK: The index stores rows only for annotated spectra. Intersecting those
    rows with the selected train IDs makes extraction scale with annotated
    spectra rather than the complete merged-dataset length.
    """
    requested = np.asarray(
        sorted({int(value) for value in selected_spectrum_ids}),
        dtype=np.int64,
    )
    indexed = np.asarray(index.spectrum_ids, dtype=np.int64)
    row_ids = np.searchsorted(indexed, requested)
    in_bounds = row_ids < indexed.size
    matched = row_ids[in_bounds]
    matched = matched[indexed[matched] == requested[in_bounds]]
    identity_targets = np.asarray(
        [
            target_indices.get("|".join(identity), -1)
            for identity in index.annotation_identities
        ],
        dtype=np.int64,
    )
    records: list[AnnotationPeakRecord] = []
    for row_id in matched:
        start, stop = int(index.spectrum_offsets[row_id]), int(index.spectrum_offsets[row_id + 1])
        entry_targets = identity_targets[index.annotation_indices[start:stop]]
        coordinates = index.coordinate_indices[start:stop]
        keep = entry_targets >= 0
        if not bool(np.any(keep)):
            continue
        grouped: dict[int, set[int]] = defaultdict(set)
        for target_index, coordinate in zip(
            entry_targets[keep],
            coordinates[keep],
            strict=True,
        ):
            grouped[int(coordinate)].add(int(target_index))
        spectrum_id = int(indexed[row_id])
        records.extend(
            AnnotationPeakRecord(
                spectrum_id=spectrum_id,
                bin_index=coordinate,
                label_indices=tuple(sorted(labels)),
            )
            for coordinate, labels in grouped.items()
        )
    return records
