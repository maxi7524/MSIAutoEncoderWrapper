"""Read one composed annotation SQLite store."""

from __future__ import annotations

import json
import sqlite3
import math
from bisect import bisect_right
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from ...utils.exceptions import raise_validation_error
from ..blobs import decode_pixel_indices
from ..index import SpectrumAnnotationIndex, build_annotation_index


class MergedAnnotationReader:
    """Expose merged classes and dataset-specific source references."""

    def __init__(
        self,
        path: Path | str,
        *,
        image_path: Path | str | None = None,
        default_filters: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise_validation_error(
                "MergedAnnotationReader",
                f"Annotation store does not exist: '{self.path}'.",
            )
        self.image_path = Path(image_path) if image_path is not None else None
        self.default_filters = dict(default_filters or {})
        self.active_context: Any = None
        self._pixel_to_classes: Optional[Dict[int, List[int]]] = None
        self._spectrum_annotation_indices: Dict[
            tuple[tuple[int, ...] | None, tuple[tuple[str, str], ...]],
            SpectrumAnnotationIndex,
        ] = {}
        self._validate_schema()
        self._config = {
            "type": "merge",
            "path": str(self.path),
            "image_path": str(self.image_path) if self.image_path is not None else None,
            "default_filters": self.default_filters,
        }

    # --------------------------------------------------
    # Section: Public annotation interface
    # --------------------------------------------------

    def get_dataset_metadata(self) -> Dict[str, Any]:
        """Return metadata for every source dataset in composition order."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM datasets_metadata ORDER BY dataset_index"
            ).fetchall()
        return {
            "type": "merge",
            "image_path": str(self.image_path) if self.image_path is not None else None,
            "datasets": [_decode_dataset(row) for row in rows],
        }

    def get_annotations(
        self,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return global ``formula + adduct`` classes in the merged image."""
        effective = {**self.default_filters, **dict(filters or {})}
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM merged_annotations ORDER BY merged_annotation_id"
            ).fetchall()
            datasets = connection.execute(
                "SELECT reference_table_name FROM datasets_metadata"
            ).fetchall()
            results = []
            for row in rows:
                record = _decode_merged_annotation(row)
                if not _matches_class_filters(record, effective):
                    continue
                if _has_reference_filters(effective) and not any(
                    _class_has_matching_reference(
                        connection,
                        str(dataset["reference_table_name"]),
                        int(row["merged_annotation_id"]),
                        effective,
                    )
                    for dataset in datasets
                ):
                    continue
                results.append(record)
        return results

    def get_spectrum_annotations(
        self,
        spectrum_id: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return source references for classes present in one merged pixel."""
        merged_pixel = int(spectrum_id)
        effective = {**self.default_filters, **dict(filters or {})}
        class_ids = self._get_pixel_to_classes().get(merged_pixel, [])
        if not class_ids:
            return []
        with self._connection() as connection:
            dataset, source_pixel = _resolve_source_pixel(connection, merged_pixel)
            if dataset is None:
                return []
            placeholders = ",".join("?" for _ in class_ids)
            table_name = str(dataset["reference_table_name"])
            rows = connection.execute(
                f"""
                SELECT * FROM {table_name}
                WHERE merged_annotation_id IN ({placeholders})
                ORDER BY merged_annotation_id, reference_annotation_id
                """,
                class_ids,
            ).fetchall()
        results = []
        for row in rows:
            record = _decode_reference(row)
            if not _matches_reference_filters(record, effective):
                continue
            record.update(
                {
                    "source": dataset["source"],
                    "source_dataset_id": dataset["source_dataset_id"],
                    "source_spectrum_id": source_pixel,
                    "merged_spectrum_id": merged_pixel,
                }
            )
            results.append(record)
        return results

    def get_spectrum_annotation_index(
        self,
        spectrum_ids: Any,
        filters: Optional[Mapping[str, Any]] = None,
        *,
        spectrum_ranges: Sequence[Sequence[int]] | None = None,
    ) -> SpectrumAnnotationIndex:
        """Return source-specific molecular m/z evidence in one bulk read.

        Every global table is scanned once and every dataset-specific reference
        table is queried once. Pixel-to-dataset routing uses the compressed
        ``pixel_segments`` ranges; no per-spectrum SQLite query is executed.

        :param spectrum_ids: Merged spectrum identifiers to represent. ``None``
            stores annotated rows only, while missing lookups remain empty.
        :type spectrum_ids: Iterable[int] | None
        :param filters: Optional class and source-reference filters.
        :type filters: Mapping[str, Any] | None
        :param spectrum_ranges: Optional half-open merged-spectrum ranges. They
            are evaluated without expanding every range into individual IDs.
        :type spectrum_ranges: Sequence[Sequence[int]] | None
        :return: Cached CSR spectrum annotation index.
        :rtype: SpectrumAnnotationIndex
        """
        selected = (
            None
            if spectrum_ids is None
            else tuple(sorted({int(value) for value in spectrum_ids}))
        )
        ranges = _normalize_spectrum_ranges(spectrum_ranges)
        effective = {**self.default_filters, **dict(filters or {})}
        filter_key = tuple(sorted((str(key), repr(value)) for key, value in effective.items()))
        cache_key = (selected, ranges, filter_key)
        cached = self._spectrum_annotation_indices.get(cache_key)
        if cached is not None:
            return cached

        selected_set = set(selected) if selected is not None else None
        entries: Dict[int, List[tuple[tuple[str, str], float]]] = {}
        with self._connection() as connection:
            segments = connection.execute(
                """
                SELECT pixel_segments.*, datasets_metadata.reference_table_name
                FROM pixel_segments
                JOIN datasets_metadata USING (dataset_index)
                ORDER BY merged_pixel_start
                """
            ).fetchall()
            if ranges:
                segments = [
                    segment
                    for segment in segments
                    if _range_overlaps(
                        int(segment["merged_pixel_start"]),
                        int(segment["merged_pixel_start"]) + int(segment["segment_length"]),
                        ranges,
                    )
                ]
            class_rows = connection.execute(
                """
                SELECT merged_annotation_id, formula, adduct, charge,
                       pixel_indices_blob
                FROM merged_annotations
                ORDER BY merged_annotation_id
                """
            ).fetchall()

            # Source-reference preload
            ## One scan per source table replaces one query per spectrum.
            references: Dict[tuple[int, int], list[float]] = {}
            loaded_datasets: set[int] = set()
            for segment in segments:
                dataset_index = int(segment["dataset_index"])
                table_name = str(segment["reference_table_name"])
                if dataset_index in loaded_datasets:
                    continue
                loaded_datasets.add(dataset_index)
                rows = connection.execute(
                    f"SELECT * FROM {table_name} ORDER BY merged_annotation_id, reference_annotation_id"
                ).fetchall()
                for row in rows:
                    record = _decode_reference(row)
                    if not _matches_reference_filters(record, effective):
                        continue
                    mz = record.get("mz")
                    resolved_mz = (
                        float(mz)
                        if mz is not None and math.isfinite(float(mz))
                        else float("nan")
                    )
                    references.setdefault(
                        (dataset_index, int(row["merged_annotation_id"])), []
                    ).append(resolved_mz)

        starts = [int(row["merged_pixel_start"]) for row in segments]
        for row in class_rows:
            record = _decode_merged_annotation(row)
            if not _matches_class_filters(record, effective):
                continue
            class_id = int(row["merged_annotation_id"])
            identity = (str(row["formula"]), str(row["adduct"]))
            for pixel in record["spectrum_ids"]:
                if selected_set is not None and pixel not in selected_set:
                    continue
                if ranges and not _spectrum_id_in_ranges(pixel, ranges):
                    continue
                position = bisect_right(starts, pixel) - 1
                segment = segments[position] if position >= 0 else None
                if segment is None or pixel >= (
                    int(segment["merged_pixel_start"]) + int(segment["segment_length"])
                ):
                    continue
                mz_values = references.get(
                    (int(segment["dataset_index"]), class_id),
                    (),
                )
                for mz in dict.fromkeys(mz_values):
                    entries.setdefault(pixel, []).append((identity, mz))

        represented_ids = list(entries) if selected is None else list(selected)
        index = build_annotation_index(represented_ids, entries)
        self._spectrum_annotation_indices[cache_key] = index
        return index

    def get_merged_spectrum_ranges(
        self,
        *,
        excluded_dataset_ids: Sequence[str] = (),
    ) -> tuple[tuple[int, int], ...]:
        """Return compact merged-spectrum ranges after source exclusion.

        The result contains sorted, non-overlapping half-open intervals
        ``(start, stop)``. The query reads only ``pixel_segments`` and
        ``datasets_metadata``; it never opens source imzML files and never
        materializes every individual pixel index.

        :param excluded_dataset_ids: Source dataset IDs withheld from the
            merged population.
        :type excluded_dataset_ids: Sequence[str]
        :return: Allowed half-open merged-spectrum intervals.
        :rtype: tuple[tuple[int, int], ...]
        :raises ValidationError: If an excluded ID is absent or every segment
            would be removed.
        """
        if isinstance(excluded_dataset_ids, (str, bytes)):
            raise_validation_error(
                "MergedAnnotationReader",
                "excluded_dataset_ids must be a sequence of dataset IDs, not a string.",
            )
        excluded = {str(value) for value in excluded_dataset_ids}
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT datasets_metadata.source_dataset_id,
                       pixel_segments.merged_pixel_start,
                       pixel_segments.segment_length
                FROM pixel_segments
                JOIN datasets_metadata USING (dataset_index)
                ORDER BY pixel_segments.merged_pixel_start
                """
            ).fetchall()
        known = {str(row["source_dataset_id"]) for row in rows}
        missing = sorted(excluded - known)
        if missing:
            raise_validation_error(
                "MergedAnnotationReader",
                f"Excluded dataset IDs are absent from the merged store: {missing}.",
            )
        ranges = [
            (
                int(row["merged_pixel_start"]),
                int(row["merged_pixel_start"]) + int(row["segment_length"]),
            )
            for row in rows
            if str(row["source_dataset_id"]) not in excluded
        ]
        normalized = _normalize_spectrum_ranges(ranges)
        if not normalized:
            raise_validation_error(
                "MergedAnnotationReader",
                "Source exclusion removed every merged spectrum segment.",
            )
        return normalized

    def get_spectrum_metadata(self, spectrum_id: int) -> Dict[str, Any]:
        """Return source dataset metadata and source index for one merged pixel."""
        with self._connection() as connection:
            dataset, source_pixel = _resolve_source_pixel(connection, int(spectrum_id))
        if dataset is None:
            return {}
        result = _decode_dataset(dataset)
        result.update(
            {
                "merged_spectrum_id": int(spectrum_id),
                "source_spectrum_id": source_pixel,
            }
        )
        return result

    def get_spectrum_groups(
        self,
        spectrum_ids: List[int],
        group_fields: Any = None,
    ) -> List[Any]:
        """Return source-dataset groups for many merged pixels in one query."""
        fields = group_fields or ("source_dataset_id", "dataset_id", "image_key")
        fields = [fields] if isinstance(fields, str) else list(fields)
        with self._connection() as connection:
            segments = connection.execute(
                """
                SELECT pixel_segments.*, datasets_metadata.*
                FROM pixel_segments
                JOIN datasets_metadata USING (dataset_index)
                ORDER BY merged_pixel_start
                """
            ).fetchall()
        starts = [int(row["merged_pixel_start"]) for row in segments]
        result = []
        for pixel in spectrum_ids:
            position = bisect_right(starts, int(pixel)) - 1
            row = segments[position] if position >= 0 else None
            if row is None or int(pixel) >= int(row["merged_pixel_start"]) + int(row["segment_length"]):
                result.append(("__all_samples__",))
                continue
            metadata = _decode_dataset(row)
            values = tuple(metadata.get(field) for field in fields)
            result.append(tuple(value for value in values if value is not None and value != "") or ("__all_samples__",))
        return result

    def get_spectrum_mask(self, spectrum_id: int, mask: str) -> bool:
        """Return the built-in annotation-presence mask for one merged pixel."""
        if mask not in {"annotated", "annotation"}:
            raise_validation_error(
                "MergedAnnotationReader",
                f"Unknown annotation mask '{mask}'.",
            )
        return bool(self._get_pixel_to_classes().get(int(spectrum_id)))

    def get_config(self) -> Dict[str, Any]:
        """Return the reader configuration used by wrapper context persistence."""
        return dict(self._config)

    # --------------------------------------------------
    # Section: Lazy reverse index
    # --------------------------------------------------

    def _get_pixel_to_classes(self) -> Dict[int, List[int]]:
        if self._pixel_to_classes is None:
            reverse: Dict[int, List[int]] = {}
            with self._connection() as connection:
                rows = connection.execute(
                    """
                    SELECT merged_annotation_id, pixel_indices_blob
                    FROM merged_annotations
                    """
                ).fetchall()
            for row in rows:
                class_id = int(row["merged_annotation_id"])
                for pixel_index in decode_pixel_indices(row["pixel_indices_blob"]):
                    reverse.setdefault(pixel_index, []).append(class_id)
            self._pixel_to_classes = reverse
        return self._pixel_to_classes

    # --------------------------------------------------
    # Section: Store validation and connections
    # --------------------------------------------------

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def _validate_schema(self) -> None:
        with self._connection() as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            required = {"datasets_metadata", "pixel_segments", "merged_annotations"}
            missing = required - tables
            if missing:
                raise_validation_error(
                    "MergedAnnotationReader",
                    f"Annotation store is missing tables: {sorted(missing)}.",
                )
            reference_tables = {
                str(row["reference_table_name"])
                for row in connection.execute(
                    "SELECT reference_table_name FROM datasets_metadata"
                ).fetchall()
            }
            missing_references = reference_tables - tables
            if missing_references:
                raise_validation_error(
                    "MergedAnnotationReader",
                    "Annotation store is missing source reference tables: "
                    f"{sorted(missing_references)}.",
                )


def _resolve_source_pixel(
    connection: sqlite3.Connection,
    merged_pixel: int,
) -> tuple[Optional[sqlite3.Row], Optional[int]]:
    segment = connection.execute(
        """
        SELECT * FROM pixel_segments
        WHERE merged_pixel_start <= ?
          AND merged_pixel_start + segment_length > ?
        ORDER BY merged_pixel_start DESC
        LIMIT 1
        """,
        (merged_pixel, merged_pixel),
    ).fetchone()
    if segment is None:
        return None, None
    dataset = connection.execute(
        "SELECT * FROM datasets_metadata WHERE dataset_index = ?",
        (segment["dataset_index"],),
    ).fetchone()
    offset = merged_pixel - int(segment["merged_pixel_start"])
    source_pixel = int(segment["source_pixel_start"]) + offset * int(
        segment["source_step"]
    )
    return dataset, source_pixel


def _decode_dataset(row: sqlite3.Row) -> Dict[str, Any]:
    metadata = json.loads(row["dataset_metadata_json"])
    filtering = json.loads(row["filtering_metadata_json"])
    return {
        "dataset_index": int(row["dataset_index"]),
        "source": str(row["source"]),
        "dataset_id": str(row["source_dataset_id"]),
        "name": row["name"],
        "image_path": str(row["source_imzml_path"]),
        "metadata": metadata,
        "filtering_metadata": filtering,
    }


def _decode_merged_annotation(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "merged_annotation_id": int(row["merged_annotation_id"]),
        "formula": str(row["formula"]),
        "adduct": str(row["adduct"]),
        "charge": row["charge"],
        "spectrum_ids": decode_pixel_indices(row["pixel_indices_blob"]),
    }


def _decode_reference(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "reference_annotation_id": int(row["reference_annotation_id"]),
        "merged_annotation_id": int(row["merged_annotation_id"]),
        "source_annotation_id": str(row["source_annotation_id"]),
        "formula": str(row["formula"]),
        "adduct": str(row["adduct"]),
        "mz": row["mz"],
        "fdr": row["fdr"],
        "database_id": row["database_id"],
        "database_name": row["database_name"],
        "database_version": row["database_version"],
        "source_record": json.loads(row["source_record_json"]),
    }


def _matches_class_filters(record: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    return all(
        record.get(key) == value
        for key, value in filters.items()
        if key in {"formula", "adduct", "charge"}
    )


def _has_reference_filters(filters: Mapping[str, Any]) -> bool:
    return bool(set(filters) - {"formula", "adduct", "charge"})


def _class_has_matching_reference(
    connection: sqlite3.Connection,
    table_name: str,
    class_id: int,
    filters: Mapping[str, Any],
) -> bool:
    rows = connection.execute(
        f"SELECT * FROM {table_name} WHERE merged_annotation_id = ?",
        (class_id,),
    ).fetchall()
    return any(_matches_reference_filters(_decode_reference(row), filters) for row in rows)


def _matches_reference_filters(
    record: Mapping[str, Any],
    filters: Mapping[str, Any],
) -> bool:
    max_fdr = filters.get("max_fdr")
    if max_fdr is not None and (
        record.get("fdr") is None or float(record["fdr"]) > float(max_fdr)
    ):
        return False
    for key, value in filters.items():
        if key == "max_fdr":
            continue
        if key in record and record.get(key) != value:
            return False
    return True


def _normalize_spectrum_ranges(
    ranges: Sequence[Sequence[int]] | None,
) -> tuple[tuple[int, int], ...]:
    """Validate and merge half-open merged-spectrum intervals."""
    if ranges is None:
        return ()
    normalized: list[tuple[int, int]] = []
    for value in ranges:
        if isinstance(value, (str, bytes)) or len(value) != 2:
            raise_validation_error(
                "MergedAnnotationReader",
                "Every spectrum range must contain exactly start and stop indices.",
            )
        start, stop = value
        if (
            isinstance(start, bool)
            or isinstance(stop, bool)
            or not isinstance(start, int)
            or not isinstance(stop, int)
            or start < 0
            or stop <= start
        ):
            raise_validation_error(
                "MergedAnnotationReader",
                "Spectrum ranges must be non-empty non-negative integer intervals.",
            )
        normalized.append((int(start), int(stop)))
    merged: list[tuple[int, int]] = []
    for start, stop in sorted(normalized):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return tuple(merged)


def _spectrum_id_in_ranges(
    spectrum_id: int,
    ranges: Sequence[tuple[int, int]],
) -> bool:
    """Return whether one merged spectrum belongs to a half-open interval."""
    starts = [start for start, _ in ranges]
    position = bisect_right(starts, int(spectrum_id)) - 1
    return position >= 0 and int(spectrum_id) < ranges[position][1]


def _range_overlaps(
    start: int,
    stop: int,
    ranges: Sequence[tuple[int, int]],
) -> bool:
    """Return whether one interval intersects any normalized selection range."""
    return any(
        start < selected_stop and selected_start < stop
        for selected_start, selected_stop in ranges
    )
