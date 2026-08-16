"""Build a merged annotation SQLite store from source-owned exports."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from ...sources.base import SourceAnnotationExport
from ...utils.exceptions import raise_validation_error
from ...utils.logger import get_custom_logger
from ..blobs import encode_pixel_indices
from .schema import create_merged_annotation_schema, create_reference_annotation_table


logger = get_custom_logger(__name__)


@dataclass(frozen=True)
class AnnotationMergeInput:
    """Describe one source dataset and its normalized annotation export."""

    source: str
    dataset_id: str
    name: str
    imzml_path: Path
    annotation_export: Optional[SourceAnnotationExport]
    metadata: Mapping[str, Any]


class MergedAnnotationWriter:
    """Create one new annotation store for one composed imzML image."""

    def write(
        self,
        *,
        path: Path | str,
        inputs: Sequence[AnnotationMergeInput],
        pixel_mappings: Sequence[Mapping[str, Any]],
        filtering_metadata: Mapping[str, Any],
        max_fdr: Optional[float] = None,
    ) -> Path:
        """Create a new SQLite store without migration or replacement logic."""
        target = Path(path)
        if target.exists():
            raise_validation_error(
                "MergedAnnotations",
                f"Output annotation store already exists: '{target}'.",
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.unlink(missing_ok=True)

        # Store construction
        ## Build the complete store in a temporary file before publishing it.
        try:
            with closing(sqlite3.connect(temporary)) as connection:
                with connection:
                    connection.execute("PRAGMA foreign_keys = ON")
                    create_merged_annotation_schema(connection)
                    dataset_indices = self._insert_datasets(
                        connection,
                        inputs,
                        filtering_metadata,
                    )
                    merged_pixels = self._map_source_pixels(pixel_mappings)
                    class_ids = self._collect_classes(
                    inputs,
                    merged_pixels,
                    max_fdr=max_fdr,
                )
                    self._insert_merged_annotations(
                        connection,
                        inputs,
                        class_ids,
                        merged_pixels,
                        max_fdr,
                    )
                    self._insert_references(
                        connection,
                        inputs,
                        dataset_indices,
                        class_ids,
                    )
                    self._insert_pixel_segments(
                        connection,
                        pixel_mappings,
                        dataset_indices,
                    )
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        logger.info("Created merged annotation store at %s", target)
        return target

    # --------------------------------------------------
    # Section: Dataset and class registration
    # --------------------------------------------------

    @staticmethod
    def _insert_datasets(
        connection: sqlite3.Connection,
        inputs: Sequence[AnnotationMergeInput],
        filtering_metadata: Mapping[str, Any],
    ) -> Dict[tuple[str, str], int]:
        indices: Dict[tuple[str, str], int] = {}
        for dataset_index, merge_input in enumerate(inputs, start=1):
            identity = (merge_input.source, merge_input.dataset_id)
            indices[identity] = dataset_index
            table_name = create_reference_annotation_table(connection, dataset_index)
            export_metadata = (
                merge_input.annotation_export.metadata
                if merge_input.annotation_export is not None
                else {}
            )
            metadata = {**dict(merge_input.metadata), **dict(export_metadata)}
            connection.execute(
                """
                INSERT INTO datasets_metadata (
                    dataset_index, source, source_dataset_id, name,
                    source_imzml_path, reference_table_name,
                    filtering_metadata_json, dataset_metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_index,
                    merge_input.source,
                    merge_input.dataset_id,
                    merge_input.name,
                    str(merge_input.imzml_path),
                    table_name,
                    _json_dump(filtering_metadata),
                    _json_dump(metadata),
                ),
            )
        return indices

    @staticmethod
    def _collect_classes(
        inputs: Sequence[AnnotationMergeInput],
        merged_pixels: Mapping[tuple[str, str, int], int],
        *,
        max_fdr: Optional[float],
    ) -> Dict[tuple[str, str], int]:
        identities: set[tuple[str, str]] = set()
        for merge_input in inputs:
            if merge_input.annotation_export is None:
                continue
            for record in merge_input.annotation_export.records:
                if max_fdr is not None and (
                    record.get("fdr") is None
                    or float(record["fdr"]) > float(max_fdr)
                ):
                    continue
                has_selected_pixel = any(
                    (
                        merge_input.source,
                        merge_input.dataset_id,
                        int(source_spectrum_id),
                    )
                    in merged_pixels
                    for source_spectrum_id in record.get("spectrum_ids", ())
                )
                if has_selected_pixel:
                    identities.add(_annotation_identity(record))
        return {
            identity: position
            for position, identity in enumerate(sorted(identities), start=1)
        }

    @staticmethod
    def _map_source_pixels(
        pixel_mappings: Sequence[Mapping[str, Any]],
    ) -> Dict[tuple[str, str, int], int]:
        return {
            (
                str(mapping["source"]),
                str(mapping["source_dataset_id"]),
                int(mapping["source_spectrum_id"]),
            ): int(mapping["merged_spectrum_index"])
            for mapping in pixel_mappings
        }

    # --------------------------------------------------
    # Section: Annotation serialization
    # --------------------------------------------------

    @staticmethod
    def _insert_merged_annotations(
        connection: sqlite3.Connection,
        inputs: Sequence[AnnotationMergeInput],
        class_ids: Mapping[tuple[str, str], int],
        merged_pixels: Mapping[tuple[str, str, int], int],
        max_fdr: Optional[float],
    ) -> None:
        pixels_by_class: Dict[tuple[str, str], set[int]] = {
            identity: set() for identity in class_ids
        }
        charge_by_class: Dict[tuple[str, str], Optional[int]] = {}
        for merge_input in inputs:
            if merge_input.annotation_export is None:
                continue
            for record in merge_input.annotation_export.records:
                if max_fdr is not None and (
                    record.get("fdr") is None
                    or float(record["fdr"]) > float(max_fdr)
                ):
                    continue
                identity = _annotation_identity(record)
                if identity not in class_ids:
                    continue
                charge_by_class.setdefault(identity, _optional_int(record.get("charge")))
                for source_spectrum_id in record.get("spectrum_ids", ()):
                    merged_index = merged_pixels.get(
                        (
                            merge_input.source,
                            merge_input.dataset_id,
                            int(source_spectrum_id),
                        )
                    )
                    if merged_index is not None:
                        pixels_by_class[identity].add(merged_index)
        connection.executemany(
            """
            INSERT INTO merged_annotations (
                merged_annotation_id, formula, adduct, charge, pixel_indices_blob
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    class_ids[identity],
                    identity[0],
                    identity[1],
                    charge_by_class.get(identity),
                    encode_pixel_indices(pixels_by_class[identity]),
                )
                for identity in class_ids
            ],
        )

    @staticmethod
    def _insert_references(
        connection: sqlite3.Connection,
        inputs: Sequence[AnnotationMergeInput],
        dataset_indices: Mapping[tuple[str, str], int],
        class_ids: Mapping[tuple[str, str], int],
    ) -> None:
        for merge_input in inputs:
            export = merge_input.annotation_export
            if export is None:
                continue
            dataset_index = dataset_indices[(merge_input.source, merge_input.dataset_id)]
            table_name = f"reference_annotations_{dataset_index:04d}"
            rows = []
            for reference_id, record in enumerate(export.records, start=1):
                identity = _annotation_identity(record)
                if identity not in class_ids:
                    continue
                source_record = record.get("source_record", {})
                rows.append(
                    (
                        reference_id,
                        class_ids[identity],
                        str(record["source_annotation_id"]),
                        identity[0],
                        identity[1],
                        _optional_float(record.get("mz")),
                        _optional_float(record.get("fdr")),
                        _optional_text(record.get("database_id")),
                        _optional_text(record.get("database_name")),
                        _optional_text(record.get("database_version")),
                        _json_dump(source_record),
                    )
                )
            connection.executemany(
                f"""
                INSERT INTO {table_name} (
                    reference_annotation_id, merged_annotation_id,
                    source_annotation_id, formula, adduct, mz, fdr,
                    database_id, database_name, database_version,
                    source_record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    # --------------------------------------------------
    # Section: Pixel provenance compression
    # --------------------------------------------------

    @staticmethod
    def _insert_pixel_segments(
        connection: sqlite3.Connection,
        pixel_mappings: Sequence[Mapping[str, Any]],
        dataset_indices: Mapping[tuple[str, str], int],
    ) -> None:
        ordered = sorted(pixel_mappings, key=lambda item: int(item["merged_spectrum_index"]))
        segments = _compress_pixel_mappings(ordered, dataset_indices)
        connection.executemany(
            """
            INSERT INTO pixel_segments (
                dataset_index, merged_pixel_start, segment_length,
                source_pixel_start, source_step
            ) VALUES (?, ?, ?, ?, ?)
            """,
            segments,
        )


def _compress_pixel_mappings(
    mappings: Sequence[Mapping[str, Any]],
    dataset_indices: Mapping[tuple[str, str], int],
) -> list[tuple[int, int, int, int, int]]:
    """Compress ordered source mappings into constant-step segments."""
    if not mappings:
        return []
    segments: list[tuple[int, int, int, int, int]] = []
    start = 0
    while start < len(mappings):
        first = mappings[start]
        identity = (str(first["source"]), str(first["source_dataset_id"]))
        merged_start = int(first["merged_spectrum_index"])
        source_start = int(first["source_spectrum_id"])
        end = start + 1
        step = 1
        if end < len(mappings):
            following = mappings[end]
            if (
                (str(following["source"]), str(following["source_dataset_id"]))
                == identity
                and int(following["merged_spectrum_index"]) == merged_start + 1
            ):
                step = int(following["source_spectrum_id"]) - source_start
                end += 1
        while end < len(mappings):
            current = mappings[end]
            if (
                (str(current["source"]), str(current["source_dataset_id"]))
                != identity
                or int(current["merged_spectrum_index"]) != merged_start + (end - start)
                or int(current["source_spectrum_id"])
                != source_start + step * (end - start)
            ):
                break
            end += 1
        segments.append(
            (
                dataset_indices[identity],
                merged_start,
                end - start,
                source_start,
                step,
            )
        )
        start = end
    return segments


def _annotation_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    formula = str(record.get("formula") or "").strip()
    adduct = str(record.get("adduct") or "").strip()
    if not formula or not adduct:
        raise_validation_error(
            "MergedAnnotations",
            "Every annotation requires non-empty formula and adduct values.",
        )
    return formula, adduct


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _optional_float(value: Any) -> Optional[float]:
    return None if value in (None, "") else float(value)


def _optional_int(value: Any) -> Optional[int]:
    return None if value in (None, "") else int(value)


def _optional_text(value: Any) -> Optional[str]:
    return None if value in (None, "") else str(value)
