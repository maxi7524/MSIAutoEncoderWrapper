"""Base contract for source-owned annotation artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


ANNOTATION_EXPORT_SCHEMA_VERSION = 1
ANNOTATION_TABLE_NAME = "annotations.csv"
PIXEL_INTENSITY_TABLE_NAME = "pixel_intensities.csv"
ANNOTATION_REQUIRED_COLUMNS = {
    "schema_version",
    "source",
    "source_annotation_id",
    "datasetId",
    "formula",
    "adduct",
    "mz",
    "fdr",
}
PIXEL_INTENSITY_REQUIRED_COLUMNS = {"mol_formula", "adduct", "mz"}


@dataclass(frozen=True)
class SourceAnnotationExport:
    """Annotations read from one normalized source dataset artifact.

    :param source: Registered source identifier.
    :param dataset_id: Stable source dataset identifier.
    :param schema_version: Exact dataset-manager export schema version.
    :param metadata: Dataset-level metadata stored in the export.
    :param records: Canonical source annotation records.
    """

    source: str
    dataset_id: str
    schema_version: int
    metadata: Dict[str, Any]
    records: list[Dict[str, Any]]


class AnnotationDatasetSource(ABC):
    """Require a source to own annotation download, normalization, and reading."""

    @staticmethod
    def annotation_export_paths(
        directory: Path,
        dataset_id: str,
    ) -> tuple[Path, Path]:
        """Return the only supported annotation artifact paths.

        ``dataset_id`` is accepted to keep the source contract extensible; the
        current schema uses fixed names inside the source dataset directory.
        """
        del dataset_id
        return (
            directory / ANNOTATION_TABLE_NAME,
            directory / PIXEL_INTENSITY_TABLE_NAME,
        )

    @classmethod
    def has_annotation_export(cls, directory: Path, dataset_id: str) -> bool:
        """Return whether both files implement the current exact schema."""
        annotations_path, intensities_path = cls.annotation_export_paths(
            directory,
            dataset_id,
        )
        if not all(
            path.is_file() and path.stat().st_size > 0
            for path in (annotations_path, intensities_path)
        ):
            return False
        try:
            with annotations_path.open("r", encoding="utf-8", newline="") as stream:
                annotation_reader = csv.DictReader(stream)
                if not ANNOTATION_REQUIRED_COLUMNS.issubset(
                    set(annotation_reader.fieldnames or ())
                ):
                    return False
                first = next(annotation_reader, None)
                if first is not None and first.get("schema_version") != str(
                    ANNOTATION_EXPORT_SCHEMA_VERSION
                ):
                    return False
            with intensities_path.open("r", encoding="utf-8", newline="") as stream:
                intensity_reader = csv.DictReader(stream)
                return PIXEL_INTENSITY_REQUIRED_COLUMNS.issubset(
                    set(intensity_reader.fieldnames or ())
                )
        except (OSError, UnicodeDecodeError, csv.Error):
            return False

    @abstractmethod
    def materialize_annotations(
        self,
        *,
        dataset_id: str,
        dataset_name: str,
        directory: Path,
        imzml_path: Path,
        options: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Fetch, normalize, and atomically write one source annotation export."""

    @classmethod
    @abstractmethod
    def read_annotation_export(
        cls,
        *,
        dataset_id: str,
        directory: Path,
        imzml_path: Path,
    ) -> SourceAnnotationExport:
        """Read exactly the current source export schema from local files."""
