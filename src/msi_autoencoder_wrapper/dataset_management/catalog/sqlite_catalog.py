"""SQLite catalog for managed MSI datasets, annotations, and merge provenance."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from ...utils.logger import get_custom_logger


logger = get_custom_logger(__name__)


class DatasetCatalog:
    """Persist dataset metadata and stable source-to-merge index mappings.

    :param path: Location of the SQLite catalog. Parent directories are created.
    :type path: pathlib.Path | str

    The catalog stores source metadata without interpreting experiment-specific
    labels. Filtering remains the responsibility of data-source queries and
    annotation readers.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a transactional SQLite connection with mapping-style rows.

        :return: Open SQLite connection committed on successful exit.
        :rtype: Iterator[sqlite3.Connection]
        """
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def upsert_dataset(
        self,
        *,
        source: str,
        dataset_id: str,
        name: str,
        metadata: Mapping[str, Any],
        local_path: Optional[Path | str] = None,
        status: str = "discovered",
    ) -> None:
        """Insert or update one dataset and its complete source metadata.

        :param source: External source identifier, for example ``metaspace``.
        :type source: str
        :param dataset_id: Stable identifier assigned by the source.
        :type dataset_id: str
        :param name: Human-readable dataset name.
        :type name: str
        :param metadata: Unfiltered metadata returned by the source.
        :type metadata: Mapping[str, Any]
        :param local_path: Optional materialized dataset directory.
        :type local_path: pathlib.Path | str | None
        :param status: Current catalog lifecycle status.
        :type status: str
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO datasets (
                    source, dataset_id, name, metadata_json, local_path,
                    status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, dataset_id) DO UPDATE SET
                    name = excluded.name,
                    metadata_json = excluded.metadata_json,
                    local_path = COALESCE(excluded.local_path, datasets.local_path),
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    source,
                    dataset_id,
                    name,
                    _json_dump(metadata),
                    str(local_path) if local_path is not None else None,
                    status,
                    timestamp,
                ),
            )

    def get_dataset(self, source: str, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Return one dataset record or ``None`` when it is unknown."""
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM datasets WHERE source = ? AND dataset_id = ?",
                (source, dataset_id),
            ).fetchone()
        return _decode_dataset_row(row) if row is not None else None

    def list_datasets(
        self,
        *,
        source: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return datasets selected by source and lifecycle status."""
        clauses: List[str] = []
        parameters: List[Any] = []
        if source is not None:
            clauses.append("source = ?")
            parameters.append(source)
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status)
        query = "SELECT * FROM datasets"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY source, dataset_id"
        with self.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [_decode_dataset_row(row) for row in rows]

    def resolve_dataset_path(self, path: Path | str) -> Optional[Dict[str, Any]]:
        """Resolve a local imzML path to a source or merged dataset record.

        :param path: Local imzML path currently opened by a data reader.
        :type path: pathlib.Path | str
        :return: Dataset identity with ``kind`` set to ``source`` or ``merged``,
            or ``None`` when the path is not registered.
        :rtype: Dict[str, Any] | None

        Path resolution does not require the registered source files to remain
        present. Metadata and annotations are stored in SQLite during import.
        """
        candidate = Path(path).expanduser().resolve()
        with self.connection() as connection:
            merged_rows = connection.execute(
                "SELECT merged_dataset_id, path FROM merged_datasets"
            ).fetchall()
            for row in merged_rows:
                if Path(row["path"]).expanduser().resolve() == candidate:
                    return {
                        "kind": "merged",
                        "merged_dataset_id": str(row["merged_dataset_id"]),
                    }

            source_rows = connection.execute(
                """
                SELECT source, dataset_id, local_path
                FROM datasets WHERE local_path IS NOT NULL
                """
            ).fetchall()
            for row in source_rows:
                local_path = Path(row["local_path"]).expanduser().resolve()
                if local_path == candidate or local_path == candidate.parent:
                    return {
                        "kind": "source",
                        "source": str(row["source"]),
                        "dataset_id": str(row["dataset_id"]),
                    }
        return None

    def replace_annotations(
        self,
        *,
        source: str,
        dataset_id: str,
        annotations: Iterable[Mapping[str, Any]],
    ) -> int:
        """Replace all imported annotations while preserving each raw record.

        :return: Number of stored annotation records.
        :rtype: int
        """
        prepared: List[tuple[Any, ...]] = []
        spectrum_links: List[tuple[Any, ...]] = []
        for position, annotation in enumerate(annotations):
            raw = dict(annotation)
            spectrum_ids = raw.pop("spectrum_ids", None)
            spectrum_values = raw.pop("spectrum_values", {})
            source_annotation_id = str(
                raw.get("annotation_id")
                or raw.get("id")
                or raw.get("annotationId")
                or position
            )
            database_name = _first_value(raw, "database", "database_name", "dbName")
            database_version = _first_value(raw, "database_version", "dbVersion")
            annotation_id = "|".join(
                (
                    database_name or "",
                    database_version or "",
                    source_annotation_id,
                    str(position),
                )
            )
            prepared.append(
                (
                    source,
                    dataset_id,
                    annotation_id,
                    database_name,
                    database_version,
                    _first_value(raw, "formula", "sumFormula"),
                    _first_value(raw, "adduct"),
                    _numeric_value(raw, "fdr", "FDR"),
                    _json_dump(raw),
                )
            )
            if spectrum_ids is not None:
                spectrum_links.extend(
                    (
                        source,
                        dataset_id,
                        annotation_id,
                        int(spectrum_id),
                        _optional_float(
                            spectrum_values.get(
                                str(spectrum_id), spectrum_values.get(spectrum_id)
                            )
                        ),
                    )
                    for spectrum_id in spectrum_ids
                )
        with self.connection() as connection:
            connection.execute(
                "DELETE FROM annotations WHERE source = ? AND dataset_id = ?",
                (source, dataset_id),
            )
            connection.executemany(
                """
                INSERT INTO annotations (
                    source, dataset_id, annotation_id, database_name,
                    database_version, formula, adduct, fdr, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                prepared,
            )
            connection.executemany(
                """
                INSERT INTO spectrum_annotations (
                    source, dataset_id, annotation_id, spectrum_id, intensity
                ) VALUES (?, ?, ?, ?, ?)
                """,
                spectrum_links,
            )
        return len(prepared)

    def get_annotations(
        self,
        *,
        source: str,
        dataset_id: str,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return raw annotations with optional read-time filters.

        Supported filters are ``database_name``, ``database_version``,
        ``formula``, ``adduct``, and ``max_fdr``. No filters are applied by
        default.
        """
        filters = dict(filters or {})
        clauses = ["source = ?", "dataset_id = ?"]
        parameters: List[Any] = [source, dataset_id]
        for field in ("database_name", "database_version", "formula", "adduct"):
            if filters.get(field) is not None:
                clauses.append(f"{field} = ?")
                parameters.append(filters[field])
        if filters.get("max_fdr") is not None:
            clauses.append("fdr <= ?")
            parameters.append(float(filters["max_fdr"]))
        query = (
            "SELECT raw_json FROM annotations WHERE "
            + " AND ".join(clauses)
            + " ORDER BY annotation_id"
        )
        with self.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [json.loads(row["raw_json"]) for row in rows]

    def get_annotation_materialization(
        self,
        *,
        source: str,
        dataset_id: str,
        options: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Return the completed annotation import matching retrieval options."""
        options_json = _json_dump(options)
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT status, annotation_count, completed_at
                FROM annotation_materializations
                WHERE source = ? AND dataset_id = ? AND options_json = ?
                """,
                (source, dataset_id, options_json),
            ).fetchone()
        return dict(row) if row is not None else None

    def record_annotation_materialization(
        self,
        *,
        source: str,
        dataset_id: str,
        options: Mapping[str, Any],
        annotation_count: int,
    ) -> None:
        """Record a successful, complete annotation materialization."""
        completed_at = datetime.now(timezone.utc).isoformat()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO annotation_materializations (
                    source, dataset_id, options_json, status,
                    annotation_count, completed_at
                ) VALUES (?, ?, ?, 'complete', ?, ?)
                ON CONFLICT(source, dataset_id, options_json) DO UPDATE SET
                    status = excluded.status,
                    annotation_count = excluded.annotation_count,
                    completed_at = excluded.completed_at
                """,
                (
                    source,
                    dataset_id,
                    _json_dump(options),
                    int(annotation_count),
                    completed_at,
                ),
            )

    def clear_annotation_materializations(
        self,
        *,
        source: str,
        dataset_id: str,
    ) -> None:
        """Invalidate completion markers before replacing canonical annotations."""
        with self.connection() as connection:
            connection.execute(
                """
                DELETE FROM annotation_materializations
                WHERE source = ? AND dataset_id = ?
                """,
                (source, dataset_id),
            )

    def get_spectrum_annotations(
        self,
        *,
        source: str,
        dataset_id: str,
        spectrum_id: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return canonical molecular annotations linked to one spectrum.

        :param source: External source identifier.
        :type source: str
        :param dataset_id: Stable source dataset identifier.
        :type dataset_id: str
        :param spectrum_id: Reader-compatible zero-based spectrum identifier.
        :type spectrum_id: int
        :param filters: Optional annotation filters.
        :type filters: Mapping[str, Any] | None
        :return: Matching canonical annotation records.
        :rtype: List[Dict[str, Any]]

        Dataset-level annotations without spectrum links are deliberately not
        returned here. They remain available through :meth:`get_annotations`.
        """
        filters = dict(filters or {})
        clauses = [
            "a.source = ?",
            "a.dataset_id = ?",
            "sa.spectrum_id = ?",
        ]
        parameters: List[Any] = [source, dataset_id, int(spectrum_id)]
        for field in ("database_name", "database_version", "formula", "adduct"):
            if filters.get(field) is not None:
                clauses.append(f"a.{field} = ?")
                parameters.append(filters[field])
        if filters.get("max_fdr") is not None:
            clauses.append("a.fdr <= ?")
            parameters.append(float(filters["max_fdr"]))
        query = (
            "SELECT a.raw_json, sa.intensity FROM annotations AS a "
            "JOIN spectrum_annotations AS sa USING (source, dataset_id, annotation_id) "
            "WHERE " + " AND ".join(clauses) + " ORDER BY a.annotation_id"
        )
        with self.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        results: List[Dict[str, Any]] = []
        for row in rows:
            record = json.loads(row["raw_json"])
            record["spectrum_id"] = int(spectrum_id)
            if row["intensity"] is not None:
                record["intensity"] = float(row["intensity"])
            results.append(record)
        return results

    def get_annotated_spectrum_ids(
        self,
        *,
        source: str,
        dataset_id: str,
    ) -> List[int]:
        """Return spectrum IDs linked to at least one molecular annotation.

        :param source: External source identifier.
        :type source: str
        :param dataset_id: Stable source dataset identifier.
        :type dataset_id: str
        :return: Sorted, unique zero-based spectrum IDs.
        :rtype: List[int]
        """
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT spectrum_id
                FROM spectrum_annotations
                WHERE source = ? AND dataset_id = ?
                ORDER BY spectrum_id
                """,
                (source, dataset_id),
            ).fetchall()
        return [int(row["spectrum_id"]) for row in rows]

    def register_merged_dataset(self, merged_dataset_id: str, path: Path | str) -> None:
        """Register an imzML dataset produced by a merge operation."""
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO merged_datasets (merged_dataset_id, path, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(merged_dataset_id) DO UPDATE SET path = excluded.path
                """,
                (
                    merged_dataset_id,
                    str(path),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def replace_spectrum_mappings(
        self,
        merged_dataset_id: str,
        mappings: Sequence[Mapping[str, Any]],
    ) -> None:
        """Atomically replace source spatial-ID to merged-index mappings."""
        rows = [
            (
                merged_dataset_id,
                str(mapping["source"]),
                str(mapping["source_dataset_id"]),
                int(mapping["source_spectrum_id"]),
                int(mapping["merged_spectrum_index"]),
            )
            for mapping in mappings
        ]
        with self.connection() as connection:
            connection.execute(
                "DELETE FROM spectrum_mappings WHERE merged_dataset_id = ?",
                (merged_dataset_id,),
            )
            connection.executemany(
                """
                INSERT INTO spectrum_mappings (
                    merged_dataset_id, source, source_dataset_id,
                    source_spectrum_id, merged_spectrum_index
                ) VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

    def get_merged_index(
        self,
        *,
        merged_dataset_id: str,
        source: str,
        source_dataset_id: str,
        source_spectrum_id: int,
    ) -> Optional[int]:
        """Resolve a source spectrum ID to its merged spectrum index."""
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT merged_spectrum_index FROM spectrum_mappings
                WHERE merged_dataset_id = ? AND source = ?
                  AND source_dataset_id = ? AND source_spectrum_id = ?
                """,
                (
                    merged_dataset_id,
                    source,
                    source_dataset_id,
                    source_spectrum_id,
                ),
            ).fetchone()
        return int(row["merged_spectrum_index"]) if row is not None else None

    def get_source_index(
        self,
        *,
        merged_dataset_id: str,
        merged_spectrum_index: int,
    ) -> Optional[Dict[str, Any]]:
        """Resolve a merged spectrum index back to its source spatial ID."""
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT source, source_dataset_id, source_spectrum_id
                FROM spectrum_mappings
                WHERE merged_dataset_id = ? AND merged_spectrum_index = ?
                """,
                (merged_dataset_id, merged_spectrum_index),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_merged_sources(self, merged_dataset_id: str) -> List[Dict[str, str]]:
        """Return distinct source datasets contributing to a merged dataset."""
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT source, source_dataset_id
                FROM spectrum_mappings
                WHERE merged_dataset_id = ?
                ORDER BY source, source_dataset_id
                """,
                (merged_dataset_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _initialize_schema(self) -> None:
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    source TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    local_path TEXT,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (source, dataset_id)
                );

                CREATE TABLE IF NOT EXISTS annotations (
                    source TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    annotation_id TEXT NOT NULL,
                    database_name TEXT,
                    database_version TEXT,
                    formula TEXT,
                    adduct TEXT,
                    fdr REAL,
                    raw_json TEXT NOT NULL,
                    PRIMARY KEY (source, dataset_id, annotation_id),
                    FOREIGN KEY (source, dataset_id)
                        REFERENCES datasets(source, dataset_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS spectrum_annotations (
                    source TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    annotation_id TEXT NOT NULL,
                    spectrum_id INTEGER NOT NULL,
                    intensity REAL,
                    PRIMARY KEY (source, dataset_id, annotation_id, spectrum_id),
                    FOREIGN KEY (source, dataset_id, annotation_id)
                        REFERENCES annotations(source, dataset_id, annotation_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS annotation_materializations (
                    source TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    options_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    annotation_count INTEGER NOT NULL,
                    completed_at TEXT NOT NULL,
                    PRIMARY KEY (source, dataset_id, options_json),
                    FOREIGN KEY (source, dataset_id)
                        REFERENCES datasets(source, dataset_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS merged_datasets (
                    merged_dataset_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS spectrum_mappings (
                    merged_dataset_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_dataset_id TEXT NOT NULL,
                    source_spectrum_id INTEGER NOT NULL,
                    merged_spectrum_index INTEGER NOT NULL,
                    PRIMARY KEY (merged_dataset_id, merged_spectrum_index),
                    UNIQUE (
                        merged_dataset_id, source, source_dataset_id,
                        source_spectrum_id
                    ),
                    FOREIGN KEY (merged_dataset_id)
                        REFERENCES merged_datasets(merged_dataset_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_annotations_dataset
                    ON annotations(source, dataset_id);
                CREATE INDEX IF NOT EXISTS idx_spectrum_annotations_lookup
                    ON spectrum_annotations(source, dataset_id, spectrum_id);
                CREATE INDEX IF NOT EXISTS idx_spectrum_mapping_source
                    ON spectrum_mappings(
                        merged_dataset_id, source, source_dataset_id,
                        source_spectrum_id
                    );
                """
            )
        logger.debug("Dataset catalog initialized at %s", self.path)


def _json_dump(value: Mapping[str, Any]) -> str:
    return json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)


def _decode_dataset_row(row: sqlite3.Row) -> Dict[str, Any]:
    result = dict(row)
    result["metadata"] = json.loads(result.pop("metadata_json"))
    return result


def _first_value(record: Mapping[str, Any], *names: str) -> Optional[str]:
    for name in names:
        value = record.get(name)
        if value is not None:
            return str(value)
    return None


def _numeric_value(record: Mapping[str, Any], *names: str) -> Optional[float]:
    value = _first_value(record, *names)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    """Convert an optional annotation intensity to a SQLite number."""
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
