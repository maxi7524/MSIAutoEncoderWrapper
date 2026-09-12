"""SQLite persistence, filtering, and loading for candidate-ion catalogues."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

from ...layout import DatasetWorkspaceLayout
from ...metadata import read_candidate_metadata
from ...utils.logger import get_custom_logger
from .model import CandidateCompound, CandidateIon, make_candidate_ions
from .providers import CandidateProvider
from .snapshots import (
    DEFAULT_CANDIDATE_SOURCE_CACHE_DIR,
    materialize_candidate_sources,
)


CATALOG_SCHEMA_VERSION = 1
logger = get_custom_logger(__name__)


class CandidateCatalogWriter:
    """Write one self-contained candidate catalogue atomically."""

    def write(
        self,
        *,
        path: Path | str,
        compounds: Iterable[CandidateCompound],
        ions: Iterable[CandidateIon],
        manifest: Mapping[str, Any],
    ) -> Path:
        """Create one new candidate SQLite store and its compact manifest."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        compound_rows = tuple(compounds)
        ion_rows = tuple(ions)
        try:
            with closing(sqlite3.connect(temporary)) as connection:
                with connection:
                    _create_schema(connection)
                    connection.execute(
                        "INSERT INTO catalog_metadata VALUES (?, ?)",
                        ("manifest", _json(manifest)),
                    )
                    _insert_compounds(connection, compound_rows)
                    _insert_ions(connection, ion_rows)
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        logger.info(
            "Created candidate catalogue at %s with %s compounds and %s ions.",
            target,
            len(compound_rows),
            len(ion_rows),
        )
        return target


class CandidateCatalogReader:
    """Read and filter a self-contained external candidate catalogue."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise ValueError(f"Candidate catalogue does not exist: '{self.path}'.")
        self._validate_schema()
        self.active_context: Any = None

    def get_manifest(self) -> dict[str, Any]:
        """Return immutable provider and filtering provenance."""
        with self._connection() as connection:
            value = connection.execute(
                "SELECT value_json FROM catalog_metadata WHERE key='manifest'"
            ).fetchone()
        return json.loads(value[0])

    def get_candidates(
        self,
        filters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return compound candidates after optional provider/class filtering."""
        filters = dict(filters or {})
        clauses, values = [], []
        if filters.get("provider") is not None:
            clauses.append("provider = ?")
            values.append(str(filters["provider"]))
        if filters.get("formula") is not None:
            clauses.append("formula = ?")
            values.append(str(filters["formula"]))
        query = "SELECT * FROM candidate_compounds"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY compound_key"
        with self._connection() as connection:
            rows = connection.execute(query, values).fetchall()
            classes = connection.execute(
                "SELECT * FROM candidate_classes ORDER BY compound_key, depth"
            ).fetchall()
        classes_by_key: dict[str, list[dict[str, Any]]] = {}
        for row in classes:
            classes_by_key.setdefault(str(row["compound_key"]), []).append(dict(row))
        result = []
        for row in rows:
            record = dict(row)
            record["source_record"] = json.loads(record.pop("source_record_json"))
            record["classes"] = classes_by_key.get(record["compound_key"], [])
            result.append(record)
        return result

    def get_candidate_ions(
        self,
        filters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return candidate ions after optional m/z, adduct, and provider filters."""
        filters = dict(filters or {})
        clauses, values = [], []
        mapping = {
            "provider": "compound.provider",
            "adduct": "ion.adduct",
            "polarity": "ion.polarity",
            "formula": "ion.formula",
        }
        for key, column in mapping.items():
            if filters.get(key) is not None:
                clauses.append(f"{column} = ?")
                values.append(str(filters[key]))
        if filters.get("mz_min") is not None:
            clauses.append("ion.theoretical_mz >= ?")
            values.append(float(filters["mz_min"]))
        if filters.get("mz_max") is not None:
            clauses.append("ion.theoretical_mz <= ?")
            values.append(float(filters["mz_max"]))
        query = """
            SELECT ion.*, compound.provider, compound.provider_version,
                   compound.identifier, compound.name
            FROM candidate_ions AS ion
            JOIN candidate_compounds AS compound USING (compound_key)
        """
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY ion.theoretical_mz, ion.candidate_ion_key"
        with self._connection() as connection:
            return [dict(row) for row in connection.execute(query, values).fetchall()]

    def get_config(self) -> dict[str, Any]:
        """Return a portable reader configuration."""
        return {"type": "candidate_catalog", "path": str(self.path)}

    def _validate_schema(self) -> None:
        with self._connection() as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        expected = {
            "catalog_metadata",
            "candidate_compounds",
            "candidate_classes",
            "candidate_ions",
        }
        if expected - tables:
            raise ValueError("Candidate catalogue has an unsupported schema.")

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection


def build_candidate_catalog(
    *,
    workspace_path: Path | str,
    dataset_id: str,
    providers: Sequence[CandidateProvider],
    adducts: Sequence[str] | None = None,
    refresh_cache: bool = False,
    source_cache_dir: Path | str | None = DEFAULT_CANDIDATE_SOURCE_CACHE_DIR,
    refresh_sources: bool = False,
) -> Path:
    """Materialize external candidates compatible with one dataset's metadata.

    The existing local candidate SQLite is the dataset-level cache. If
    ``source_cache_dir`` is supplied, raw provider exports are cached there and
    re-used while rebuilding a catalogue. ``refresh_sources`` replaces those
    source exports independently of ``refresh_cache``.
    """
    if not providers:
        raise ValueError("At least one candidate provider is required.")
    layout = DatasetWorkspaceLayout(workspace_path)
    target = layout.candidate_catalog_path(dataset_id)
    if target.is_file() and not refresh_cache:
        logger.info("Using local candidate catalogue cache at %s", target)
        CandidateCatalogReader(target)
        return target
    metadata_artifact = read_candidate_metadata(
        workspace_path=workspace_path,
        dataset_id=dataset_id,
    )
    metadata = dict(metadata_artifact.get("metadata", {}))
    polarity = metadata.get("polarity")
    mz_min = _optional_float(metadata.get("mz_min"))
    mz_max = _optional_float(metadata.get("mz_max"))
    selected_adducts = tuple(str(value) for value in adducts) if adducts else None

    # Provider normalization
    ## Providers own source retrieval; cataloguing owns deterministic filtering.
    snapshot_capable = all(
        callable(getattr(provider, "materialize_snapshot", None))
        for provider in providers
    )
    resolved_source_cache_dir = (
        source_cache_dir if snapshot_capable else None
    )
    snapshot_paths = (
        materialize_candidate_sources(
            cache_dir=resolved_source_cache_dir,
            providers=providers,
            refresh_cache=refresh_sources,
        )
        if resolved_source_cache_dir is not None
        else (None,) * len(providers)
    )
    compound_by_key: dict[str, CandidateCompound] = {}
    for provider, snapshot_path in zip(providers, snapshot_paths, strict=True):
        candidates = (
            provider.fetch_candidates(snapshot_path)
            if snapshot_path is not None
            else provider.fetch_candidates()
        )
        for compound in candidates:
            compound_by_key.setdefault(compound.key, compound)
    ions = tuple(
        ion
        for compound in compound_by_key.values()
        for ion in _compatible_ions(
            compound,
            polarity=polarity,
            adducts=selected_adducts,
        )
        if (mz_min is None or ion.theoretical_mz >= mz_min)
        and (mz_max is None or ion.theoretical_mz <= mz_max)
    )
    used_keys = {ion.compound_key for ion in ions}
    compounds = tuple(
        compound for key, compound in compound_by_key.items() if key in used_keys
    )
    manifest = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "dataset_id": str(dataset_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {"name": provider.name, "version": provider.version}
            for provider in providers
        ],
        "filters": {
            "polarity": polarity,
            "mz_min": mz_min,
            "mz_max": mz_max,
            "adducts": selected_adducts,
        },
        "dataset_metadata_path": str(layout.dataset_metadata_path(dataset_id)),
        "source_cache_dir": (
            str(Path(resolved_source_cache_dir).resolve())
            if resolved_source_cache_dir is not None
            else None
        ),
    }
    CandidateCatalogWriter().write(
        path=target,
        compounds=compounds,
        ions=ions,
        manifest=manifest,
    )
    _write_manifest(layout.candidate_catalog_manifest_path(dataset_id), manifest)
    return target


def _create_schema(connection: sqlite3.Connection) -> None:
    """Create the fixed SQLite schema for one candidate catalogue."""
    connection.executescript(
        """
        CREATE TABLE catalog_metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        CREATE TABLE candidate_compounds (
            compound_key TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            provider_version TEXT NOT NULL,
            identifier TEXT NOT NULL,
            name TEXT NOT NULL,
            formula TEXT NOT NULL,
            monoisotopic_mass REAL,
            smiles TEXT,
            inchi TEXT,
            inchi_key TEXT,
            source_record_json TEXT NOT NULL
        );
        CREATE TABLE candidate_classes (
            compound_key TEXT NOT NULL,
            namespace TEXT NOT NULL,
            identifier TEXT NOT NULL,
            name TEXT NOT NULL,
            depth INTEGER NOT NULL,
            PRIMARY KEY (compound_key, namespace, identifier),
            FOREIGN KEY (compound_key) REFERENCES candidate_compounds(compound_key)
        );
        CREATE TABLE candidate_ions (
            candidate_ion_key TEXT PRIMARY KEY,
            compound_key TEXT NOT NULL,
            formula TEXT NOT NULL,
            adduct TEXT NOT NULL,
            charge INTEGER NOT NULL,
            polarity TEXT NOT NULL,
            theoretical_mz REAL NOT NULL,
            calculator_version TEXT NOT NULL,
            FOREIGN KEY (compound_key) REFERENCES candidate_compounds(compound_key)
        );
        CREATE INDEX candidate_ions_mz ON candidate_ions(theoretical_mz);
        CREATE INDEX candidate_ions_identity ON candidate_ions(formula, adduct);
        """
    )


def _insert_compounds(
    connection: sqlite3.Connection,
    compounds: Sequence[CandidateCompound],
) -> None:
    """Write compound rows and their ordered provider classifications."""
    connection.executemany(
        """
        INSERT INTO candidate_compounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                compound.key,
                compound.provider,
                compound.provider_version,
                compound.identifier,
                compound.name,
                compound.formula,
                compound.monoisotopic_mass,
                compound.smiles,
                compound.inchi,
                compound.inchi_key,
                _json(compound.source_record or {}),
            )
            for compound in compounds
        ],
    )
    connection.executemany(
        "INSERT INTO candidate_classes VALUES (?, ?, ?, ?, ?)",
        [
            (
                compound.key,
                item.namespace,
                item.identifier,
                item.name,
                item.depth,
            )
            for compound in compounds
            for item in compound.classes
        ],
    )


def _insert_ions(connection: sqlite3.Connection, ions: Sequence[CandidateIon]) -> None:
    """Write deterministic formula/adduct ion rows."""
    connection.executemany(
        "INSERT INTO candidate_ions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                ion.key,
                ion.compound_key,
                ion.formula,
                ion.adduct,
                ion.charge,
                ion.polarity,
                ion.theoretical_mz,
                ion.calculator_version,
            )
            for ion in ions
        ],
    )


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write the human-readable candidate provenance artifact atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _json(value: Any) -> str:
    """Serialize one SQLite JSON column deterministically."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _optional_float(value: Any) -> float | None:
    """Return one optional finite metadata float."""
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _compatible_ions(
    compound: CandidateCompound,
    *,
    polarity: str | None,
    adducts: tuple[str, ...] | None,
) -> tuple[CandidateIon, ...]:
    """Return ions supported by the current METASPACE mass calculator.

    Provider exports can include formulae with elements outside the explicitly
    supported monoisotopic mass table. Those records remain source-valid but
    cannot yet become model candidates, so the catalogue records a diagnostic
    and continues with the remaining provider population.
    """
    try:
        return make_candidate_ions(compound, polarity=polarity, adducts=adducts)
    except ValueError as error:
        logger.warning(
            "Skipping unsupported candidate %s from %s: %s",
            compound.identifier,
            compound.provider,
            error,
        )
        return ()
