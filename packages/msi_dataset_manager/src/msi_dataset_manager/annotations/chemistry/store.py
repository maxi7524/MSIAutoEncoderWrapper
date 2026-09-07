"""Additive SQLite enrichment with explicit candidate sets and provenance."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping

from .formula import parse_formula
from .providers import ChemistryProvider
from ...utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def enrich_annotations(records: Iterable[Mapping[str, Any]], path: Path | str,
                       provider: ChemistryProvider) -> int:
    """Append chemical descriptors to an existing annotation DB or a new sidecar.

    :param records: Normalized annotation records with formula and adduct.
    :param path: SQLite destination; existing annotation tables remain unchanged.
    :param provider: Versioned local or remote lookup implementation.
    :return: Number of newly enriched ion identities.
    :raises ValueError: If a candidate formula contradicts its annotation.
    :raises Exception: Provider failures propagate and roll back this enrichment.
    """
    # Resolve all source identifiers per ion before contacting a provider
    grouped: dict[tuple[str, str], set[str]] = {}
    for record in records:
        identity = str(record["formula"]), str(record.get("adduct") or "")
        nested = record.get("source_record") or {}
        values = record.get("molecule_ids") or nested.get("molecule_ids") or nested.get("moleculeIds") or []
        if isinstance(values, str):
            values = [values]
        grouped.setdefault(identity, set()).update(str(value) for value in values)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(destination) as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS chemistry_annotations (
            formula TEXT NOT NULL, adduct TEXT NOT NULL, provider TEXT NOT NULL,
            provider_version TEXT NOT NULL, schema_version INTEGER NOT NULL,
            atom_counts_json TEXT NOT NULL, candidates_json TEXT NOT NULL,
            source_ids_json TEXT NOT NULL, status TEXT NOT NULL,
            PRIMARY KEY (formula, adduct, provider, provider_version))""")
        updated = 0
        for (formula, adduct), identifiers in sorted(grouped.items()):
            key = (formula, adduct, provider.name, provider.version)
            existing = connection.execute(
                "SELECT source_ids_json FROM chemistry_annotations WHERE formula=? AND adduct=? AND provider=? AND provider_version=?", key,
            ).fetchone()
            if existing is not None and set(json.loads(existing[0])) == identifiers:
                continue
            counts = parse_formula(formula)
            try:
                candidates = provider.lookup(formula, tuple(sorted(identifiers)))
            except Exception:
                logger.error("Chemical enrichment failed for provider %s; rolling back.", provider.name, exc_info=True)
                raise
            if any(parse_formula(candidate.formula) != counts for candidate in candidates):
                raise ValueError("Chemical lookup returned a conflicting molecular formula.")
            status = "unresolved" if not candidates else "unique" if len(candidates) == 1 else "ambiguous"
            connection.execute("""INSERT OR REPLACE INTO chemistry_annotations
                VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)""",
                (*key, json.dumps(counts, sort_keys=True),
                 json.dumps([c.get_config() for c in candidates], sort_keys=True),
                 json.dumps(sorted(identifiers)), status))
            updated += 1
        logger.info("Enriched %s ion identities using %s (%s).", updated, provider.name, provider.version)
    return updated


def read_chemistry(path: Path | str, *, provider: str, version: str) -> dict[str, dict[str, Any]]:
    """Read one explicit frozen snapshot without write or network access.

    :param path: Enriched annotation database or sidecar.
    :param provider: Exact provider name.
    :param version: Exact snapshot version.
    :return: Records keyed by canonical formula/adduct identity.
    :raises ValueError: If the snapshot is missing or has an unsupported schema.
    """
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM chemistry_annotations WHERE provider=? AND provider_version=?", (provider, version)).fetchall()
    if not rows:
        raise ValueError("Requested chemical snapshot is empty or unavailable.")
    result = {}
    for row in rows:
        if row["schema_version"] != 1:
            raise ValueError("Unsupported chemical annotation schema version.")
        candidates = json.loads(row["candidates_json"])
        class_sets = [set(c["classes"]) for c in candidates]
        certain = sorted(set.intersection(*class_sets)) if class_sets else []
        possible = sorted(set.union(*class_sets)) if class_sets else []
        result[f"{row['formula']}|{row['adduct']}"] = {
            "atom_counts": json.loads(row["atom_counts_json"]),
            "candidates": candidates, "certain_classes": certain,
            "possible_classes": possible, "status": row["status"],
        }
    return result
