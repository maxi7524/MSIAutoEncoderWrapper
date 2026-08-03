"""Validation for canonical SQLite annotation stores."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ...utils.exceptions import raise_validation_error


def validate_annotation_store(path: Path | str) -> None:
    """Verify that a SQLite file exposes the canonical annotation tables.

    :param path: SQLite catalog path.
    :type path: pathlib.Path | str
    :raises ValidationError: If the file is missing required canonical tables.
    """
    target = Path(path)
    if not target.is_file():
        raise_validation_error("AnnotationStore", f"SQLite store '{target}' does not exist.")
    with sqlite3.connect(target) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    required = {"datasets", "annotations", "spectrum_annotations"}
    missing = sorted(required - tables)
    if missing:
        raise_validation_error(
            "AnnotationStore", f"SQLite store is missing tables: {', '.join(missing)}."
        )
