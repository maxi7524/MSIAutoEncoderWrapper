"""Persistent task and report status records."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def update_manifest(path: Path, identifier: str, values: dict[str, Any]) -> None:
    """Update one manifest record using an atomic filesystem replacement."""
    payload: dict[str, Any] = {"records": {}}
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
        if isinstance(loaded, dict):
            payload = loaded
    records = payload.setdefault("records", {})
    record = records.setdefault(identifier, {})
    record.update(values)
    record["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)
    temporary.replace(path)
