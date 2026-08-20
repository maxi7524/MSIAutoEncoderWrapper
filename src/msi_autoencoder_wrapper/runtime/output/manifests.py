"""Persistent task and report status records."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
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


def task_fingerprint(task: dict[str, Any]) -> str:
    """Return a stable content fingerprint for one materialized task."""
    serialized = yaml.safe_dump(task, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def is_completed_task(path: Path, task: dict[str, Any]) -> bool:
    """Return whether a validated terminal record matches the current task."""
    if not path.is_file():
        return False
    with path.open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, dict):
        return False
    record = manifest.get("records", {}).get(task["task_id"], {})
    return (
        record.get("status") == "completed"
        and record.get("task_fingerprint") == task_fingerprint(task)
    )
