#!/usr/bin/env python3
"""Rebind completed task results from node-local to persistent model bundles."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import yaml

from msi_autoencoder_wrapper.runtime.output import is_completed_task, update_manifest


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping."""
    with path.open(encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one model artifact file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize_paths(plan: Path, local_workspace: Path, source_workspace: Path) -> int:
    """Verify all completed models then persist their source-workspace paths.

    :param plan: Materialized campaign plan directory.
    :type plan: pathlib.Path
    :param local_workspace: Node-local staging workspace.
    :type local_workspace: pathlib.Path
    :param source_workspace: Persistent workspace.
    :type source_workspace: pathlib.Path
    :return: Number of task result records whose paths changed.
    :rtype: int
    :raises ValueError: If a result or its persisted artifact is inconsistent.
    """
    task_paths = sorted((plan / "tasks").glob("task_*.yaml"))
    records: dict[str, tuple[Path, dict[str, Any], str]] = {}
    updates: list[tuple[Path, str, dict[str, Any]]] = []

    # Validate the complete graph before mutating any status record.
    for task_path in task_paths:
        task = _read_yaml(task_path)
        task_id = task["task_id"]
        if task_id != task_path.stem or task_id in records:
            raise ValueError(f"Invalid or duplicate task identifier: {task_path}")
        status_path = plan / "status" / f"{task_id}.yaml"
        if not is_completed_task(status_path, task):
            raise ValueError(f"Task is not completed with matching fingerprint: {task_id}")
        record = _read_yaml(status_path)["records"][task_id]
        result = record.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("model_path"), str):
            raise ValueError(f"Task has no model result: {task_id}")
        records[task_id] = (status_path, result, task["parameters"]["model_name"])

    for task_id, (status_path, result, model_name) in records.items():
        persistent = source_workspace / "models" / "kidney" / model_name
        local = local_workspace / "models" / "kidney" / model_name
        if Path(result["model_path"]).resolve() not in (persistent.resolve(), local.resolve()):
            raise ValueError(f"Unexpected result path for {task_id}: {result['model_path']}")
        if not all((persistent / "config" / name).is_file() for name in (
            "config.json", "history.json", "weights.pt"
        )):
            raise ValueError(f"Persistent model is incomplete: {persistent}")
        changed = Path(result["model_path"]).resolve() != persistent.resolve()
        if changed:
            for name in ("config.json", "history.json", "weights.pt"):
                if _sha256(local / "config" / name) != _sha256(persistent / "config" / name):
                    raise ValueError(f"Persistent model file differs: {task_id}/{name}")
        updated = dict(result)
        updated["model_path"] = str(persistent.resolve())
        initialization = updated.get("initialization")
        if isinstance(initialization, dict):
            parent_id = initialization.get("parent_task_id")
            if parent_id not in records:
                raise ValueError(f"Unknown recorded parent for {task_id}: {parent_id}")
            parent_name = records[parent_id][2]
            parent_persistent = source_workspace / "models" / "kidney" / parent_name
            parent_local = local_workspace / "models" / "kidney" / parent_name
            if Path(initialization.get("model_path", "")).resolve() not in (
                parent_local.resolve(), parent_persistent.resolve()
            ):
                raise ValueError(f"Unexpected recorded parent path for {task_id}")
            updated["initialization"] = {
                **initialization, "model_path": str(parent_persistent.resolve())
            }
            changed |= updated["initialization"] != initialization
        if changed:
            updates.append((status_path, task_id, updated))

    # Commit verified path changes without altering the task fingerprint.
    for status_path, task_id, result in updates:
        update_manifest(status_path, task_id, {"result": result})
    return len(updates)


def main() -> int:
    """Rewrite all verified completed result paths after model synchronization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-directory", type=Path, required=True)
    parser.add_argument("--local-workspace", type=Path, required=True)
    parser.add_argument("--source-workspace", type=Path, required=True)
    args = parser.parse_args()
    print(finalize_paths(args.plan_directory, args.local_workspace, args.source_workspace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
