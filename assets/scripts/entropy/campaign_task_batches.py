#!/usr/bin/env python3
"""Select parent-ready Slurm arrays from immutable materialized task descriptors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from msi_autoencoder_wrapper.runtime.output import task_fingerprint
from msi_autoencoder_wrapper.runtime.planning.graph import verify_plan_graph


INDEX_NAME = "task-index.json"


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping with bounded per-file memory."""
    with path.open(encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    """Hash one descriptor without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_index(plan: Path) -> dict[str, Any]:
    """Record descriptor identities once after staging finishes.

    :param plan: Materialized campaign plan directory.
    :type plan: pathlib.Path
    :return: Small checksum-verified task index.
    :rtype: dict[str, typing.Any]
    :raises ValueError: If task IDs or parent references are invalid.
    """
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted((plan / "tasks").glob("task_*.yaml")):
        task = _read_yaml(path)
        task_id = task["task_id"]
        if task_id != path.stem or task_id in entries:
            raise ValueError(f"Invalid or duplicate task identifier: {path}")
        entries[task_id] = {
            "descriptor_sha256": _sha256(path),
            "task_fingerprint": task_fingerprint(task),
            "depends_on": list(task.get("depends_on", [])),
            "model_name": task["parameters"].get("model_name"),
        }
    if not entries:
        raise ValueError("Cannot index an empty campaign plan.")
    if any(
        parent not in entries
        for entry in entries.values()
        for parent in entry["depends_on"]
    ):
        raise ValueError("A task depends on an unknown parent.")
    payload = {"schema_version": 1, "tasks": entries}
    index_path = plan / INDEX_NAME
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(index_path)
    return payload


def _load_index(plan: Path) -> dict[str, dict[str, Any]]:
    """Reject a stale index if any staged descriptor bytes have changed."""
    index_path = plan / INDEX_NAME
    if not index_path.is_file():
        build_index(plan)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    entries = payload.get("tasks")
    if payload.get("schema_version") != 1 or not isinstance(entries, dict) or not entries:
        raise ValueError("Invalid campaign task index.")
    paths = {path.stem: path for path in (plan / "tasks").glob("task_*.yaml")}
    if paths.keys() != entries.keys():
        raise ValueError("Task index and staged descriptors differ.")
    for task_id, path in paths.items():
        if _sha256(path) != entries[task_id]["descriptor_sha256"]:
            raise ValueError(f"Task descriptor changed after indexing: {task_id}")
    manifest_path = plan / "resolved-experiment.yaml"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as stream:
            if stream.readline().strip() == "runtime_schema_version: 3":
                verify_plan_graph(plan, expected_count=len(entries))
    return entries


def _completed(
    plan: Path,
    task_id: str,
    entry: dict[str, Any],
    local_workspace: Path | None,
) -> bool:
    """Check terminal fingerprint and an accessible or node-local model path."""
    status_path = plan / "status" / f"{task_id}.yaml"
    if not status_path.is_file():
        return False
    record = _read_yaml(status_path).get("records", {}).get(task_id, {})
    if record.get("task_fingerprint") != entry["task_fingerprint"]:
        raise ValueError(f"Invalid existing completion record: {task_id}")
    if record.get("status") != "completed":
        return False
    result = record.get("result")
    model_path = result.get("model_path") if isinstance(result, dict) else None
    if not isinstance(model_path, str) or Path(model_path).name != entry["model_name"]:
        raise ValueError(f"Completed task has wrong model path: {task_id}")
    model_dir = Path(model_path)
    if all((model_dir / "config" / name).is_file() for name in (
        "config.json", "history.json", "weights.pt"
    )):
        return True
    if local_workspace is not None:
        expected_local = local_workspace / "models/kidney" / entry["model_name"]
        if model_dir.resolve() == expected_local.resolve():
            # REMARK: The login node cannot see another node's local NVMe.
            # The same-node worker saved the model before writing completed status.
            return True
    raise ValueError(f"Completed task has no accessible model bundle: {task_id}")


def select_batch(
    plan: Path,
    limit: int,
    local_workspace: Path | None = None,
) -> list[int]:
    """Select a bounded batch whose parents are already completed.

    :param plan: Materialized campaign plan directory.
    :type plan: pathlib.Path
    :param limit: Maximum Slurm array size.
    :type limit: int
    :param local_workspace: Staging workspace on the execution node, if any.
    :type local_workspace: pathlib.Path | None
    :return: Ready task indices, or empty when the plan is complete.
    :rtype: list[int]
    :raises ValueError: If a descriptor, dependency or status is inconsistent.
    """
    if limit < 1:
        raise ValueError("Batch limit must be positive.")
    entries = _load_index(plan)
    completed = {
        task_id for task_id, entry in entries.items()
        if _completed(plan, task_id, entry, local_workspace)
    }
    pending = set(entries) - completed
    if not pending:
        return []
    ready = sorted(
        task_id for task_id in pending
        if set(entries[task_id]["depends_on"]) <= completed
    )
    if not ready:
        raise ValueError("No ready task: dependency cycle or incomplete parent.")
    indices = []
    for task_id in ready[:limit]:
        prefix, separator, number = task_id.rpartition("_")
        if prefix != "task" or separator != "_" or not number.isdigit():
            raise ValueError(f"Invalid Slurm task identifier: {task_id}")
        indices.append(int(number))
    return indices


def verify_batch(
    plan: Path,
    indices: list[int],
    local_workspace: Path | None = None,
) -> None:
    """Require each submitted task to complete with its exact fingerprint."""
    if not indices:
        raise ValueError("Cannot verify an empty batch.")
    entries = _load_index(plan)
    for index in indices:
        task_id = f"task_{index:06d}"
        if task_id not in entries or not _completed(
            plan, task_id, entries[task_id], local_workspace
        ):
            raise ValueError(f"Submitted task did not complete: {task_id}")


def main() -> int:
    """Build a task index, print one sparse array, or verify an array."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("index", "next", "verify"))
    parser.add_argument("--plan-directory", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--indices", default="")
    parser.add_argument("--local-workspace", type=Path)
    args = parser.parse_args()
    if args.command == "index":
        print(len(build_index(args.plan_directory)["tasks"]))
    elif args.command == "next":
        print(",".join(map(str, select_batch(
            args.plan_directory, args.limit, args.local_workspace
        ))))
    else:
        verify_batch(
            args.plan_directory,
            [int(value) for value in args.indices.split(",") if value],
            args.local_workspace,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
