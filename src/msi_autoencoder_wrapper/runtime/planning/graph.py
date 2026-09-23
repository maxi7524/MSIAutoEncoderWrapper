"""Validate compact campaign indexes and shared source-population references."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from ...utils.logger import get_custom_logger


logger = get_custom_logger(__name__)
RESOLVED_YAML_KEYS = frozenset({
    "model_config", "context_config", "binner_config",
    "inverse_binner_config", "split_manifest",
})


def _read_mapping(path: Path) -> dict[str, Any]:
    """Read one YAML mapping without retaining other descriptor documents."""
    with path.open(encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    """Hash a file in bounded-size blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shared_source_reference(task: dict[str, Any]) -> dict[str, str] | None:
    """Return a task's shared source-index reference when configured."""
    subset = (
        task.get("parameters", {})
        .get("factory_parameters", {})
        .get("dataset", {})
        .get("parameters", {})
        .get("subset", {})
    )
    if not isinstance(subset, dict) or "indices_ref" not in subset:
        return None
    if subset.get("method") != "source_indices" or "indices" in subset:
        raise ValueError(f"Invalid compact subset in {task['task_id']}.")
    reference = subset["indices_ref"]
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"Invalid source-index reference in {task['task_id']}.")
    return reference


def verify_plan_graph(directory: Path, expected_count: int | None = None) -> int:
    """Verify graph nodes, lineage and shared YAMLs before task publication.

    :param directory: Materialized plan directory.
    :type directory: pathlib.Path
    :param expected_count: Optional exact number of task descriptors.
    :type expected_count: int | None
    :return: Number of validated task nodes.
    :rtype: int
    :raises ValueError: If the graph is incomplete, cyclic or inconsistent.
    """
    root = directory.resolve()
    manifest = _read_mapping(root / "resolved-experiment.yaml")
    if manifest.get("runtime_schema_version") != 3:
        raise ValueError("Compact plan graph requires runtime schema version 3.")
    nodes = manifest.get("tasks")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Plan graph contains no task nodes.")
    if expected_count is not None and len(nodes) != expected_count:
        raise ValueError("Plan graph task count differs from the expected count.")

    # Task nodes
    ## Compare each compact index entry to its independently materialized task.
    node_ids: set[str] = set()
    dependencies: dict[str, set[str]] = {}
    shared_refs: dict[Path, str] = {}
    resolved_artifacts: set[Path] = set()
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("task_id"), str):
            raise ValueError("Plan graph contains an invalid task node.")
        task_id = node["task_id"]
        if re.fullmatch(r"task_[0-9]{6}", task_id) is None:
            raise ValueError(f"Invalid task identifier: {task_id}")
        if task_id in node_ids:
            raise ValueError(f"Duplicate task node: {task_id}")
        node_ids.add(task_id)
        task = _read_mapping(root / "tasks" / f"{task_id}.yaml")
        if any(task.get(key) != node.get(key) for key in (
            "task_id", "grid_id", "repetition", "workflow", "depends_on"
        )):
            raise ValueError(f"Plan graph node differs from descriptor: {task_id}")
        parents = node["depends_on"]
        if not isinstance(parents, list) or not all(isinstance(v, str) for v in parents):
            raise ValueError(f"Invalid dependency list: {task_id}")
        dependencies[task_id] = set(parents)
        resolved = task.get("parameters", {}).get("resolved", {})
        if not isinstance(resolved, dict):
            raise ValueError(f"Invalid resolved artifact references: {task_id}")
        for name, value in resolved.items():
            if name not in RESOLVED_YAML_KEYS:
                continue
            if not isinstance(value, str):
                raise ValueError(f"Invalid resolved artifact path {name}: {task_id}")
            path = Path(value)
            if not path.is_absolute():
                raise ValueError(f"Resolved artifact path must be absolute: {task_id}/{name}")
            resolved_artifacts.add(path)
        reference = _shared_source_reference(task)
        if reference is not None:
            if not isinstance(reference["path"], str):
                raise ValueError(f"Invalid shared source path: {task_id}")
            path = Path(reference["path"])
            allowed = root / "resolved" / "source_populations"
            if not path.is_absolute() or path.resolve().parent != allowed:
                raise ValueError(f"Shared source file escapes campaign plan: {task_id}")
            digest = reference["sha256"]
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"Invalid shared source checksum: {task_id}")
            previous = shared_refs.setdefault(path, digest)
            if previous != digest:
                raise ValueError(f"Conflicting shared source checksums: {path}")

    descriptor_ids = {path.stem for path in (root / "tasks").glob("task_*.yaml")}
    if descriptor_ids != node_ids:
        raise ValueError("Task descriptors differ from campaign graph nodes.")

    # Dependency edges
    ## Every parent exists and a topological traversal consumes every node.
    if any(not parents <= node_ids for parents in dependencies.values()):
        raise ValueError("Plan graph references an unknown parent task.")
    remaining = dict(dependencies)
    while remaining:
        ready = {name for name, parents in remaining.items() if not parents & remaining.keys()}
        if not ready:
            raise ValueError("Plan graph contains a dependency cycle.")
        for name in ready:
            del remaining[name]

    # Shared YAML leaves
    ## Validate each existing resolved artifact once, then checksum source populations.
    for path in resolved_artifacts:
        _read_mapping(path)
    for path, digest in shared_refs.items():
        if _sha256(path) != digest:
            raise ValueError(f"Shared source checksum mismatch: {path}")
        values = _read_mapping(path).get("indices")
        if (
            not isinstance(values, list)
            or not values
            or any(isinstance(v, bool) or not isinstance(v, int) for v in values)
            or values != sorted(set(values))
        ):
            raise ValueError(f"Invalid shared source indices: {path}")
    logger.info(
        "Verified plan graph: %s tasks, %s resolved YAMLs, %s source populations.",
        len(node_ids), len(resolved_artifacts), len(shared_refs),
    )
    return len(node_ids)
