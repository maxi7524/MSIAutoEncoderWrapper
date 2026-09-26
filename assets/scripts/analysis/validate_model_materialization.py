#!/usr/bin/env python3
"""Validate persisted model materialization for a runtime campaign.

The validator consumes a completed runtime campaign directory, not the source
experiment YAML.  It therefore checks the artifacts that were actually written by
the orchestrator rather than accepting an intended phase layout as evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from msi_autoencoder_wrapper.runtime.planning.graph import verify_plan_graph
from msi_autoencoder_wrapper.utils.logger import get_custom_logger


logger = get_custom_logger(__name__)



def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping.

    :param path: YAML file to read.
    :type path: pathlib.Path
    :return: Parsed mapping.
    :rtype: dict[str, typing.Any]
    :raises ValueError: If the document root is not a mapping.
    """
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"YAML document must contain a mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file.

    :param path: File to hash.
    :type path: pathlib.Path
    :return: Lower-case hexadecimal SHA-256 digest.
    :rtype: str
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_fingerprint(task: dict[str, Any]) -> str:
    """Reproduce the runtime fingerprint of a materialized task descriptor.

    :param task: Materialized task mapping.
    :type task: dict[str, typing.Any]
    :return: Stable SHA-256 digest.
    :rtype: str
    """
    serialized = yaml.safe_dump(task, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load and structurally validate a persisted state dictionary.

    :param path: ``weights.pt`` path.
    :type path: pathlib.Path
    :return: Non-empty tensor state dictionary loaded on CPU.
    :rtype: dict[str, torch.Tensor]
    :raises ValueError: If the payload is empty or contains non-tensor values.
    """
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not value:
        raise ValueError("weights.pt must contain a non-empty state dictionary")
    non_tensors = sorted(
        key
        for key, tensor in value.items()
        if not isinstance(tensor, torch.Tensor)
    )
    if non_tensors:
        raise ValueError(f"state dictionary contains non-tensor entries: {non_tensors}")
    return value


def _states_have_same_structure(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> bool:
    """Return whether state dictionaries have identical keys, shapes and dtypes."""
    return left.keys() == right.keys() and all(
        left[name].shape == right[name].shape and left[name].dtype == right[name].dtype
        for name in left
    )


def validate_campaign(campaign_directory: Path | str) -> list[str]:
    """Validate materialized model bundles and declared workflow lineage.

    Workflow roles and group sizes are read from the campaign manifest. Tasks
    without workflow metadata are still checked for completed status and valid
    persisted model bundles. A task with ``workflow.parent_task_id`` must declare
    that parent in ``depends_on`` and record the loaded artifact in
    ``result.initialization``.

    :param campaign_directory: Directory containing ``resolved-experiment.yaml``,
        ``tasks/`` and ``status/``.
    :type campaign_directory: pathlib.Path | str
    :return: Human-readable contract violations. An empty list means PASS.
    :rtype: list[str]
    """
    root = Path(campaign_directory).resolve()
    issues: list[str] = []
    manifest_path = root / "resolved-experiment.yaml"
    if not manifest_path.is_file():
        return [f"missing campaign manifest: {manifest_path}"]

    try:
        manifest = _read_yaml(manifest_path)
    except (OSError, ValueError, yaml.YAMLError) as error:
        return [f"cannot read campaign manifest '{manifest_path}': {error}"]
    schema_version = manifest.get("runtime_schema_version")
    if schema_version not in {2, 3}:
        issues.append("campaign manifest runtime_schema_version must be 2 or 3")
    elif schema_version == 3:
        try:
            verify_plan_graph(root)
        except (OSError, ValueError, yaml.YAMLError) as error:
            issues.append(f"invalid campaign plan graph: {error}")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return ["campaign manifest must contain a non-empty tasks list"]

    # Task graph contract
    ## Index task identifiers and any declared workflow groups.
    tasks_by_id: dict[str, dict[str, Any]] = {}
    parents: dict[str, str] = {}
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            issues.append(f"tasks[{index}] is not a mapping")
            continue
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            issues.append(f"tasks[{index}] has no non-empty task_id")
            continue
        if task_id in tasks_by_id:
            issues.append(f"duplicate task_id: {task_id}")
            continue
        tasks_by_id[task_id] = task
        workflow = task.get("workflow")
        if workflow is None:
            continue
        if not isinstance(workflow, dict):
            issues.append(f"{task_id}: workflow must be a mapping or null")
            continue
        group_id = workflow.get("group_id")
        role = workflow.get("role")
        if not isinstance(group_id, str) or not group_id:
            issues.append(f"{task_id}: workflow.group_id must be non-empty")
            continue
        if not isinstance(role, str) or not role:
            issues.append(f"{task_id}: workflow.role must be non-empty")
            continue
        parent_id = workflow.get("parent_task_id")
        if parent_id is not None:
            if not isinstance(parent_id, str) or not parent_id:
                issues.append(f"{task_id}: workflow.parent_task_id must be a task ID or null")
            else:
                parents[task_id] = parent_id
                dependencies = task.get("depends_on", [])
                if not isinstance(dependencies, list):
                    issues.append(f"{task_id}: depends_on must be a list")
                elif parent_id not in dependencies:
                    issues.append(
                        f"{task_id}: depends_on does not include workflow parent {parent_id}"
                    )

    ## Validate declared parent references against the same workflow group.
    for child_id, parent_id in sorted(parents.items()):
        parent = tasks_by_id.get(parent_id)
        if parent is None:
            issues.append(f"{child_id}: workflow parent task does not exist: {parent_id}")
            continue
        child_workflow = tasks_by_id[child_id].get("workflow") or {}
        parent_workflow = parent.get("workflow") or {}
        if child_workflow.get("group_id") != parent_workflow.get("group_id"):
            issues.append(f"{child_id}: workflow parent {parent_id} belongs to another group")

    # Persisted artifact contract
    ## Validate terminal status, unique model paths and loadable model bundles.
    records: dict[str, dict[str, Any]] = {}
    model_paths: dict[str, Path] = {}
    weight_paths: dict[str, Path] = {}
    states: dict[str, dict[str, torch.Tensor]] = {}
    path_owners: dict[Path, str] = {}
    for task_id, manifest_task in sorted(tasks_by_id.items()):
        descriptor_path = root / "tasks" / f"{task_id}.yaml"
        if not descriptor_path.is_file():
            issues.append(f"{task_id}: missing materialized task descriptor {descriptor_path}")
            continue
        try:
            descriptor = _read_yaml(descriptor_path)
        except (OSError, ValueError, yaml.YAMLError) as error:
            issues.append(f"{task_id}: cannot read task descriptor: {error}")
            continue
        if schema_version == 2 and descriptor != manifest_task:
            issues.append(f"{task_id}: task descriptor differs from resolved manifest")
        elif schema_version == 3 and any(
            descriptor.get(key) != manifest_task.get(key)
            for key in ("task_id", "grid_id", "repetition", "workflow", "depends_on")
        ):
            issues.append(f"{task_id}: task descriptor differs from plan graph node")

        status_path = root / "status" / f"{task_id}.yaml"
        if not status_path.is_file():
            issues.append(f"{task_id}: missing terminal status {status_path}")
            continue
        try:
            status = _read_yaml(status_path)
        except (OSError, ValueError, yaml.YAMLError) as error:
            issues.append(f"{task_id}: cannot read terminal status: {error}")
            continue
        record = status.get("records", {}).get(task_id)
        if not isinstance(record, dict):
            issues.append(f"{task_id}: terminal status has no matching record")
            continue
        records[task_id] = record
        if record.get("status") != "completed":
            issues.append(f"{task_id}: status is {record.get('status')!r}, expected 'completed'")
            continue
        expected_fingerprint = _task_fingerprint(descriptor)
        if record.get("task_fingerprint") != expected_fingerprint:
            issues.append(f"{task_id}: terminal task fingerprint does not match descriptor")
        result = record.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("model_path"), str):
            issues.append(f"{task_id}: completed result has no model_path")
            continue
        model_path = Path(result["model_path"])
        if not model_path.is_absolute():
            issues.append(f"{task_id}: model_path is not absolute: {model_path}")
            continue
        model_path = model_path.resolve()
        previous_owner = path_owners.get(model_path)
        if previous_owner is not None:
            issues.append(f"{task_id}: model_path is shared with {previous_owner}: {model_path}")
        else:
            path_owners[model_path] = task_id
        model_paths[task_id] = model_path
        required = {
            "config": model_path / "config" / "config.json",
            "weights": model_path / "config" / "weights.pt",
            "history": model_path / "config" / "history.json",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            issues.append(f"{task_id}: model bundle is missing {missing} under {model_path}")
            continue
        try:
            config = json.loads(required["config"].read_text(encoding="utf-8"))
            history = json.loads(required["history"].read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("config.json root is not an object")
            if not isinstance(history, list):
                raise ValueError("history.json root is not a list")
            states[task_id] = _load_state_dict(required["weights"])
            weight_paths[task_id] = required["weights"]
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as error:
            issues.append(f"{task_id}: invalid model bundle: {error}")

    # On-disk lineage contract
    ## Compare each declared parent link with persisted initialization provenance.
    for child_id, parent_id in sorted(parents.items()):
        if parent_id not in weight_paths or parent_id not in states:
            continue
        parent_weights = weight_paths[parent_id]
        parent_hash = _sha256(parent_weights)
        parent_state = states[parent_id]
        record = records.get(child_id, {})
        result = record.get("result", {}) if isinstance(record, dict) else {}
        initialization = result.get("initialization") if isinstance(result, dict) else None
        if not isinstance(initialization, dict):
            issues.append(f"{child_id}: result.initialization provenance is missing")
            continue
        expected_parent_path = model_paths.get(parent_id)
        reported_parent_path = initialization.get("model_path")
        reported_resolved = (
            Path(reported_parent_path).resolve()
            if isinstance(reported_parent_path, str)
            else None
        )
        if initialization.get("parent_task_id") != parent_id:
            issues.append(f"{child_id}: initialization.parent_task_id is not {parent_id}")
        if reported_resolved != expected_parent_path:
            issues.append(f"{child_id}: initialization.model_path does not identify parent artifact")
        if initialization.get("weights_sha256") != parent_hash:
            issues.append(f"{child_id}: initialization.weights_sha256 does not match parent weights")

        child_state = states.get(child_id)
        if child_state is not None and not _states_have_same_structure(parent_state, child_state):
            issues.append(f"{child_id}: state-dictionary structure differs from parent model")


    return issues


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for artifact validation."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate completed model artifacts and any declared workflow lineage "
            "in a runtime campaign directory."
        )
    )
    parser.add_argument("campaign_directory", type=Path)
    return parser


def main() -> int:
    """Run validation and return a shell-compatible status code."""
    args = build_parser().parse_args()
    issues = validate_campaign(args.campaign_directory)
    if issues:
        logger.error("Campaign model materialization: FAIL (%s violation(s)).", len(issues))
        for issue in issues:
            logger.error("- %s", issue)
        return 1
    logger.info("Campaign model materialization: PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
