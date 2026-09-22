#!/usr/bin/env python3
"""Validate persisted model materialization for the pretraining campaign.

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

from msi_autoencoder_wrapper.utils.logger import get_custom_logger


logger = get_custom_logger(__name__)

EXPECTED_BRANCH_ROLES = frozenset({"pretrained", "frozen_head", "unfrozen_head"})
BASELINE_ROLE = "real_only"
DEFAULT_HEAD_PREFIX = "heads.molecule_vpu."
EXPECTED_BRANCH_GROUP_COUNT = 100
EXPECTED_BASELINE_GROUP_COUNT = 10


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


def _changed_keys(
    parent: dict[str, torch.Tensor],
    child: dict[str, torch.Tensor],
) -> set[str]:
    """Return tensor keys whose persisted values differ exactly."""
    return {
        name
        for name in parent
        if name in child and not torch.equal(parent[name], child[name])
    }


def validate_campaign(
    campaign_directory: Path | str,
    *,
    head_prefix: str = DEFAULT_HEAD_PREFIX,
    expected_branch_groups: int = EXPECTED_BRANCH_GROUP_COUNT,
    expected_baseline_groups: int = EXPECTED_BASELINE_GROUP_COUNT,
) -> list[str]:
    """Validate task lineage and separately persisted model artifacts.

    Expected task descriptors contain a top-level ``workflow`` mapping with
    ``group_id``, ``role`` and ``parent_task_id`` fields.  Dependent tasks also
    contain a top-level ``depends_on`` list.  Completed child results record the
    exact loaded parent under ``result.initialization`` using ``parent_task_id``,
    ``model_path`` and ``weights_sha256``.

    :param campaign_directory: Directory containing ``resolved-experiment.yaml``,
        ``tasks/`` and ``status/``.
    :type campaign_directory: pathlib.Path | str
    :param head_prefix: State-dictionary prefix frozen during real-data adaptation.
    :type head_prefix: str
    :param expected_branch_groups: Required synthetic schedule/axis/repetition groups.
    :type expected_branch_groups: int
    :param expected_baseline_groups: Required real-only axis/repetition groups.
    :type expected_baseline_groups: int
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
    if manifest.get("runtime_schema_version") != 2:
        issues.append(
            "campaign manifest runtime_schema_version must be 2 for workflow dependencies"
        )
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return ["campaign manifest must contain a non-empty tasks list"]

    # Task graph contract
    ## Index every task and enforce one complete artifact-role set per group.
    tasks_by_id: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, str]] = {}
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
        if not isinstance(workflow, dict):
            issues.append(f"{task_id}: missing workflow materialization metadata")
            continue
        group_id = workflow.get("group_id")
        role = workflow.get("role")
        if not isinstance(group_id, str) or not group_id:
            issues.append(f"{task_id}: workflow.group_id must be non-empty")
            continue
        if role not in EXPECTED_BRANCH_ROLES | {BASELINE_ROLE}:
            issues.append(f"{task_id}: unsupported workflow.role {role!r}")
            continue
        role_tasks = groups.setdefault(group_id, {})
        if role in role_tasks:
            issues.append(
                f"{group_id}: role {role!r} is duplicated by {role_tasks[role]} and {task_id}"
            )
        role_tasks[role] = task_id

    for group_id, role_tasks in sorted(groups.items()):
        roles = set(role_tasks)
        expected = {BASELINE_ROLE} if BASELINE_ROLE in roles else set(EXPECTED_BRANCH_ROLES)
        if roles != expected:
            issues.append(
                f"{group_id}: materialized roles are {sorted(roles)}, expected {sorted(expected)}"
            )
            continue
        if roles == {BASELINE_ROLE}:
            baseline = tasks_by_id[role_tasks[BASELINE_ROLE]]
            if baseline.get("depends_on", []) != []:
                issues.append(f"{baseline['task_id']}: real_only must not depend on another task")
            continue
        parent_id = role_tasks["pretrained"]
        parent = tasks_by_id[parent_id]
        if parent.get("depends_on", []) != []:
            issues.append(f"{parent_id}: pretrained must not depend on another task")
        for role in ("frozen_head", "unfrozen_head"):
            child_id = role_tasks[role]
            child = tasks_by_id[child_id]
            if child.get("depends_on") != [parent_id]:
                issues.append(
                    f"{child_id}: depends_on must contain only pretrained task {parent_id}"
                )
            if child.get("workflow", {}).get("parent_task_id") != parent_id:
                issues.append(
                    f"{child_id}: workflow.parent_task_id must equal {parent_id}"
                )
        if parent.get("workflow", {}).get("parent_task_id") is not None:
            issues.append(f"{parent_id}: pretrained workflow.parent_task_id must be null")

    branch_group_count = sum(
        set(role_tasks) != {BASELINE_ROLE} for role_tasks in groups.values()
    )
    baseline_group_count = sum(
        set(role_tasks) == {BASELINE_ROLE} for role_tasks in groups.values()
    )
    expected_task_count = expected_branch_groups * len(EXPECTED_BRANCH_ROLES) + (
        expected_baseline_groups
    )
    if branch_group_count != expected_branch_groups:
        issues.append(
            "campaign has "
            f"{branch_group_count} synthetic workflow groups, expected {expected_branch_groups}"
        )
    if baseline_group_count != expected_baseline_groups:
        issues.append(
            "campaign has "
            f"{baseline_group_count} real-only workflow groups, expected {expected_baseline_groups}"
        )
    if len(tasks_by_id) != expected_task_count:
        issues.append(
            f"campaign has {len(tasks_by_id)} tasks, expected {expected_task_count} "
            "separately persisted models"
        )

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
        if descriptor != manifest_task:
            issues.append(f"{task_id}: task descriptor differs from resolved manifest")

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
        role = manifest_task.get("workflow", {}).get("role")
        if isinstance(role, str) and not model_path.name.endswith(f"__{role}"):
            issues.append(
                f"{task_id}: model directory name must end with '__{role}': "
                f"{model_path.name}"
            )
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
    ## Prove that both adaptation tasks loaded the same persisted parent weights.
    for _group_id, role_tasks in sorted(groups.items()):
        parent_id = role_tasks.get("pretrained")
        if parent_id is None:
            continue
        if parent_id not in weight_paths or parent_id not in states:
            continue
        parent_weights = weight_paths[parent_id]
        parent_hash = _sha256(parent_weights)
        parent_state = states[parent_id]
        for role in ("frozen_head", "unfrozen_head"):
            child_id = role_tasks.get(role)
            if child_id is None:
                continue
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
            if child_state is None:
                continue
            if not _states_have_same_structure(parent_state, child_state):
                issues.append(f"{child_id}: state-dictionary structure differs from pretrained parent")
                continue
            changed = _changed_keys(parent_state, child_state)
            if not changed:
                issues.append(f"{child_id}: no persisted tensor changed from pretrained parent")
            if role == "frozen_head":
                head_keys = {name for name in parent_state if name.startswith(head_prefix)}
                if not head_keys:
                    issues.append(
                        f"{child_id}: pretrained state has no tensors with prefix {head_prefix!r}"
                    )
                changed_head = sorted(head_keys & changed)
                if changed_head:
                    issues.append(f"{child_id}: frozen head tensors changed: {changed_head}")
                if not (changed - head_keys):
                    issues.append(f"{child_id}: no non-head tensor changed during adaptation")

    return issues


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for artifact validation."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate that every pretraining workflow task produced an independent "
            "model bundle with auditable parent-weight lineage."
        )
    )
    parser.add_argument("campaign_directory", type=Path)
    parser.add_argument("--head-prefix", default=DEFAULT_HEAD_PREFIX)
    parser.add_argument(
        "--expected-branch-groups",
        type=int,
        default=EXPECTED_BRANCH_GROUP_COUNT,
    )
    parser.add_argument(
        "--expected-baseline-groups",
        type=int,
        default=EXPECTED_BASELINE_GROUP_COUNT,
    )
    return parser


def main() -> int:
    """Run validation and return a shell-compatible status code."""
    args = build_parser().parse_args()
    issues = validate_campaign(
        args.campaign_directory,
        head_prefix=args.head_prefix,
        expected_branch_groups=args.expected_branch_groups,
        expected_baseline_groups=args.expected_baseline_groups,
    )
    if issues:
        logger.error("Pretraining model materialization: FAIL (%s violation(s)).", len(issues))
        for issue in issues:
            logger.error("- %s", issue)
        return 1
    logger.info("Pretraining model materialization: PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
