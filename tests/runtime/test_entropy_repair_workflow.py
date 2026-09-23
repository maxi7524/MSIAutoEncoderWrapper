"""Tests for repaired Entropy dependency batches and model finalization."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from msi_autoencoder_wrapper.runtime.output import task_fingerprint, update_manifest


SCRIPTS = Path(__file__).parents[2] / "assets/scripts/entropy"


def _script(name: str) -> ModuleType:
    """Load one script as an isolated Python module."""
    specification = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _task(plan: Path, index: int, parents: tuple[int, ...] = ()) -> dict:
    """Write a minimal materialized task descriptor."""
    task = {
        "task_id": f"task_{index:06d}",
        "depends_on": [f"task_{parent:06d}" for parent in parents],
        "parameters": {"model_name": f"campaign__task_{index:06d}"},
    }
    path = plan / "tasks" / f"{task['task_id']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(task), encoding="utf-8")
    return task


def _model(workspace: Path, task: dict) -> Path:
    """Create the three required model-bundle files."""
    path = workspace / "models/kidney" / task["parameters"]["model_name"]
    config = path / "config"
    config.mkdir(parents=True)
    for name in ("config.json", "history.json", "weights.pt"):
        (config / name).write_bytes(b"test")
    return path


def _complete(plan: Path, task: dict, model: Path, initialization: dict | None = None) -> None:
    """Write a matching completed task status."""
    result = {"model_path": str(model)}
    if initialization is not None:
        result["initialization"] = initialization
    update_manifest(
        plan / "status" / f"{task['task_id']}.yaml", task["task_id"],
        {"status": "completed", "task_fingerprint": task_fingerprint(task), "result": result},
    )


def test_batch_never_contains_a_pending_parent_and_child(tmp_path: Path) -> None:
    """A completed imported parent unlocks children; a pending parent does not."""
    planner = _script("campaign_task_batches")
    plan = tmp_path / "plan"
    parent = _task(plan, 0)
    child_a = _task(plan, 1, (0,))
    _task(plan, 2, (0,))
    independent = _task(plan, 3)

    assert planner.select_batch(plan, 12) == [0, 3]
    _complete(plan, parent, _model(tmp_path, parent))
    assert planner.select_batch(plan, 12) == [1, 2, 3]
    _complete(plan, child_a, _model(tmp_path, child_a))
    _complete(plan, independent, _model(tmp_path, independent))
    assert planner.select_batch(plan, 12) == [2]
    planner.verify_batch(plan, [0, 1, 3])


def test_batch_rejects_invalid_completed_status(tmp_path: Path) -> None:
    """A stale fingerprint must not silently skip an existing status file."""
    planner = _script("campaign_task_batches")
    plan = tmp_path / "plan"
    task = _task(plan, 0)
    _complete(plan, task, _model(tmp_path, task))
    planner.build_index(plan)
    task["parameters"]["model_name"] = "changed"
    (plan / "tasks/task_000000.yaml").write_text(yaml.safe_dump(task), encoding="utf-8")
    with pytest.raises(ValueError, match="Task descriptor changed"):
        planner.select_batch(plan, 12)


def test_finalization_rewrites_local_result_and_parent_paths(tmp_path: Path) -> None:
    """The node-local directory can be removed after path rebinding."""
    finalizer = _script("finalize_model_paths")
    plan = tmp_path / "plan"
    local = tmp_path / "local"
    source = tmp_path / "source"
    parent = _task(plan, 0)
    child = _task(plan, 1, (0,))
    parent_local = _model(local, parent)
    child_local = _model(local, child)
    parent_source = _model(source, parent)
    child_source = _model(source, child)
    _complete(plan, parent, parent_local)
    _complete(plan, child, child_local, {
        "parent_task_id": parent["task_id"], "model_path": str(parent_local),
    })

    assert finalizer.finalize_paths(plan, local, source) == 2
    parent_record = yaml.safe_load((plan / "status/task_000000.yaml").read_text())["records"][parent["task_id"]]
    child_record = yaml.safe_load((plan / "status/task_000001.yaml").read_text())["records"][child["task_id"]]
    assert parent_record["result"]["model_path"] == str(parent_source)
    assert child_record["result"]["model_path"] == str(child_source)
    assert child_record["result"]["initialization"]["model_path"] == str(parent_source)
    assert finalizer.finalize_paths(plan, local, source) == 0


def test_finalization_validates_every_model_before_status_writes(tmp_path: Path) -> None:
    """A missing copied child model leaves every old result untouched."""
    finalizer = _script("finalize_model_paths")
    plan = tmp_path / "plan"
    local = tmp_path / "local"
    source = tmp_path / "source"
    parent = _task(plan, 0)
    child = _task(plan, 1, (0,))
    parent_local = _model(local, parent)
    child_local = _model(local, child)
    _model(source, parent)
    _complete(plan, parent, parent_local)
    _complete(plan, child, child_local)
    with pytest.raises(ValueError, match="Persistent model is incomplete"):
        finalizer.finalize_paths(plan, local, source)
    record = yaml.safe_load((plan / "status/task_000000.yaml").read_text())["records"][parent["task_id"]]
    assert record["result"]["model_path"] == str(parent_local)


def test_repair_rejects_incomplete_or_invalid_aggregate_plan(tmp_path: Path) -> None:
    """A crash-truncated aggregate plan cannot authorize model import."""
    repair = _script("pretrain_repair")
    path = tmp_path / "resolved-experiment.yaml"
    identifiers = [f"task_{index:06d}" for index in range(310)]
    path.write_text(yaml.safe_dump({
        "runtime_schema_version": 2,
        "tasks": [{"task_id": task_id} for task_id in identifiers[:-1]],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="complete 310-task"):
        repair._plan_task_ids(path)

    path.write_text(yaml.safe_dump({
        "runtime_schema_version": 2,
        "tasks": [{"task_id": task_id} for task_id in identifiers],
    }), encoding="utf-8")
    assert repair._plan_task_ids(path) == set(identifiers)

    path.write_text(yaml.safe_dump({
        "runtime_schema_version": 3,
        "tasks": [{"task_id": task_id} for task_id in identifiers],
    }), encoding="utf-8")
    assert repair._plan_task_ids(path) == set(identifiers)

    with path.open("a", encoding="utf-8") as stream:
        stream.write("broken: [\n")
    with pytest.raises(yaml.YAMLError):
        repair._plan_task_ids(path)


def test_node_local_parent_requires_the_recorded_execution_workspace(tmp_path: Path) -> None:
    """A login node may trust only a matching node-local completed result."""
    planner = _script("campaign_task_batches")
    plan = tmp_path / "plan"
    local = tmp_path / "node-local"
    parent = _task(plan, 0)
    _task(plan, 1, (0,))
    parent_path = local / "models/kidney" / parent["parameters"]["model_name"]
    _complete(plan, parent, parent_path)

    with pytest.raises(ValueError, match="no accessible model bundle"):
        planner.select_batch(plan, 12)
    with pytest.raises(ValueError, match="no accessible model bundle"):
        planner.select_batch(plan, 12, tmp_path / "other-node")
    assert planner.select_batch(plan, 12, local) == [1]


def test_failed_status_remains_pending_for_restart(tmp_path: Path) -> None:
    """A failed task with the same fingerprint can be retried on restart."""
    planner = _script("campaign_task_batches")
    plan = tmp_path / "plan"
    task = _task(plan, 0)
    update_manifest(
        plan / "status/task_000000.yaml", task["task_id"],
        {"status": "failed", "task_fingerprint": task_fingerprint(task)},
    )
    assert planner.select_batch(plan, 12) == [0]


def test_finalization_rejects_nonidentical_copied_weights(tmp_path: Path) -> None:
    """A transferred bundle must match node-local bytes before path mutation."""
    finalizer = _script("finalize_model_paths")
    plan = tmp_path / "plan"
    local = tmp_path / "local"
    source = tmp_path / "source"
    task = _task(plan, 0)
    local_model = _model(local, task)
    source_model = _model(source, task)
    (source_model / "config/weights.pt").write_bytes(b"changed")
    _complete(plan, task, local_model)

    with pytest.raises(ValueError, match="Persistent model file differs"):
        finalizer.finalize_paths(plan, local, source)
    record = yaml.safe_load((plan / "status/task_000000.yaml").read_text())["records"][task["task_id"]]
    assert record["result"]["model_path"] == str(local_model)
