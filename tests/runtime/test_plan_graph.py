"""Contracts for compact materialized campaign graphs and shared YAML leaves."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from msi_autoencoder_wrapper.runtime.planning.graph import verify_plan_graph
from msi_autoencoder_wrapper.runtime.planning.plan import (
    ExperimentPlan,
    PlannedTask,
    materialize_plan,
)
from msi_autoencoder_wrapper.runtime.workflows.configured import (
    _source_indices_from_reference,
    _write_source_indices,
)


def _plan(root: Path, parents: tuple[tuple[str, ...], ...]) -> ExperimentPlan:
    """Construct two small tasks that share one deterministic source population."""
    reference = _write_source_indices(root / "resolved/source_populations", [2, 7, 11, 19])
    tasks = tuple(
        PlannedTask(
            task_id=f"task_{index:06d}",
            grid_id="grid_0000",
            repetition=0,
            reproducibility={"common_seeds": {"split": 42}},
            grid_parameters={},
            entrypoint="tests.runtime.test_cli:task_entrypoint",
            parameters={
                "factory_parameters": {
                    "dataset": {
                        "parameters": {
                            "subset": {"method": "source_indices", "indices_ref": reference}
                        }
                    }
                }
            },
            workflow={"group_id": "grid_0000__rep_00", "role": "pretrained"},
            depends_on=parents[index],
        )
        for index in range(2)
    )
    return ExperimentPlan("graph-test", "test.yaml", "abc", tasks, {}, ())


def test_compact_graph_reuses_one_verified_source_population(tmp_path: Path) -> None:
    """Two tasks point to one leaf, while the aggregate contains no copied indices."""
    plan = _plan(tmp_path, ((), ("task_000000",)))
    model_path = tmp_path / "resolved/models/model.yaml"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("model: {}\n", encoding="utf-8")
    for task in plan.tasks:
        task.parameters["resolved"] = {"model_config": str(model_path.resolve())}
    materialize_plan(plan, tmp_path)

    assert verify_plan_graph(tmp_path, expected_count=2) == 2
    assert len(list((tmp_path / "resolved/source_populations").glob("*.yaml"))) == 1
    aggregate = yaml.safe_load((tmp_path / "resolved-experiment.yaml").read_text())
    assert aggregate["runtime_schema_version"] == 3
    assert all("parameters" not in node for node in aggregate["tasks"])
    assert "indices:" not in (tmp_path / "tasks/task_000000.yaml").read_text()
    subset = plan.tasks[0].parameters["factory_parameters"]["dataset"]["parameters"]["subset"]
    assert _source_indices_from_reference(subset)["indices"] == [2, 7, 11, 19]


def test_graph_rejects_changed_shared_yaml(tmp_path: Path) -> None:
    """A shared leaf cannot silently change after task fingerprints are recorded."""
    plan = _plan(tmp_path, ((), ("task_000000",)))
    materialize_plan(plan, tmp_path)
    reference = plan.tasks[0].parameters["factory_parameters"]["dataset"]["parameters"]["subset"]["indices_ref"]
    Path(reference["path"]).write_text("indices: [2, 8, 11, 19]\n")

    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_plan_graph(tmp_path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        _source_indices_from_reference({"method": "source_indices", "indices_ref": reference})


@pytest.mark.parametrize(
    "parents, message",
    [
        (((), ("task_999999",)), "unknown parent"),
        ((("task_000001",), ("task_000000",)), "dependency cycle"),
    ],
)
def test_graph_rejects_broken_lineage(
    tmp_path: Path, parents: tuple[tuple[str, ...], ...], message: str
) -> None:
    """Unknown and cyclic task edges are rejected before task-count publication."""
    materialize_plan(_plan(tmp_path, parents), tmp_path)
    with pytest.raises(ValueError, match=message):
        verify_plan_graph(tmp_path)


def test_graph_rejects_missing_or_mismatched_descriptor(tmp_path: Path) -> None:
    """The index cannot authorize an absent or renamed task descriptor."""
    materialize_plan(_plan(tmp_path, ((), ("task_000000",))), tmp_path)
    path = tmp_path / "tasks/task_000001.yaml"
    path.rename(tmp_path / "tasks/other.yaml")
    with pytest.raises(FileNotFoundError):
        verify_plan_graph(tmp_path)


def test_graph_rejects_missing_shared_model_yaml(tmp_path: Path) -> None:
    """Every existing resolved-model edge must identify a readable YAML node."""
    plan = _plan(tmp_path, ((), ("task_000000",)))
    plan.tasks[0].parameters["resolved"] = {
        "model_config": str((tmp_path / "resolved/models/missing.yaml").resolve())
    }
    materialize_plan(plan, tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_plan_graph(tmp_path)


def test_shared_population_is_content_addressed(tmp_path: Path) -> None:
    """Equal lists map to one file; a changed list maps to a distinct file."""
    root = tmp_path / "resolved/source_populations"
    first = _write_source_indices(root, [2, 7, 11])
    assert _write_source_indices(root, [2, 7, 11]) == first
    assert _write_source_indices(root, [2, 7, 12]) != first
    assert len(list(root.glob("*.yaml"))) == 2
