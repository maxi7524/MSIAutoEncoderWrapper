"""Tests for transfer of planned artifacts into wrapper task factories."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from msi_autoencoder_wrapper.runtime.workflows import wrapper as workflow


def test_build_wrapper_passes_resolved_artifacts_to_factory(monkeypatch) -> None:
    """A factory receives plan artifacts nested with its factory parameters."""
    received: dict = {}

    def factory(parameters: dict) -> SimpleNamespace:
        received.update(parameters)
        return SimpleNamespace(active_model=object(), active_dataset=object())

    monkeypatch.setattr(workflow, "resolve_entrypoint", lambda _entrypoint: factory)
    task = {
        "parameters": {
            "factory": "tests.runtime.test_wrapper_workflow:factory",
            "factory_parameters": {"project_path": "/workspace"},
            "resolved": {"context_config": "/plan/context.yaml"},
        }
    }

    workflow._build_wrapper(task)

    assert received == {
        "project_path": "/workspace",
        "resolved": {"context_config": "/plan/context.yaml"},
    }


def test_build_wrapper_rejects_task_without_resolved_artifacts(monkeypatch) -> None:
    """Training cannot silently construct an unresolved runtime pipeline."""
    monkeypatch.setattr(workflow, "resolve_entrypoint", lambda _entrypoint: lambda _: None)
    task = {"parameters": {"factory": "unused", "factory_parameters": {}}}

    with pytest.raises(ValueError, match="parameters.resolved"):
        workflow._build_wrapper(task)


def test_dependent_task_loads_exact_persisted_parent_weights(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A child initializes from the parent artifact and records its exact digest."""
    parent_model = torch.nn.Linear(2, 1)
    child_model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        parent_model.weight.copy_(torch.tensor([[3.0, 4.0]]))
        parent_model.bias.copy_(torch.tensor([5.0]))
        child_model.weight.zero_()
        child_model.bias.zero_()
    parent_path = tmp_path / "models" / "parent_pretrained"
    weights_path = parent_path / "config" / "weights.pt"
    weights_path.parent.mkdir(parents=True)
    torch.save(parent_model.state_dict(), weights_path)
    model_config = {"name": "fixture", "type": "autoencoder"}
    wrapper = SimpleNamespace(
        active_model=child_model,
        models_manager=SimpleNamespace(
            get_model_config=lambda: {"model": model_config}
        ),
    )
    monkeypatch.setattr(
        workflow.ModelLoader,
        "load_artifact",
        lambda _reference: (parent_model, {"model": model_config}, parent_path),
    )
    task = {
        "task_id": "task_000001",
        "workflow": {
            "role": "frozen_head",
            "parent_task_id": "task_000000",
        },
        "runtime": {
            "dependency_results": {
                "task_000000": {"model_path": str(parent_path)}
            }
        },
    }

    initialization = workflow._load_parent_model(wrapper, task)

    assert initialization == {
        "parent_task_id": "task_000000",
        "model_path": str(parent_path.resolve()),
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
    }
    for name, value in parent_model.state_dict().items():
        torch.testing.assert_close(child_model.state_dict()[name], value)
