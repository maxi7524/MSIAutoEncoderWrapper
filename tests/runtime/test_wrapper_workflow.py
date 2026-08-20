"""Tests for transfer of planned artifacts into wrapper task factories."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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
