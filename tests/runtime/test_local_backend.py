"""Tests for persistent local runtime workers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from msi_autoencoder_wrapper.runtime.backends.local import run_local_tasks
from msi_autoencoder_wrapper.runtime.workflows import configured


def test_local_backend_batches_tasks_into_persistent_workers(monkeypatch) -> None:
    """One subprocess receives several tasks instead of reopening data per task."""
    commands: list[list[str]] = []

    def record(command: list[str], *, check: bool) -> None:
        assert check is True
        commands.append(command)

    monkeypatch.setattr(subprocess, "run", record)
    task_paths = [Path(f"task_{index:06d}.yaml") for index in range(5)]

    run_local_tasks(task_paths, max_parallel_runs=2)

    assert len(commands) == 2
    assert all(command[3] == "task-batch" for command in commands)
    assert sorted(path for command in commands for path in command[4:]) == sorted(
        str(path) for path in task_paths
    )


def test_configured_factory_reuses_reader_within_worker(monkeypatch) -> None:
    """Repeated dataset definitions initialize one native reader per process."""
    configured._READER_CACHE.clear()
    initialized: list[object] = []

    class ContextManager:
        def __init__(self, wrapper: object) -> None:
            self.wrapper = wrapper

        def set_reader(self, target, *_args, **_kwargs):
            if isinstance(target, str):
                target = SimpleNamespace(active_context=self.wrapper.active_context)
                initialized.append(target)
            return target

        def load_reader(self, _config, *_args, reader_instance=None, **_kwargs):
            if reader_instance is not None:
                reader_instance.active_context = self.wrapper.active_context
                return reader_instance
            initialized.append(object())
            return SimpleNamespace(active_context=self.wrapper.active_context)

    first = SimpleNamespace(active_context=object())
    first.context_manager = ContextManager(first)
    second = SimpleNamespace(active_context=object())
    second.context_manager = ContextManager(second)
    definition = {"strategy": "M2aiaReader", "parameters": {}}
    first_reader = configured._get_or_create_reader(
        first,
        image_path=Path("dataset.imzML"),
        definition=definition,
    )
    second_reader = configured._get_or_create_reader(
        second,
        image_path=Path("dataset.imzML"),
        definition=definition,
    )

    assert len(initialized) == 1
    assert second_reader is first_reader
    assert second_reader.active_context is second.active_context
    configured._READER_CACHE.clear()
