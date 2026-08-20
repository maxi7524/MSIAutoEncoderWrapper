"""Resolve and run the importable function assigned to a planned task."""

from __future__ import annotations

import importlib
from typing import Any, Callable

from ..reproducibility import set_execution_seed


def resolve_entrypoint(value: str) -> Callable[[dict[str, Any]], Any]:
    """Resolve a ``module:function`` reference without evaluating user code."""
    module_name, function_name = value.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"Task entrypoint is not callable: {value}")
    return function


def execute_task(task: dict[str, Any]) -> Any:
    """Apply the task seed and invoke its public experiment entrypoint."""
    # Reproducibility boundary
    ## Seed before the factory constructs the model, controlling weight initialization
    reproducibility = task["reproducibility"]
    set_execution_seed(
        int(reproducibility["derived_run_seeds"]["model_initialization"])
    )

    # Workflow dispatch
    ## Resolve the validated module:function reference assigned to this task
    return resolve_entrypoint(task["entrypoint"])(task)
