"""Standard execution tasks built on the wrapper model and training mixins."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .entrypoints import resolve_entrypoint


def _build_wrapper(task: dict[str, Any]) -> Any:
    parameters = task["parameters"]
    factory = parameters.get("factory")
    if not isinstance(factory, str):
        raise ValueError("Standard wrapper tasks require parameters.factory")
    wrapper = resolve_entrypoint(factory)(deepcopy(parameters.get("factory_parameters", {})))
    if getattr(wrapper, "active_model", None) is None:
        raise ValueError("Experiment factory did not configure an active model")
    if getattr(wrapper, "active_dataset", None) is None:
        raise ValueError("Experiment factory did not configure an active dataset")
    return wrapper


def _apply_split_seed(wrapper: Any, seed: int) -> None:
    dataset = wrapper.active_dataset
    split = getattr(dataset, "_split_config", None)
    if isinstance(split, dict):
        split["seed"] = seed
        dataset._partitions = None


def preflight_wrapper_training(task: dict[str, Any]) -> dict[str, Any]:
    """Build the complete pipeline and execute its existing probe forward pass."""
    wrapper = _build_wrapper(task)
    _apply_split_seed(wrapper, int(task["seed"]))
    training = deepcopy(task["parameters"]["training"])
    training["seed"] = int(task["seed"])
    report = wrapper.models_manager.estimate_training_resources(
        training,
        auto_adjust_batch_size=False,
        print_return=False,
    )
    return {"status": "ready", "resources": report.get("available", {})}


def run_wrapper_training(task: dict[str, Any]) -> dict[str, Any]:
    """Build, train and persist one wrapper model task."""
    wrapper = _build_wrapper(task)
    seed = int(task["seed"])
    _apply_split_seed(wrapper, seed)
    training = deepcopy(task["parameters"]["training"])
    training["seed"] = seed
    history = wrapper.models_manager.fit(training)
    model_name = task["parameters"].get("model_name")
    model_path = wrapper.workspace.save_model(model_name=model_name, history=history)
    return {"model_path": str(Path(model_path).resolve()), "epochs": len(history)}
