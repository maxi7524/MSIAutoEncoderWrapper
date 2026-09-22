"""Standard runtime workflows built on the wrapper model and training mixins."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

import torch

from ...models.model_loader import ModelLoader
from ...utils.logger import get_custom_logger
from ..reproducibility import set_execution_seed
from .entrypoints import resolve_entrypoint

logger = get_custom_logger(__name__)


def _build_wrapper(task: dict[str, Any]) -> Any:
    """Construct and validate the wrapper required by one training task."""
    # Declarative factory dispatch
    ## Convert the resolved YAML parameters into reader, binner, dataset, and model objects
    parameters = task["parameters"]
    factory = parameters.get("factory")
    if not isinstance(factory, str):
        raise ValueError("Standard wrapper tasks require parameters.factory")
    factory_parameters = deepcopy(parameters.get("factory_parameters", {}))
    resolved = parameters.get("resolved")
    if not isinstance(resolved, dict):
        raise ValueError("Standard wrapper tasks require parameters.resolved")
    # Plan artifacts
    ## The factory receives portable model, context, and split references together
    ## with its original construction parameters.
    factory_parameters["resolved"] = deepcopy(resolved)
    wrapper = resolve_entrypoint(factory)(factory_parameters)
    if getattr(wrapper, "active_model", None) is None:
        raise ValueError("Experiment factory did not configure an active model")
    if getattr(wrapper, "active_dataset", None) is None:
        raise ValueError("Experiment factory did not configure an active dataset")
    return wrapper


def _apply_split_seed(wrapper: Any, seed: int) -> None:
    """Bind the repetition seed to the dataset split and invalidate its cache."""
    # Paired data partition
    ## All grid variants in one repetition receive identical sample assignments
    dataset = wrapper.active_dataset
    split = getattr(dataset, "_split_config", None)
    if isinstance(split, dict):
        split["seed"] = seed
        dataset._partitions = None


def _weights_sha256(model_path: Path) -> str:
    """Hash the exact persisted parent weights loaded by an adaptation task."""
    digest = hashlib.sha256()
    weights_path = model_path / "config" / "weights.pt"
    with weights_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_parent_model(wrapper: Any, task: dict[str, Any]) -> dict[str, str] | None:
    """Load a completed parent artifact into one dependent training task.

    :param wrapper: Fully constructed child wrapper.
    :type wrapper: typing.Any
    :param task: Materialized task with dependency results injected by the runtime.
    :type task: dict[str, typing.Any]
    :return: Auditable parent initialization metadata, or ``None`` for root tasks.
    :rtype: dict[str, str] | None
    :raises ValueError: If dependency metadata is missing or inconsistent.
    """
    workflow = task.get("workflow")
    if not isinstance(workflow, dict):
        return None
    parent_task_id = workflow.get("parent_task_id")
    if parent_task_id is None:
        return None
    dependencies = task.get("runtime", {}).get("dependency_results", {})
    if not isinstance(dependencies, dict):
        raise ValueError("runtime.dependency_results must be a mapping.")
    parent_result = dependencies.get(parent_task_id)
    if not isinstance(parent_result, dict):
        raise ValueError(
            f"Task '{task['task_id']}' has no completed result for parent "
            f"'{parent_task_id}'."
        )
    parent_reference = parent_result.get("model_path")
    if not isinstance(parent_reference, str) or not parent_reference:
        raise ValueError(f"Parent task '{parent_task_id}' has no model_path result.")

    parent_model, _, parent_path = ModelLoader.load_artifact(parent_reference)
    active_model = wrapper.active_model
    active_model.load_state_dict(parent_model.state_dict(), strict=True)
    del parent_model
    if any(not bool(torch.isfinite(value).all()) for value in active_model.state_dict().values()):
        raise ValueError(f"Parent task '{parent_task_id}' contains non-finite weights.")
    initialization = {
        "parent_task_id": str(parent_task_id),
        "model_path": str(parent_path.resolve()),
        "weights_sha256": _weights_sha256(parent_path),
    }
    logger.info(
        "Loaded parent model '%s' from %s (weights_sha256=%s).",
        parent_task_id,
        parent_path,
        initialization["weights_sha256"],
    )
    return initialization


def preflight_wrapper_training(task: dict[str, Any]) -> dict[str, Any]:
    """Build the complete pipeline and execute its existing probe forward pass."""
    # Complete dry construction
    ## Detect reader, binner, dataset, architecture, and resource errors before submission
    wrapper = _build_wrapper(task)
    _apply_split_seed(wrapper, int(task["reproducibility"]["common_seeds"]["split"]))
    training = deepcopy(task["parameters"]["training"])
    training["seed"] = int(task["reproducibility"]["derived_run_seeds"]["training"])
    report = wrapper.models_manager.estimate_training_resources(
        training,
        auto_adjust_batch_size=False,
        print_return=False,
    )
    return {"status": "ready", "resources": report.get("available", {})}


def run_wrapper_training(task: dict[str, Any]) -> dict[str, Any]:
    """Build, train and persist one wrapper model task."""
    # Task construction and paired split
    wrapper = _build_wrapper(task)
    initialization = _load_parent_model(wrapper, task)
    reproducibility = task["reproducibility"]
    _apply_split_seed(wrapper, int(reproducibility["common_seeds"]["split"]))
    # Training randomness
    ## Pass the repetition seed into DataLoader and training-loop configuration
    training = deepcopy(task["parameters"]["training"])
    training["seed"] = int(reproducibility["derived_run_seeds"]["training"])
    continuation = training.pop("continuation")
    training["runtime"] = {**deepcopy(task["runtime"]), **continuation}
    trainable_parameters = sum(
        parameter.numel()
        for parameter in wrapper.active_model.parameters()
        if parameter.requires_grad
    )
    logger.info(
        "Starting %s | trainable parameters=%s | model seed=%s | training seed=%s.",
        training["runtime"]["task_label"],
        trainable_parameters,
        task["reproducibility"]["derived_run_seeds"]["model_initialization"],
        training["seed"],
    )
    for phase in training.get("phases", []):
        phase["dataloader_seed"] = int(
            reproducibility["common_seeds"]["dataloader"]
        )
    set_execution_seed(training["seed"])
    # Training and persistence
    history = wrapper.models_manager.fit(training)
    model_name = (
        task["parameters"].get("model_name")
        or task.get("runtime", {}).get("model_name")
        or task["task_id"]
    )
    model_path = wrapper.workspace.save_model(model_name=model_name, history=history)
    result: dict[str, Any] = {
        "model_path": str(Path(model_path).resolve()),
        "epochs": len(history),
    }
    if initialization is not None:
        result["initialization"] = initialization
    return result


def test_wrapper_training(task: dict[str, Any]) -> dict[str, Any]:
    """Run a bounded train-step probe without persisting a model or history.

    :param task: Materialized task with resolved component artifacts.
    :type task: dict[str, Any]
    :return: Number of completed probe epochs.
    :rtype: dict[str, Any]
    """
    wrapper = _build_wrapper(task)
    reproducibility = task["reproducibility"]
    _apply_split_seed(wrapper, int(reproducibility["common_seeds"]["split"]))
    training = deepcopy(task["parameters"]["training"])
    training["seed"] = int(reproducibility["derived_run_seeds"]["training"])
    training["test_mode"] = True
    training["runtime"] = deepcopy(task["runtime"])
    training["runtime"]["resume"] = False
    training["checkpoint"] = {"enabled": False, "restore_best": False}
    for phase in training.get("phases", []):
        phase["epochs"] = 1
        phase["max_batches"] = int(phase.get("test_max_batches", 2))
        phase["dataloader_seed"] = int(
            reproducibility["common_seeds"]["dataloader"]
        )
    set_execution_seed(training["seed"])
    history = wrapper.models_manager.fit(training)
    return {"epochs": len(history), "test_mode": True}
