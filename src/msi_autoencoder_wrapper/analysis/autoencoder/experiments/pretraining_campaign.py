"""Readers and data contracts of the synthetic-pretraining campaign analyses.

The saved models of this campaign contain only their ``model`` configuration, not
the data configuration or split assignments. The data contract is therefore
reconstructed from the campaign YAML through the *same* plan resolver the runtime used
before training (:func:`~msi_autoencoder_wrapper.runtime.workflows.configured.resolve_single_image_campaign`),
and verified against the persisted artifacts: the resolved model configuration must be
identical to every saved one, and the resolved split may additionally be compared with a
recovered split manifest.

Three pixel populations are defined on top of that contract:

``train`` / ``test``
    The saved partitions of the 20 % proportional subset of the training cohort.
``test_extended``
    Additional pixels of training-cohort images that lie *outside* the 20 % subset,
    drawn only to raise the number of test positives of under-represented classes.
``heldout_image``
    Every non-empty pixel of the datasets excluded from the cohort before subsetting.

All populations are decoded by source-spectrum identifier through the dataset's own
reader, binner and normalization, so they share one preprocessing path with training.
"""

from __future__ import annotations

import json
import re
import sqlite3
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from ....utils.logger import get_custom_logger
from .predictive_campaign import fingerprint

logger = get_custom_logger(__name__)

#: Keys of the analysis settings holding repository-relative paths.
PATH_SETTINGS = ("workspace", "model_store", "experiment_config", "cache_directory", "merged_store",
                 "reference_split")

#: Population names in their canonical reporting order.
POPULATIONS = ("train", "test", "test_extended", "heldout_image")


# --------------------------------------------------
# Section: settings and model inventory
# --------------------------------------------------

def load_settings(path: Path | str) -> dict:
    """Load analysis settings and resolve repository-relative paths.

    :param path: Analysis settings YAML inside the repository.
    :type path: pathlib.Path | str
    :return: Settings with absolute paths, ``repository_root`` and ``settings_path``.
    :rtype: dict
    :raises ValueError: If the file is outside a repository or required keys are missing.
    """
    path = Path(path).resolve()
    root = next((parent for parent in path.parents if (parent / "pyproject.toml").is_file()), None)
    if root is None:
        raise ValueError("Analysis settings must live inside the repository.")
    settings = yaml.safe_load(path.read_text())
    required = ("workspace", "model_store", "experiment_config", "campaign_id", "cache_directory",
                "merged_store", "roles", "axes", "target_field", "evidence", "windows", "populations")
    missing = [key for key in required if key not in settings]
    if missing:
        raise ValueError(f"Analysis settings miss required key(s): {missing}.")
    for key in PATH_SETTINGS:
        if settings.get(key):
            settings[key] = str((root / settings[key]).resolve())
    settings["repository_root"] = str(root)
    settings["settings_path"] = str(path)
    settings.setdefault("device", "cuda")
    settings.setdefault("batch_size", 1024)
    settings.setdefault("baseline_role", "real_only")
    expand_cell_models(settings)
    return settings


# --------------------------------------------------
# Section: variants, stages and model cells
# --------------------------------------------------

#: Variant name of models without synthetic pretraining (the real-data baseline).
BASELINE_VARIANT = "baseline"


def cell_alias(variant: str, stage: str, axis_directory: str) -> str:
    """Return the model alias of one (variant, stage, axis) cell.

    :param variant: Variant key of the settings ``variants`` block.
    :type variant: str
    :param stage: Stage (workflow role) key of the settings ``stages`` block.
    :type stage: str
    :param axis_directory: Folder name of the axis.
    :type axis_directory: str
    :return: Stable alias ``<variant>__<stage>__<axis directory>``.
    :rtype: str
    """
    return f"{variant}__{stage}__{axis_directory}"


def expand_cell_models(settings: dict) -> None:
    """Generate one model alias per (axis, variant, stage) cell in ``settings['models']``.

    The ``variants`` block identifies every pretraining variant by the ordered names of
    its synthetic phases; the ``stages`` block lists the pretraining-lineage roles.
    Explicit ``models`` entries (the baselines) keep their position; generated cells
    follow in axis, variant and stage order and take the variant colour and the stage
    line style, so one colour denotes one variant in every figure. Display labels name
    the axis, because the visualization contract requires unique labels.

    :param settings: Settings mapping, modified in place.
    :type settings: dict
    :raises ValueError: If a generated alias collides with an explicit one.
    """
    variants, stages = settings.get("variants") or {}, settings.get("stages") or {}
    if not variants or not stages:
        return
    models = settings.setdefault("models", {})
    order = len(models)
    for axis, axis_definition in settings["axes"].items():
        for variant, definition in variants.items():
            for stage, stage_definition in stages.items():
                alias = cell_alias(variant, stage, axis_definition["directory"])
                if alias in models:
                    raise ValueError(f"Generated model alias '{alias}' is also defined explicitly.")
                models[alias] = {
                    "display_label": f"{definition['label']}, {stage_definition['label']}, "
                                     f"{axis_definition.get('label', axis)}",
                    "select": {"axis": axis, "role": stage, "variant": variant},
                    "tags": {"axis": axis, "role": stage, "variant": variant, **definition.get("factors", {})},
                    "visualization": {"order": order, "color": definition["color"],
                                      "line_style": stage_definition.get("line_style", "solid"),
                                      "marker": stage_definition.get("marker", "o")},
                }
                order += 1


def lineage_phases(task: Any, tasks: Optional[dict[str, Any]] = None) -> list[dict]:
    """Every training phase that produced a task's model, in training order.

    A fine-tuning task declares only its own phase and continues from its parent task
    (``workflow.parent_task_id``); the phases of the parent chain come first.

    :param task: Planned task.
    :type task: msi_autoencoder_wrapper.runtime.planning.plan.PlannedTask
    :param tasks: Every planned task keyed by identifier (needed for tasks with a parent).
    :type tasks: dict | None
    :return: Phase declarations.
    :rtype: list[dict]
    :raises ValueError: If a parent task is not available.
    """
    phases = list(task.parameters["training"]["phases"])
    parent = (task.workflow or {}).get("parent_task_id")
    while parent is not None:
        if tasks is None or parent not in tasks:
            raise ValueError(f"Parent task {parent} of {task.task_id} is not available.")
        phases = list(tasks[parent].parameters["training"]["phases"]) + phases
        parent = (tasks[parent].workflow or {}).get("parent_task_id")
    return phases


def synthetic_phase_names(task: Any, tasks: Optional[dict[str, Any]] = None) -> tuple[str, ...]:
    """Names of the synthetic-pretraining phases of one task's lineage, in training order."""
    return tuple(phase["phase_name"] for phase in lineage_phases(task, tasks) if "pretraining" in phase)


def task_variant(task: Any, variants: dict, tasks: Optional[dict[str, Any]] = None) -> str:
    """Identify the pretraining variant of one task from the synthetic phases of its lineage.

    :param task: Planned task.
    :type task: msi_autoencoder_wrapper.runtime.planning.plan.PlannedTask
    :param variants: Settings ``variants`` block (``phases`` per variant); may be empty.
    :type variants: dict
    :param tasks: Every planned task keyed by identifier (parents of fine-tuning tasks).
    :type tasks: dict | None
    :return: :data:`BASELINE_VARIANT` without synthetic phases, the matching variant
        key, or the ``+``-joined phase names when no configured variant matches (such
        tasks get no model alias and are not analysed).
    :rtype: str
    :raises ValueError: If more than one configured variant matches the task.
    """
    phases = synthetic_phase_names(task, tasks)
    if not phases:
        return BASELINE_VARIANT
    matches = [name for name, definition in variants.items() if tuple(definition["phases"]) == phases]
    if len(matches) > 1:
        raise ValueError(f"Task {task.task_id} with synthetic phases {phases} matches variants {matches}.")
    return matches[0] if matches else "+".join(phases)


def all_campaign_tasks(settings: dict) -> dict[str, Any]:
    """Every planned task of the campaign keyed by identifier, in plan order.

    :param settings: Resolved analysis settings.
    :type settings: dict
    :return: Planned tasks.
    :rtype: dict[str, msi_autoencoder_wrapper.runtime.planning.plan.PlannedTask]
    """
    from ....runtime import build_plan, load_experiment_config

    plan = build_plan(load_experiment_config(settings["experiment_config"]))
    return {task.task_id: task for task in plan.tasks}


def campaign_tasks(settings: dict) -> list[Any]:
    """Return the planned tasks of the configured workflow roles.

    :param settings: Resolved analysis settings.
    :type settings: dict
    :return: Planned tasks in plan order.
    :rtype: list[msi_autoencoder_wrapper.runtime.planning.plan.PlannedTask]
    :raises ValueError: If no task has a configured role.
    """
    roles = set(settings["roles"])
    tasks = [task for task in all_campaign_tasks(settings).values() if (task.workflow or {}).get("role") in roles]
    if not tasks:
        raise ValueError(f"The campaign has no task with role(s) {sorted(roles)}.")
    return tasks


def reference_tasks(settings: dict, tasks: list[Any]) -> list[Any]:
    """Tasks whose data contract is resolved: the baseline-role tasks.

    Every task of the campaign shares one paired split and one data configuration per
    axis (verified by :func:`verify_campaign_plan`), so resolving the baseline tasks is
    sufficient for every role; resolving 310 tasks would only repeat the same split.

    :param settings: Resolved analysis settings.
    :type settings: dict
    :param tasks: Planned tasks of the configured roles.
    :type tasks: list
    :return: Baseline-role tasks in plan order.
    :rtype: list
    :raises ValueError: If no baseline task is planned.
    """
    selected = [task for task in tasks if task.workflow["role"] == settings["baseline_role"]]
    if not selected:
        raise ValueError(f"The campaign has no task with the baseline role '{settings['baseline_role']}'.")
    return selected


def _frozen_paths(phase: dict) -> tuple[str, ...]:
    """Module paths frozen during one phase."""
    return tuple(str(value) for value in phase.get("freeze") or ())


def _head_objective(task: Any, tasks: Optional[dict[str, Any]] = None) -> dict:
    """Identify the evaluated head, its trained loss family and the Masserstein options.

    The objective (and Masserstein options) are those of the last phase, which is the
    real-data objective for every role. The ``family`` is the loss that last *trained*
    the head: a phase with zero epochs or a frozen head leaves the head as the previous
    phase produced it, so pretrained and frozen-head models carry the synthetic BCE
    head. Both families yield binary ``(N, C)`` logits read identically by the ranking
    evaluation.
    """
    phases = lineage_phases(task, tasks)
    objective = phases[-1]["criterions"]
    active = [(head, name, spec) for head, losses in objective.get("heads", {}).items()
              for name, spec in losses.items() if float(spec.get("weight", 1.0)) != 0]
    if len(active) != 1:
        raise ValueError(f"Task {task.task_id} must activate exactly one head loss.")
    head, _, spec = active[0]
    masserstein = [value for value in objective.get("reconstruction", {}).values()
                   if value.get("target") == "MassersteinLoss"]
    if len(masserstein) != 1:
        raise ValueError(f"Task {task.task_id} must use exactly one Masserstein reconstruction loss.")
    family = spec["target"]
    for phase in reversed(phases):
        trained = int(phase.get("epochs", 0)) > 0 and f"heads.{head}" not in _frozen_paths(phase)
        losses = [value for value in phase.get("criterions", {}).get("heads", {}).get(head, {}).values()
                  if float(value.get("weight", 1.0)) != 0]
        if trained and losses:
            family = losses[0]["target"]
            break
    return {"head": head, "family": family, "objective_json": json.dumps(objective, sort_keys=True),
            "masserstein_params_json": json.dumps(masserstein[0].get("params", {}), sort_keys=True)}


def inventory(settings: dict) -> pd.DataFrame:
    """List the campaign models of the configured roles and their artifact state.

    :param settings: Resolved analysis settings.
    :type settings: dict
    :return: One row per planned task with the columns required by the common model
        catalog (``model_id``, ``source``, ``grid_id``, ``task_id``, ``repetition``,
        ``label``, ``condition``, ``ready``) plus ``axis``, ``role``, ``stage``,
        ``variant``, ``lineage`` (one synthetic-pretraining run shared by its stages),
        ``artifact``, head and objective identity. The optional ``repetitions`` setting
        restricts the inventory (smoke runs).
    :rtype: pandas.DataFrame
    """
    from ....runtime.naming import run_identifier

    variants = settings.get("variants") or {}
    repetitions = settings.get("repetitions")
    everything = all_campaign_tasks(settings)
    roles = set(settings["roles"])
    rows = []
    for task in (task for task in everything.values() if (task.workflow or {}).get("role") in roles):
        if repetitions is not None and int(task.repetition) not in set(repetitions):
            continue
        name = run_identifier(settings["campaign_id"], asdict(task))
        artifact = Path(settings["model_store"]) / name
        axis = task.grid_parameters["axes"]["name"]
        role = task.workflow["role"]
        files = {key: (artifact / "config" / f"{key}").is_file() for key in ("config.json", "weights.pt", "history.json")}
        rows.append({
            "model_id": name, "source": settings["campaign_id"], "task_id": task.task_id, "grid_id": task.grid_id,
            "repetition": int(task.repetition), "axis": axis, "role": role, "stage": role,
            "variant": task_variant(task, variants, everything), "lineage": task.workflow.get("group_id", name),
            "label": f"{role} ({axis})", "condition": f"{axis}/{role}", "artifact": str(artifact),
            "initialization_seed": task.reproducibility["derived_run_seeds"]["model_initialization"],
            "training_seed": task.reproducibility["derived_run_seeds"]["training"],
            **_head_objective(task, everything), "ready": all(files.values()),
            "history_available": files["history.json"],
        })
    frame = pd.DataFrame(rows)
    unconfigured = sorted(set(frame.variant) - set(variants) - {BASELINE_VARIANT})
    if variants and unconfigured:
        logger.warning("Synthetic phase sequences without a configured variant (not analysed): %s", unconfigured)
    missing = frame[~frame.ready]
    if not missing.empty:
        logger.warning("Campaign models without complete artifacts: %s", missing.model_id.tolist())
    logger.info("Inventoried %s campaign models (%s ready).", len(frame), int(frame.ready.sum()))
    return frame


def read_history(artifact: Path | str) -> list[dict]:
    """Read the epoch and final-evaluation records saved with one model.

    :param artifact: Model directory containing ``config/history.json``.
    :type artifact: pathlib.Path | str
    :return: History records in saved order.
    :rtype: list[dict]
    """
    return json.loads((Path(artifact) / "config" / "history.json").read_text())


# --------------------------------------------------
# Section: plan resolution and verification
# --------------------------------------------------

def task_parameter_fingerprint(task: Any) -> str:
    """Semantic fingerprint of one task's unresolved parameters (comments never matter).

    :param task: Planned task.
    :type task: msi_autoencoder_wrapper.runtime.planning.plan.PlannedTask
    :return: SHA-256 digest of the JSON-normalized ``parameters`` mapping.
    :rtype: str
    """
    return fingerprint(json.loads(json.dumps(task.parameters, default=str)))


def resolve_campaign_plan(tasks: list[Any], directory: Path) -> dict:
    """Resolve the tasks' data and model artifacts into ``directory``.

    The resolver builds the reference-axis planning pipeline, freezes the paired split,
    derives the train-only molecule mapping and writes the model/context/split
    artifacts, exactly as the runtime did before training. Cache identity and reuse are
    decided by :mod:`.pretraining_campaign_cache`, not here.

    :param tasks: Planned tasks to resolve.
    :type tasks: list
    :param directory: Empty level directory.
    :type directory: pathlib.Path
    :return: Resolved parameters keyed by task identifier.
    :rtype: dict
    """
    from ....runtime.workflows.configured import resolve_single_image_campaign

    directory.mkdir(parents=True, exist_ok=True)
    logger.info("Resolving %s campaign tasks into %s.", len(tasks), directory)
    resolved = resolve_single_image_campaign([asdict(task) for task in tasks], directory)
    parameters = {task.task_id: value for task, value in zip(tasks, resolved)}
    temporary = directory / "resolved_parameters.tmp.json"
    temporary.write_text(json.dumps(parameters, default=str))
    temporary.replace(directory / "resolved_parameters.json")
    return parameters


def comparable_model_config(config: dict) -> dict:
    """Model configuration without serialization-only fields.

    REMARK: artifacts saved by the fine-tuning phases additionally record the import path
    of every component (``module``) and ``preset: null``; neither changes the
    architecture that ``ModelLoader`` builds, so both are dropped before comparing.

    :param config: ``model`` section of a saved or resolved configuration.
    :type config: dict
    :return: Copy without those fields.
    :rtype: dict
    """
    value = deepcopy(config)
    if value.get("preset", "absent") is None:
        value.pop("preset")

    def strip(components: dict) -> None:
        ## A component declares its ``type``; any other mapping is a collection (heads)
        for component in components.values():
            if not isinstance(component, dict):
                continue
            if "type" in component:
                component.pop("module", None)
            else:
                strip(component)

    strip(value.get("components", {}))
    return value


def _head_tensors(artifact: Path | str, head: str) -> dict[str, torch.Tensor]:
    """Parameters of one head read from a saved state dict (no model construction)."""
    state = torch.load(Path(artifact) / "config" / "weights.pt", map_location="cpu", weights_only=True)
    prefix = f"heads.{head}."
    return {key: value for key, value in state.items() if key.startswith(prefix)}


def verify_lineages(settings: dict, models: pd.DataFrame) -> list[dict]:
    """Verify seed pairing and the head semantics of the pretraining stages.

    * every repetition uses one initialization seed across all grid groups, so a
      variant and the baseline of the same repetition start from the same seed and
      their difference is a paired comparison;
    * a ``frozen_head`` model carries the head of its ``pretrained`` model bit for bit;
    * an ``unfrozen_head`` model has a changed head.

    :param settings: Resolved analysis settings.
    :type settings: dict
    :param models: Inventory rows of the selected models.
    :type models: pandas.DataFrame
    :return: Check rows (``check``, ``subject``, ``passed``, ``detail``).
    :rtype: list[dict]
    """
    rows = []
    for repetition, frame in models.groupby("repetition"):
        seeds = frame.initialization_seed.nunique()
        rows.append({"check": "paired_initialization_seed", "subject": f"repetition {repetition}",
                     "passed": seeds == 1, "detail": f"{seeds} distinct initialization seed(s)"})
    stages = models[models.role != settings["baseline_role"]]
    for lineage, frame in stages.groupby("lineage"):
        by_role = {row.role: row for row in frame.itertuples()}
        if "pretrained" not in by_role:
            continue
        pretrained = _head_tensors(by_role["pretrained"].artifact, by_role["pretrained"].head)
        for role, expected_equal in (("frozen_head", True), ("unfrozen_head", False)):
            if role not in by_role:
                continue
            other = _head_tensors(by_role[role].artifact, by_role[role].head)
            equal = pretrained.keys() == other.keys() and all(torch.equal(pretrained[key], other[key])
                                                                for key in pretrained)
            rows.append({"check": f"{role}_head_{'identical' if expected_equal else 'changed'}",
                         "subject": lineage, "passed": equal == expected_equal,
                         "detail": f"{len(pretrained)} head tensors compared"})
    return rows


def verify_campaign_plan(settings: dict, models: pd.DataFrame, parameters: dict,
                         tasks: dict[str, Any]) -> pd.DataFrame:
    """Verify the resolved data contract against every persisted model artifact.

    Only the baseline tasks are resolved (:func:`reference_tasks`). Every other model is
    verified against the resolved baseline task of its axis: identical unresolved data
    and model parameters (so identical split, class mapping and architecture), identical
    saved model configuration and matching head width.

    :param settings: Resolved analysis settings.
    :type settings: dict
    :param models: Inventory rows of the selected models.
    :type models: pandas.DataFrame
    :param parameters: Resolved parameters of the reference tasks, keyed by task identifier.
    :type parameters: dict
    :param tasks: Planned tasks keyed by task identifier.
    :type tasks: dict
    :return: One row per check with ``check``, ``subject``, ``passed`` and ``detail``.
    :rtype: pandas.DataFrame
    :raises ValueError: If any check fails.
    """
    rows = []
    splits: dict[str, list[str]] = {}
    references: dict[str, str] = {}
    for task_id in parameters:
        axis = tasks[task_id].grid_parameters["axes"]["name"]
        references.setdefault(axis, task_id)
        splits.setdefault(parameters[task_id]["resolved"]["split_manifest"], []).append(task_id)
    model_configs: dict[str, dict] = {}
    for row in models.itertuples():
        reference_id = references[row.axis]
        resolved = parameters[reference_id]["resolved"]
        if reference_id not in model_configs:
            model_configs[reference_id] = yaml.load(Path(resolved["model_config"]).read_text(),
                                                    Loader=yaml.CSafeLoader)["model"]
        model_config = model_configs[reference_id]
        saved = json.loads((Path(row.artifact) / "config" / "config.json").read_text())["model"]
        rows.append({"check": "model_configuration", "subject": row.model_id,
                     "passed": comparable_model_config(model_config) == comparable_model_config(saved),
                     "detail": f"resolved={fingerprint(model_config)[:12]} saved={fingerprint(saved)[:12]}"})
        mapping = parameters[reference_id]["factory_parameters"]["dataset"]["parameters"]["target_specs"]
        classes = len(mapping[settings["target_field"]].get("class_mapping", {}))
        head = saved["components"]["heads"][row.head]["parameters"]["output_dim"]
        rows.append({"check": "head_columns", "subject": row.model_id, "passed": classes == head,
                     "detail": f"class_mapping={classes} head_output_dim={head}"})
        if row.task_id != reference_id:
            ## Same unresolved data/model parameters as the resolved reference task
            own = fingerprint(json.loads(json.dumps(tasks[row.task_id].parameters["factory_parameters"], default=str)))
            expected = fingerprint(json.loads(json.dumps(tasks[reference_id].parameters["factory_parameters"],
                                                         default=str)))
            rows.append({"check": "data_parameters_match_reference", "subject": row.model_id,
                         "passed": own == expected, "detail": f"reference={reference_id}"})
    ## One paired split must be shared by every model of the campaign
    rows.append({"check": "single_paired_split", "subject": "campaign", "passed": len(splits) == 1,
                 "detail": f"{len(splits)} distinct split manifest(s)"})
    rows.extend(verify_lineages(settings, models))
    reference = settings.get("reference_split")
    if reference and not Path(reference).is_file():
        ## REMARK: the recovered split lives outside version control; its absence
        ## removes one independent check but does not invalidate the resolved contract.
        logger.warning("Reference split %s is unavailable; skipping that comparison.", reference)
        rows.append({"check": "reference_split_available", "subject": Path(reference).name, "passed": True,
                     "detail": "not available on this machine; comparison skipped"})
        reference = None
    if reference:
        expected_split = yaml.load(Path(reference).read_text(), Loader=yaml.CSafeLoader)
        for path in splits:
            actual = yaml.load(Path(path).read_text(), Loader=yaml.CSafeLoader)
            for field in ("seed", "dataset_fingerprint", "assignments"):
                rows.append({"check": f"reference_split_{field}", "subject": Path(path).name,
                             "passed": actual.get(field) == expected_split.get(field),
                             "detail": f"reference={Path(reference).name}"})
    frame = pd.DataFrame(rows)
    failed = frame[~frame.passed]
    if not failed.empty:
        raise ValueError(f"Campaign data contract verification failed:\n{failed.to_string(index=False)}")
    logger.info("Campaign data contract verified: %s checks passed.", len(frame))
    return frame


def split_assignments(parameters: dict) -> dict[str, np.ndarray]:
    """Return the saved split assignments as sorted source identifiers.

    :param parameters: Resolved parameters of one task.
    :type parameters: dict
    :return: ``train``, ``validation`` and ``test`` identifiers.
    :rtype: dict[str, numpy.ndarray]
    """
    manifest = yaml.load(Path(parameters["resolved"]["split_manifest"]).read_text(), Loader=yaml.CSafeLoader)
    return {name: np.sort(np.asarray(values, dtype=np.int64)) for name, values in manifest["assignments"].items()}


# --------------------------------------------------
# Section: datasets and decoding
# --------------------------------------------------

def build_axis_wrapper(parameters: dict) -> Any:
    """Construct the training wrapper of one task (reader, binner, dataset, model).

    :param parameters: Resolved parameters of one task.
    :type parameters: dict
    :return: Wrapper whose active dataset is the training dataset of the task.
    :rtype: msi_autoencoder_wrapper.MSIAutoEncoderWrapper
    """
    from ....runtime.workflows.configured import build_single_image_autoencoder

    factory = deepcopy(parameters["factory_parameters"])
    factory["resolved"] = deepcopy(parameters["resolved"])
    return build_single_image_autoencoder(factory)


def build_population_dataset(wrapper: Any, parameters: dict, spectrum_ranges: list[tuple[int, int]]) -> Any:
    """Build a dataset over another source population with the frozen class mapping.

    The dataset shares the wrapper's reader, binner and annotation reader, and its
    molecule columns are the train-only mapping resolved for the task. Only the source
    population differs, which makes the annotation index cover those spectra.

    :param wrapper: Wrapper returned by :func:`build_axis_wrapper`.
    :type wrapper: msi_autoencoder_wrapper.MSIAutoEncoderWrapper
    :param parameters: Resolved parameters of the same task.
    :type parameters: dict
    :param spectrum_ranges: Half-open merged-spectrum intervals.
    :type spectrum_ranges: list[tuple[int, int]]
    :return: Annotation-aware dataset over the requested population.
    :rtype: msi_autoencoder_wrapper.models.datasets.strategies.pixel_dataset.PixelDataset
    """
    from ....models.datasets.dataset_manager import DatasetManager

    definition = parameters["factory_parameters"]["dataset"]
    dataset_parameters = deepcopy(definition["parameters"])
    dataset_parameters.pop("subset", None)
    dataset_parameters["source_population"] = {"spectrum_ranges": [list(value) for value in spectrum_ranges]}
    return DatasetManager.load_config({"type": definition["strategy"], "parameters": dataset_parameters},
                                      active_context=wrapper.active_context)


def decode_source_spectra(dataset: Any, source_ids: np.ndarray, *, target_field: str,
                          batch_size: int) -> dict[str, np.ndarray]:
    """Decode spectra and targets by source-spectrum identifier.

    Uses the dataset's native batched reader, the binner's batch transform and the
    dataset's batch normalization, the same operations as
    :func:`~.sweep_evaluation.materialize_split`.

    REMARK: ``_get_raw_source_batch`` is addressed directly because ``test_extended`` and
    held-out pixels are, by design, not members of the training dataset's public index
    space. For members, this path is identical to ``get_raw_batch`` (verified on data).

    :param dataset: Annotation-aware dataset whose annotation index covers the IDs.
    :type dataset: typing.Any
    :param source_ids: Source-spectrum identifiers, shape ``(N,)``.
    :type source_ids: numpy.ndarray
    :param target_field: Target name, e.g. ``molecule``.
    :type target_field: str
    :param batch_size: Spectra decoded per reader call.
    :type batch_size: int
    :return: ``spectra`` ``(N, M)`` float32, ``targets`` ``(N, C)`` uint8, ``mask``
        ``(N, C)`` bool and the pre-normalization total ion current ``tic`` ``(N,)``, in
        the order of ``source_ids``.
    :rtype: dict[str, numpy.ndarray]
    :raises ValueError: If the reader reorders spectra or produces non-finite values.
    """
    ids = np.asarray(source_ids, dtype=np.int64)
    binner = dataset.active_context.binner
    width = int(binner.GetXAxisDepth())
    classes = len(dataset.get_class_mappings()[target_field])
    spectra = np.empty((ids.size, width), dtype=np.float32)  # (N, M)
    targets = np.empty((ids.size, classes), dtype=np.uint8)  # (N, C)
    mask = np.empty((ids.size, classes), dtype=bool)  # (N, C)
    tic = np.empty(ids.size, dtype=np.float32)  # (N,)
    for start in range(0, ids.size, batch_size):
        chunk = ids[start:start + batch_size]
        raw = dataset._get_raw_source_batch(chunk.tolist())
        if not np.array_equal(np.asarray(raw.sample_ids), chunk):
            raise ValueError("The reader returned spectra in a different order than requested.")
        dense = binner.transform(raw).spectra  # (B, M)
        normalized = dataset.normalize_batch(dense.to(torch.float32))  # (B, M)
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError(f"Non-finite decoded spectra among source IDs {chunk[0]}..{chunk[-1]}.")
        spectra[start:start + chunk.size] = normalized.cpu().numpy()
        tic[start:start + chunk.size] = dense.abs().sum(dim=1).cpu().numpy()
        targets[start:start + chunk.size] = raw.targets.values[target_field].cpu().numpy() > 0.5
        mask[start:start + chunk.size] = raw.targets.masks[target_field].cpu().numpy()
    logger.debug("Decoded %s spectra with %s bins and %s classes.", ids.size, width, classes)
    return {"spectra": spectra, "targets": targets, "mask": mask, "tic": tic}


def class_positive_ids(dataset: Any, class_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Return, for every class, the source spectra carrying its in-range annotation.

    :param dataset: Annotation-aware dataset; its mapped index defines "in range".
    :type dataset: typing.Any
    :param class_names: ``formula|adduct`` identities of interest.
    :type class_names: tuple[str, ...]
    :return: Sorted source identifiers per class name (empty when never annotated).
    :rtype: dict[str, numpy.ndarray]
    """
    index = dataset.get_mapped_annotation_index()
    rows = np.repeat(np.arange(index.spectrum_ids.size), np.diff(index.spectrum_offsets))  # (E,)
    ## Integer (identity, spectrum) pairs; names are resolved once per identity
    pairs = np.unique(np.stack([np.asarray(index.annotation_indices, dtype=np.int64),
                                np.asarray(index.spectrum_ids, dtype=np.int64)[rows]], axis=1), axis=0)  # (P, 2)
    boundaries = np.searchsorted(pairs[:, 0], np.arange(len(index.annotation_identities) + 1))
    positions = {"|".join(identity): position for position, identity in enumerate(index.annotation_identities)}
    empty = np.empty(0, dtype=np.int64)
    return {name: (pairs[boundaries[positions[name]]:boundaries[positions[name] + 1], 1] if name in positions else empty)
            for name in class_names}


def select_test_extension(test_positives: dict[str, int], candidates: dict[str, np.ndarray], *,
                          minimum: int, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Draw extra evaluation pixels until every class has ``minimum`` positives.

    Classes are processed in sorted order with one seeded generator; a pixel drawn for
    one class also counts toward every other class it is positive for. When a class has
    too few candidates, all of them are taken and the shortfall is reported.

    :param test_positives: Positives already present in the test split, by class.
    :type test_positives: dict[str, int]
    :param candidates: Pool identifiers positive for each class.
    :type candidates: dict[str, numpy.ndarray]
    :param minimum: Required number of positives per class.
    :type minimum: int
    :param seed: Seed of the draw.
    :type seed: int
    :return: Sorted selected identifiers and one report row per class.
    :rtype: tuple[numpy.ndarray, pandas.DataFrame]
    """
    generator = np.random.default_rng(seed)
    selected: set[int] = set()
    rows = []
    for name in sorted(test_positives):
        pool = candidates.get(name, np.empty(0, dtype=np.int64))
        already = int(np.isin(pool, np.fromiter(selected, dtype=np.int64, count=len(selected))).sum()) if selected else 0
        current = int(test_positives[name]) + already
        needed = max(0, minimum - current)
        added = 0
        if needed:
            remaining = np.setdiff1d(pool, np.fromiter(selected, dtype=np.int64, count=len(selected)))
            draw = generator.permutation(remaining)[:needed]
            selected.update(int(value) for value in draw)
            added = int(draw.size)
        rows.append({"class_name": name, "test_positives": int(test_positives[name]),
                     "from_earlier_draws": already, "added": added, "final_positives": current + added,
                     "pool_candidates": int(pool.size), "shortfall": max(0, minimum - current - added)})
    return np.array(sorted(selected), dtype=np.int64), pd.DataFrame(rows)


# --------------------------------------------------
# Section: class catalogue and METASPACE reference images
# --------------------------------------------------

def class_mz_table(sqlite_path: Path | str) -> pd.DataFrame:
    """Return the median annotated m/z of every ``formula|adduct`` identity.

    :param sqlite_path: Merged annotation store.
    :type sqlite_path: pathlib.Path | str
    :return: ``class_name`` and ``mz``.
    :rtype: pandas.DataFrame
    """
    with sqlite3.connect(f"file:{Path(sqlite_path)}?mode=ro", uri=True) as connection:
        tables = [row[0] for row in connection.execute(
            "SELECT reference_table_name FROM datasets_metadata ORDER BY dataset_index")]
        frames = [pd.read_sql(f"SELECT formula, adduct, mz FROM {table}", connection) for table in tables]
    records = pd.concat(frames, ignore_index=True)
    records["class_name"] = records.formula + "|" + records.adduct
    return records.groupby("class_name", as_index=False).mz.median()


def class_set_labels(axis_classes: dict[str, tuple[str, ...]], class_mz: pd.Series,
                     reference_range: tuple[float, float]) -> pd.DataFrame:
    """Label every class by where its m/z lies relative to the common axis range.

    ``common``: head column on every axis. ``extension``: only on some axes and m/z
    outside the reference range. ``boundary``: only on some axes although its m/z lies
    inside the reference range (e.g. isotope bins crossing an axis edge).

    :param axis_classes: Head class names per axis.
    :type axis_classes: dict[str, tuple[str, ...]]
    :param class_mz: Annotated m/z indexed by class name.
    :type class_mz: pandas.Series
    :param reference_range: Inclusive common m/z range.
    :type reference_range: tuple[float, float]
    :return: ``class_name``, ``mz``, ``class_set`` and one Boolean column per axis.
    :rtype: pandas.DataFrame
    """
    names = sorted(set().union(*axis_classes.values()))
    frame = pd.DataFrame({"class_name": names})
    for axis, classes in axis_classes.items():
        frame[f"on_{axis}"] = frame.class_name.isin(classes)
    frame["mz"] = class_mz.reindex(frame.class_name).to_numpy()
    on_all = frame[[f"on_{axis}" for axis in axis_classes]].all(axis=1)
    inside = frame.mz.between(*reference_range)
    frame["class_set"] = np.where(on_all, "common", np.where(inside, "boundary", "extension"))
    return frame


_PIXEL_COLUMN = re.compile(r"^x(\d+)_y(\d+)$")


def read_metaspace_images(dataset_directory: Path | str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Read the METASPACE ion images exported beside one source dataset.

    :param dataset_directory: Directory containing ``pixel_intensities.csv``.
    :type dataset_directory: pathlib.Path | str
    :return: Annotation table (``class_name``, ``formula``, ``adduct``, ``mz``), pixel
        coordinates ``(P, 2)`` in METASPACE convention, and intensities ``(A, P)``.
    :rtype: tuple[pandas.DataFrame, numpy.ndarray, numpy.ndarray]
    """
    frame = pd.read_csv(Path(dataset_directory) / "pixel_intensities.csv")
    pixel_columns = [column for column in frame.columns if _PIXEL_COLUMN.match(column)]
    coordinates = np.array([[int(value) for value in _PIXEL_COLUMN.match(column).groups()]
                            for column in pixel_columns], dtype=np.int64)  # (P, 2)
    intensities = frame[pixel_columns].to_numpy(np.float32)  # (A, P)
    annotations = pd.DataFrame({"formula": frame.mol_formula, "adduct": frame.adduct, "mz": frame.mz})
    annotations["class_name"] = annotations.formula + "|" + annotations.adduct
    return annotations, coordinates, intensities


def align_metaspace(annotations_xy: np.ndarray, intensities: np.ndarray, pixel_xy: np.ndarray,
                    offset: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Align METASPACE pixel columns to analysis pixels.

    :param annotations_xy: METASPACE coordinates, shape ``(P, 2)``.
    :type annotations_xy: numpy.ndarray
    :param intensities: METASPACE intensities, shape ``(A, P)``.
    :type intensities: numpy.ndarray
    :param pixel_xy: Analysis pixel coordinates in imzML convention, shape ``(N, 2)``.
    :type pixel_xy: numpy.ndarray
    :param offset: Added to imzML coordinates to obtain METASPACE coordinates.
    :type offset: tuple[int, int]
    :return: Intensities per analysis pixel ``(N, A)`` (``nan`` where METASPACE has no
        column) and the covered fraction of analysis pixels.
    :rtype: tuple[numpy.ndarray, float]
    """
    lookup = {(int(x), int(y)): position for position, (x, y) in enumerate(annotations_xy)}
    shifted = pixel_xy + np.asarray(offset, dtype=np.int64)
    columns = np.array([lookup.get((int(x), int(y)), -1) for x, y in shifted], dtype=np.int64)  # (N,)
    aligned = np.full((pixel_xy.shape[0], intensities.shape[0]), np.nan, dtype=np.float32)  # (N, A)
    covered = columns >= 0
    aligned[covered] = intensities[:, columns[covered]].T
    return aligned, float(covered.mean()) if covered.size else float("nan")
