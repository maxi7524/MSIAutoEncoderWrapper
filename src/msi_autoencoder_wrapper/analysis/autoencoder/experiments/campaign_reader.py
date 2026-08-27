"""Generic reader for ``msi-wrapper`` grid-experiment campaigns.

A campaign is the on-disk output of one ``runtime.cli`` execution of an experiment
config that sweeps a Cartesian grid of parameters over several repetitions (see
``runtime.planning.plan.build_plan``). This module knows only the campaign's
*storage layout* (``configs/execution/<experiment_name>/status/*.yaml`` manifests,
each pointing at a ``models/<context>/<task_id>/config/{config.json,history.json}``
artifact folder written by ``ModelStore``) — it does not know what any particular
experiment's grid axes mean (architecture, binning step, learning rate, ...). Callers
interpret ``CampaignTask.grid_parameters`` and the loaded ``model_config``/``history``
payloads according to their own experiment config.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import json
import yaml

from ....utils.exceptions import raise_workspace_error
from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@dataclass(frozen=True)
class CampaignTask:
    """One materialized task of a grid-experiment campaign.

    :param task_id: Stable task identifier (``task_XXXXXX``), unique within the campaign.
    :type task_id: str
    :param status: Terminal or in-progress status recorded by the runtime (``completed``,
        ``failed``, ``running``, ...).
    :type status: str
    :param grid_parameters: Raw grid-axis values resolved for this task, exactly as the
        experiment config's ``grid`` section defines them (e.g. ``{"architectures": {...},
        "binning_steps": 0.55}``). This reader does not assume fixed axis names.
    :type grid_parameters: Mapping[str, Any]
    :param repetition: Zero-based repetition index within this task's grid cell.
    :type repetition: int | None
    :param result: Raw ``result`` mapping from the status record (e.g. ``model_path``,
        ``epochs``), or ``None`` if the task has no terminal result yet.
    :type result: Mapping[str, Any] | None
    :param model_config: Parsed ``config.json`` of the persisted model artifact, or
        ``None`` if artifacts were not requested or are unavailable.
    :type model_config: Mapping[str, Any] | None
    :param history: Parsed ``history.json`` of the persisted model artifact (one entry
        per training-loop callback invocation), or ``None`` under the same conditions as
        ``model_config``. Not every entry represents a training epoch — entries carry a
        ``metrics.epoch`` key only for epoch records; a trailing entry with a ``split``
        key instead (e.g. ``split: test``) is a one-off post-training evaluation and has
        no ``epoch``/``duration``. Callers computing per-epoch curves must filter on
        ``metrics.epoch is not None``.
    :type history: list[Any] | None
    """

    task_id: str
    status: str
    grid_parameters: Mapping[str, Any]
    repetition: Optional[int]
    result: Optional[Mapping[str, Any]]
    model_config: Optional[Mapping[str, Any]]
    history: Optional[list]


def read_campaign(
    workspace: Path | str,
    experiment_name: str,
    *,
    load_artifacts: bool = True,
) -> list[CampaignTask]:
    """Read every materialized task of one grid-experiment campaign.

    One :class:`CampaignTask` is returned per task manifest found under
    ``<workspace>/configs/execution/<experiment_name>/status/``. Per-task progress
    snapshots (``task_XXXXXX-progress.yaml``, written continuously during training)
    are not manifests and are skipped — only files matching ``task_*.yaml`` without
    that suffix are read.

    :param workspace: Project workspace root (the directory containing ``configs/``
        and ``models/``).
    :type workspace: pathlib.Path | str
    :param experiment_name: ``experiment.name`` of the campaign's config, i.e. the
        directory name under ``configs/execution/``.
    :type experiment_name: str
    :param load_artifacts: Also load each ``completed`` task's ``config.json`` and
        ``history.json`` from ``result.model_path``. Disable to only inspect grid
        coverage and status without touching model artifacts.
    :type load_artifacts: bool
    :return: One task record per manifest, in ``task_id`` sort order.
    :rtype: list[CampaignTask]
    :raises WorkspaceConfigError: If the campaign has no ``status`` directory, or if
        its ``__cfg_<fingerprint>``-suffixed directory cannot be resolved unambiguously.
    """
    status_directory = _resolve_status_directory(Path(workspace), experiment_name)

    # Manifest discovery
    ## Progress snapshots share the "task_*" prefix but end in "-progress.yaml".
    manifest_paths = sorted(
        path
        for path in status_directory.glob("task_*.yaml")
        if not path.name.endswith("-progress.yaml")
    )
    logger.info(
        "Reading %s task manifest(s) for campaign '%s'.",
        len(manifest_paths),
        experiment_name,
    )

    tasks: list[CampaignTask] = []
    for manifest_path in manifest_paths:
        with manifest_path.open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
        records = (payload or {}).get("records", {})
        for task_id, record in records.items():
            tasks.append(_build_task(task_id, record, load_artifacts=load_artifacts))

    tasks.sort(key=lambda task: task.task_id)
    status_counts = Counter(task.status for task in tasks)
    logger.info(
        "Campaign '%s' status breakdown: %s.",
        experiment_name,
        dict(status_counts),
    )
    return tasks


def _resolve_status_directory(workspace: Path, experiment_name: str) -> Path:
    """Locate one campaign's ``status`` directory under ``configs/execution``.

    The runtime CLI namespaces every campaign directory as
    ``<experiment_name>__cfg_<fingerprint>`` (``runtime.naming.campaign_identifier``),
    so the plain ``<experiment_name>`` directory this reader originally expected no
    longer exists for campaigns materialized after that change. The exact plain name
    is tried first for backward compatibility with any pre-existing directory or
    compatibility symlink; only when that is absent does this fall back to the single
    ``<experiment_name>__cfg_*`` match.

    :param workspace: Project workspace root.
    :type workspace: pathlib.Path
    :param experiment_name: ``experiment.name`` of the campaign's config.
    :type experiment_name: str
    :return: Resolved ``status`` directory.
    :rtype: pathlib.Path
    :raises WorkspaceConfigError: If neither the exact directory nor exactly one
        ``__cfg_*``-suffixed directory can be resolved.
    """
    execution_root = workspace / "configs" / "execution"
    exact = execution_root / experiment_name / "status"
    if exact.is_dir():
        return exact

    candidates = sorted(execution_root.glob(f"{experiment_name}__cfg_*"))
    matching = [candidate for candidate in candidates if (candidate / "status").is_dir()]
    if len(matching) == 1:
        logger.info(
            "Resolved campaign '%s' to namespaced directory '%s'.",
            experiment_name,
            matching[0].name,
        )
        return matching[0] / "status"
    if len(matching) > 1:
        raise_workspace_error(
            context_name="CampaignReader",
            message=(
                f"Experiment '{experiment_name}' under workspace '{workspace}' has "
                f"{len(matching)} distinct '__cfg_*' campaign directories with a "
                "status/ subdirectory: "
                f"{[candidate.name for candidate in matching]}. Pass the exact "
                "'<experiment_name>__cfg_<fingerprint>' directory name instead."
            ),
        )
    raise_workspace_error(
        context_name="CampaignReader",
        message=(
            f"No status directory for experiment '{experiment_name}' under "
            f"workspace '{workspace}' (tried '{exact}' and "
            f"'{execution_root}/{experiment_name}__cfg_*/status')."
        ),
    )


def _build_task(
    task_id: str,
    record: Mapping[str, Any],
    *,
    load_artifacts: bool,
) -> CampaignTask:
    """Assemble one :class:`CampaignTask` from a raw manifest record."""
    task_definition = record.get("task", {})
    status = record.get("status", "unknown")
    result = record.get("result")
    model_config: Optional[Mapping[str, Any]] = None
    history: Optional[list] = None
    if (
        load_artifacts
        and status == "completed"
        and isinstance(result, Mapping)
        and result.get("model_path")
    ):
        model_config, history = _load_model_artifacts(Path(result["model_path"]))
    return CampaignTask(
        task_id=task_id,
        status=status,
        grid_parameters=task_definition.get("grid_parameters", {}),
        repetition=task_definition.get("repetition"),
        result=result,
        model_config=model_config,
        history=history,
    )


def _load_model_artifacts(
    model_directory: Path,
) -> tuple[Optional[Mapping[str, Any]], Optional[list]]:
    """Load the ``ModelStore``-layout ``config.json``/``history.json`` for one task."""
    model_config = _read_json(model_directory / "config" / "config.json")
    history = _read_json(model_directory / "config" / "history.json")
    return model_config, history


def _read_json(path: Path) -> Any:
    """Parse one JSON artifact, logging and returning ``None`` if it is missing."""
    if not path.is_file():
        logger.warning("Expected campaign artifact is missing: %s", path)
        return None
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)
