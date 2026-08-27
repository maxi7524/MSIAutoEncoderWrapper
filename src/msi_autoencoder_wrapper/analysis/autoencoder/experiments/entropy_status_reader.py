"""Reader for campaigns submitted through the SLURM (entropy) execution backend.

Campaigns run locally materialize their task manifests under
``<workspace>/configs/execution/<experiment_name>/status/`` (see
``campaign_reader.read_campaign``). Campaigns submitted through
``assets/scripts/entropy/`` instead stage tasks, checkpoints, and status under a
separate SLURM orchestration directory (``<entropy_run>/plan/status/``), entirely
outside the project workspace, because the SLURM array jobs execute on compute nodes
with their own ephemeral local storage. This module reads that second layout into the
same :class:`~.campaign_reader.CampaignTask` shape so every downstream analysis
function (``training_curves_frame``, per-objective grouping, ...) works identically
regardless of which backend produced the campaign.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from ....utils.logger import get_custom_logger
from .campaign_reader import CampaignTask, _read_json

logger = get_custom_logger(__name__)


def read_entropy_campaign(
    status_directory: Path | str,
    model_store_directory: Path | str,
    *,
    load_artifacts: bool = True,
) -> list[CampaignTask]:
    """Read every materialized task of one SLURM-submitted campaign.

    Each status record's own ``result.model_path`` is the SLURM compute node's
    ephemeral local scratch path (e.g. ``/tmp/<user>/msi-wrapper/<run>/...``) and is
    never used here — by the time this is read, that path no longer exists on any
    machine. Instead, ``task.runtime.model_name`` (the stable, namespaced model
    identifier written by ``runtime.naming.run_identifier``) is used to resolve the
    artifact directory under ``model_store_directory`` — the same directory a local
    workspace's ``models/<context>/`` uses, which is where the finalize step actually
    copies completed models back to.

    :param status_directory: The entropy run's ``plan/status/`` directory (e.g.
        ``~/entropy-runs/<experiment_name>/<run>/plan/status``).
    :type status_directory: pathlib.Path | str
    :param model_store_directory: Directory containing one subdirectory per saved
        model (e.g. ``data/<workspace>/models/<context>``), matching
        ``ModelStore``'s layout (``<model_name>/config/{config.json,history.json}``).
    :type model_store_directory: pathlib.Path | str
    :param load_artifacts: Also load each ``completed`` task's ``config.json``/
        ``history.json`` from ``model_store_directory``. Disable to only inspect grid
        coverage and status.
    :type load_artifacts: bool
    :return: One task record per manifest, in ``task_id`` sort order.
    :rtype: list[CampaignTask]
    :raises FileNotFoundError: If ``status_directory`` does not exist.
    """
    status_path = Path(status_directory)
    if not status_path.is_dir():
        raise FileNotFoundError(
            f"No entropy status directory at '{status_path}'. Sync it locally first "
            "(see tmp/runny/runny.md's rsync recipe) before reading this campaign."
        )
    model_store_path = Path(model_store_directory)

    manifest_paths = sorted(
        path for path in status_path.glob("task_*.yaml") if not path.name.endswith("-progress.yaml")
    )
    logger.info(
        "Reading %s entropy task manifest(s) from '%s'.",
        len(manifest_paths),
        status_path,
    )

    tasks: list[CampaignTask] = []
    for manifest_path in manifest_paths:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        for task_id, record in payload.get("records", {}).items():
            tasks.append(
                _build_entropy_task(
                    task_id, record, model_store_path, load_artifacts=load_artifacts
                )
            )

    tasks.sort(key=lambda task: task.task_id)
    logger.info(
        "Entropy campaign at '%s': %s task(s) read.", status_path, len(tasks)
    )
    return tasks


def _build_entropy_task(
    task_id: str,
    record: dict,
    model_store_directory: Path,
    *,
    load_artifacts: bool,
) -> CampaignTask:
    """Assemble one :class:`CampaignTask` from a raw entropy manifest record."""
    task_definition = record.get("task", {})
    status = record.get("status", "unknown")
    result = record.get("result")
    model_name: Optional[str] = task_definition.get("runtime", {}).get("model_name")

    model_config = None
    history = None
    if load_artifacts and status == "completed" and model_name:
        model_directory = model_store_directory / model_name / "config"
        model_config = _read_json(model_directory / "config.json")
        history = _read_json(model_directory / "history.json")

    return CampaignTask(
        task_id=task_id,
        status=status,
        grid_parameters=task_definition.get("grid_parameters", {}),
        repetition=task_definition.get("repetition"),
        result=result,
        model_config=model_config,
        history=history,
    )
