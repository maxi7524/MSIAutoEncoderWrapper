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

    # The campaign identifier is the run directory two levels above ``plan/status``
    # (``<entropy_runs>/<campaign_id>/plan/status``). It is only needed as a fallback
    # for locating stored artifacts; see :func:`_resolve_artifact_directory`.
    campaign_id = _campaign_identifier(status_path)

    tasks: list[CampaignTask] = []
    for manifest_path in manifest_paths:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        for task_id, record in payload.get("records", {}).items():
            tasks.append(
                _build_entropy_task(
                    task_id,
                    record,
                    model_store_path,
                    load_artifacts=load_artifacts,
                    campaign_id=campaign_id,
                )
            )

    tasks.sort(key=lambda task: task.task_id)
    logger.info(
        "Entropy campaign at '%s': %s task(s) read.", status_path, len(tasks)
    )
    return tasks


def _campaign_identifier(status_path: Path) -> Optional[str]:
    """Derive the campaign identifier from a ``<campaign_id>/plan/status`` path.

    :param status_path: The run's ``plan/status`` directory.
    :type status_path: pathlib.Path
    :return: The campaign directory name, or ``None`` if the path does not have the
        expected ``plan/status`` suffix.
    :rtype: str | None
    """
    resolved = status_path.resolve()
    if resolved.name != "status" or resolved.parent.name != "plan":
        return None
    return resolved.parent.parent.name


def _resolve_artifact_directory(
    model_store_directory: Path,
    runtime_model_name: Optional[str],
    campaign_id: Optional[str],
    task_id: str,
) -> Optional[Path]:
    """Locate one task's saved ``config/`` directory across both store conventions.

    Two naming conventions coexist in a model store, because the finalize step that
    copies models back from the compute node changed over time:

    1. ``<runtime.model_name>/`` — the namespaced identifier written into the task
       manifest by ``runtime.naming.run_identifier`` (e.g.
       ``kidney-nnpu-prior-sensitivity__cfg_b89b9dfd__grid_0000__rep_00``);
    2. ``<campaign_id>__<task_id>/`` — the campaign-scoped identifier used by later
       campaigns (e.g. ``contractive-20260905-01__task_000034``).

    The manifest only records the first, so a store written under the second
    convention yields no artifacts at all unless the fallback is tried. The manifest
    name is checked first, so a store using convention 1 resolves exactly as before.

    :param model_store_directory: Directory holding one subdirectory per saved model.
    :type model_store_directory: pathlib.Path
    :param runtime_model_name: ``task.runtime.model_name`` from the manifest.
    :type runtime_model_name: str | None
    :param campaign_id: Campaign identifier derived from the status directory path.
    :type campaign_id: str | None
    :param task_id: Task identifier (``task_XXXXXX``).
    :type task_id: str
    :return: The task's ``config/`` directory, or ``None`` if neither exists.
    :rtype: pathlib.Path | None
    """
    candidates = []
    if runtime_model_name:
        candidates.append(model_store_directory / runtime_model_name)
    if campaign_id:
        candidates.append(model_store_directory / f"{campaign_id}__{task_id}")
    for candidate in candidates:
        if (candidate / "config").is_dir():
            return candidate / "config"
    return None


def _build_entropy_task(
    task_id: str,
    record: dict,
    model_store_directory: Path,
    *,
    load_artifacts: bool,
    campaign_id: Optional[str] = None,
) -> CampaignTask:
    """Assemble one :class:`CampaignTask` from a raw entropy manifest record."""
    task_definition = record.get("task", {})
    status = record.get("status", "unknown")
    result = record.get("result")
    model_name: Optional[str] = task_definition.get("runtime", {}).get("model_name")

    model_config = None
    history = None
    if load_artifacts and status == "completed":
        model_directory = _resolve_artifact_directory(
            model_store_directory, model_name, campaign_id, task_id
        )
        if model_directory is None:
            logger.warning(
                "No stored artifact directory for task '%s' under '%s' (tried the "
                "manifest name '%s' and the campaign name '%s__%s').",
                task_id,
                model_store_directory,
                model_name,
                campaign_id,
                task_id,
            )
        else:
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
