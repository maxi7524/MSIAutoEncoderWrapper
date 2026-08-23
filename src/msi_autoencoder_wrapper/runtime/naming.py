"""Stable filesystem identities for experiment campaigns and model runs."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping


CONFIG_FINGERPRINT_LENGTH = 12
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def campaign_identifier(experiment_name: str, config_fingerprint: str) -> str:
    """Return the stable directory name for one exact experiment configuration.

    :param experiment_name: User-facing experiment name.
    :type experiment_name: str
    :param config_fingerprint: Full normalized configuration fingerprint.
    :type config_fingerprint: str
    :return: Collision-resistant campaign directory name.
    :rtype: str
    """
    return (
        f"{experiment_name}__cfg_"
        f"{config_fingerprint[:CONFIG_FINGERPRINT_LENGTH]}"
    )


def run_identifier(campaign_id: str, task: Mapping[str, Any]) -> str:
    """Return the persistent model name for one grid repetition.

    :param campaign_id: Stable campaign identifier.
    :type campaign_id: str
    :param task: Materialized task containing ``grid_id`` and ``repetition``.
    :type task: Mapping[str, Any]
    :return: Model-artifact name unique across campaigns and grid repetitions.
    :rtype: str
    """
    repetition = int(task["repetition"])
    return f"{campaign_id}__{task['grid_id']}__rep_{repetition:02d}"


def campaign_instance_identifier(campaign_id: str, run_id: str | None) -> str:
    """Add an optional independent execution identity to a campaign.

    :param campaign_id: Configuration-derived campaign identifier.
    :type campaign_id: str
    :param run_id: Optional user-selected independent execution identifier.
    :type run_id: str | None
    :return: Resume-stable campaign instance identifier.
    :rtype: str
    :raises ValueError: If ``run_id`` is unsafe as a path component.
    """
    if run_id is None:
        return campaign_id
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_' or '-'."
        )
    return f"{campaign_id}__run_{run_id}"


def scoped_directory(base: Path, campaign_id: str) -> Path:
    """Append a campaign identity to a user-configured directory basename.

    :param base: Explicitly configured campaign directory.
    :type base: pathlib.Path
    :param campaign_id: Stable campaign identifier.
    :type campaign_id: str
    :return: Sibling path whose basename contains the configuration identity.
    :rtype: pathlib.Path
    """
    return base.parent / campaign_id
