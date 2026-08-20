"""Atomic task progress snapshots for terminal and file-based monitoring."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import yaml
from tqdm import tqdm


def create_terminal_progress(
    *,
    total: int,
    initial: int = 0,
    description: str,
    position: int = 0,
    leave: bool = True,
) -> tqdm:
    """Create an interactive progress bar without corrupting redirected logs.

    :param total: Total number of work units represented by the bar.
    :type total: int
    :param initial: Completed work units at creation time.
    :type initial: int
    :param description: Short terminal label.
    :type description: str
    :param position: Terminal row reserved for the bar.
    :type position: int
    :param leave: Keep the final bar visible after completion.
    :type leave: bool
    :return: Configured tqdm progress bar.
    :rtype: tqdm.tqdm
    """
    return tqdm(
        total=total,
        initial=initial,
        desc=description,
        position=position,
        leave=leave,
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )


def format_duration(seconds: float | None) -> str:
    """Render a non-negative duration as a compact clock string.

    :param seconds: Duration in seconds.
    :type seconds: float | None
    :return: ``HH:MM:SS`` or ``--:--:--`` when unavailable.
    :rtype: str
    """
    if seconds is None or seconds < 0:
        return "--:--:--"
    whole_seconds = int(seconds)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def update_progress(path: Path, values: dict[str, Any]) -> None:
    """Atomically replace one task or campaign progress snapshot."""
    payload = dict(values)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)
    temporary.replace(path)
