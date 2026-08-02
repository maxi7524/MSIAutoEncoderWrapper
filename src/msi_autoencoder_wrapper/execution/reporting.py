"""Sequential Quarto report rendering."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ..utils.logger import get_custom_logger
from .manifests import update_manifest

logger = get_custom_logger(__name__)


def render_reports(
    reports: list[Any] | tuple[Any, ...],
    *,
    config_directory: Path,
    manifest_path: Path,
    only: str | None = None,
    continue_on_error: bool = True,
) -> bool:
    """Render every configured report in order and record independent statuses."""
    all_succeeded = True
    for index, report in enumerate(reports):
        definition = {"source": report} if isinstance(report, str) else dict(report)
        source = (config_directory / definition["source"]).resolve()
        name = definition.get("name", source.stem)
        if only is not None and name != only:
            continue
        formats = definition.get("formats", [None])
        report_succeeded = True
        for output_format in formats:
            command = ["quarto", "render", str(source)]
            if output_format:
                command.extend(["--to", str(output_format)])
            logger.info("Rendering report '%s' (%s/%s).", name, index + 1, len(reports))
            try:
                subprocess.run(command, check=True)
            except (OSError, subprocess.CalledProcessError) as error:
                report_succeeded = False
                all_succeeded = False
                update_manifest(
                    manifest_path,
                    name,
                    {"status": "failed", "source": str(source), "error": str(error)},
                )
                if not continue_on_error:
                    return False
                break
        if report_succeeded:
            update_manifest(
                manifest_path,
                name,
                {"status": "completed", "source": str(source), "formats": formats},
            )
    return all_succeeded
