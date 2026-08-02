"""Preflight checks performed before staging or scheduler submission."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any
from dataclasses import asdict

from .entrypoints import resolve_entrypoint
from .planning import build_plan


def validate_preflight(
    config: dict[str, Any],
    *,
    require_external_tools: bool = False,
) -> list[str]:
    """Validate a complete plan without reserving scheduler resources."""
    messages: list[str] = []
    resolve_entrypoint(config["task"]["entrypoint"])
    plan = build_plan(config)
    messages.append(f"Prepared {len(plan.tasks)} tasks")
    preflight = resolve_entrypoint(config["task"]["preflight_entrypoint"])
    result = preflight(asdict(plan.tasks[0]))
    messages.append(f"Pipeline preflight completed: {result!r}")
    directory = Path(config["_config_directory"])
    for report in config.get("reports", []):
        definition = {"source": report} if isinstance(report, str) else report
        source = (directory / definition["source"]).resolve()
        if not source.is_file() or source.suffix.lower() not in {".qmd", ".rmd"}:
            raise ValueError(f"Report source does not exist or is unsupported: {source}")
    backend = config.get("execution", {}).get("backend", "local")
    required = ["sbatch"] if backend == "slurm" else []
    if config.get("reports"):
        required.append("quarto")
    for command in required:
        if shutil.which(command) is None:
            if require_external_tools:
                raise ValueError(f"Required command is unavailable: {command}")
            messages.append(f"Command unavailable in validation environment: {command}")
    if importlib.util.find_spec("torch") is None:
        raise ValueError("Torch is unavailable")
    return messages
