"""Slurm batch script generation and submission."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any


def write_sbatch_script(plan_directory: Path, task_count: int, options: dict[str, Any]) -> Path:
    """Write a job-array script that selects one materialized task by index."""
    if task_count < 1:
        raise ValueError("A Slurm plan requires at least one task")
    parallelism = int(options.get("array_parallelism", 1))
    directives = [
        "#!/usr/bin/env bash",
        f"#SBATCH --array=0-{task_count - 1}%{parallelism}",
    ]
    mapping = {
        "partition": "partition",
        "qos": "qos",
        "account": "account",
        "nodelist": "nodelist",
        "time": "time",
        "cpus_per_task": "cpus-per-task",
        "memory": "mem",
    }
    for key, flag in mapping.items():
        if options.get(key) is not None:
            value = str(options[key])
            if "\n" in value or "\r" in value:
                raise ValueError(f"Invalid newline in Slurm option: {key}")
            directives.append(f"#SBATCH --{flag}={value}")
    if options.get("gpus_per_task") is not None:
        gpu_count = str(options["gpus_per_task"])
        if "\n" in gpu_count or "\r" in gpu_count:
            raise ValueError("Invalid newline in Slurm option: gpus_per_task")
        # REMARK: Entropy documents GPU requests through the portable GRES
        # syntax. Its Slurm deployment rejects --gpus-per-task.
        directives.append(f"#SBATCH --gres=gpu:{gpu_count}")
    tasks = (plan_directory / "tasks").resolve()
    task_pattern = shlex.quote(str(tasks / "task_%06d.yaml"))
    python = shlex.quote(sys.executable)
    directives.extend(
        [
            "set -euo pipefail",
            f'TASK_FILE=$(printf {task_pattern} "$SLURM_ARRAY_TASK_ID")',
            f'{python} -m msi_autoencoder_wrapper.runtime.cli task "$TASK_FILE"',
        ]
    )
    path = plan_directory / "run.sbatch"
    path.write_text("\n".join(directives) + "\n", encoding="utf-8")
    return path


def build_sbatch_command(script: Path, *, parsable: bool = False) -> list[str]:
    """Return the argument vector used to submit a generated script."""
    return ["sbatch", *(("--parsable",) if parsable else ()), str(script.resolve())]


def write_finalize_script(
    plan_directory: Path,
    *,
    job_id: str,
    config_path: Path,
    persistent_directory: Path,
    staging_directory: Path,
    execution_id: str,
) -> Path:
    """Write a dependent job that restores results, reports and cleans RAM."""
    path = plan_directory / "finalize.sbatch"
    python = shlex.quote(sys.executable)
    config = shlex.quote(str(config_path.resolve()))
    staging = shlex.quote(str(staging_directory.resolve()))
    persistent = shlex.quote(str(persistent_directory.resolve()))
    identifier = shlex.quote(execution_id)
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --dependency=afterany:{job_id}",
        "set -euo pipefail",
        f"{python} -m msi_autoencoder_wrapper.runtime.cli finalize "
        f"{config} --staging-directory {staging} "
        f"--output {persistent} --execution-id {identifier}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
