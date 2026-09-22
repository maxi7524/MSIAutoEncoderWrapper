"""Slurm batch script generation and submission."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any, Sequence


def write_sbatch_script(
    plan_directory: Path,
    tasks: int | Sequence[str],
    options: dict[str, Any],
    *,
    dependency_job_id: str | None = None,
    script_name: str = "run.sbatch",
) -> Path:
    """Write a job-array script for one dependency-safe task layer."""
    task_count = tasks if isinstance(tasks, int) else len(tasks)
    if task_count < 1:
        raise ValueError("A Slurm plan requires at least one task")
    parallelism = int(options.get("array_parallelism", 1))
    directives = [
        "#!/usr/bin/env bash",
        f"#SBATCH --array=0-{task_count - 1}%{parallelism}",
    ]
    if dependency_job_id is not None:
        if not dependency_job_id.isdigit():
            raise ValueError("dependency_job_id must contain only digits.")
        directives.append(f"#SBATCH --dependency=afterany:{dependency_job_id}")
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
    python = shlex.quote(sys.executable)
    directives.append("set -euo pipefail")
    if isinstance(tasks, int):
        task_pattern = shlex.quote(str((plan_directory / "tasks") / "task_%06d.yaml"))
        directives.append(
            f'TASK_FILE=$(printf {task_pattern} "$SLURM_ARRAY_TASK_ID")'
        )
    else:
        task_paths = [
            shlex.quote(str((plan_directory / "tasks" / f"{task_id}.yaml").resolve()))
            for task_id in tasks
        ]
        directives.append(f"TASK_FILES=({' '.join(task_paths)})")
        directives.append('TASK_FILE="${TASK_FILES[$SLURM_ARRAY_TASK_ID]}"')
    directives.append(
        f'{python} -m msi_autoencoder_wrapper.runtime.cli task "$TASK_FILE"'
    )
    path = plan_directory / script_name
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
