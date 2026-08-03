"""Bounded local task execution using isolated Python processes."""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run_local_tasks(task_paths: list[Path], max_parallel_runs: int) -> None:
    """Execute task descriptors with bounded process concurrency."""
    if max_parallel_runs < 1:
        raise ValueError("max_parallel_runs must be positive")

    def execute(path: Path) -> None:
        subprocess.run(
            [sys.executable, "-m", "msi_autoencoder_wrapper.execution.cli", "task", str(path)],
            check=True,
        )

    with ThreadPoolExecutor(max_workers=max_parallel_runs) as executor:
        futures = [executor.submit(execute, path) for path in task_paths]
        for future in futures:
            future.result()
