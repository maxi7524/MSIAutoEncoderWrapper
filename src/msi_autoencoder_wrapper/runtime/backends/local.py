"""Bounded local task execution using persistent worker processes."""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run_local_tasks(task_paths: list[Path], max_parallel_runs: int) -> None:
    """Execute task descriptors in persistent, bounded worker processes.

    Each parallel worker receives a sequence of tasks instead of one task. Runtime
    factories can therefore retain expensive read-only dataset resources in a
    process-local cache while models, seeds, splits, and training remain
    task-specific.
    """
    if max_parallel_runs < 1:
        raise ValueError("max_parallel_runs must be positive")
    if not task_paths:
        return

    # Worker allocation
    ## Round-robin assignment balances task counts without spawning per-task processes
    worker_count = min(max_parallel_runs, len(task_paths))
    chunks = [task_paths[index::worker_count] for index in range(worker_count)]

    def execute(paths: list[Path]) -> None:
        # Persistent worker process
        ## All task factories in this command share the worker's in-memory caches
        subprocess.run(
            [
                sys.executable,
                "-m",
                "msi_autoencoder_wrapper.runtime.cli",
                "task-batch",
                *(str(path) for path in paths),
            ],
            check=True,
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(execute, chunk) for chunk in chunks]
        for future in futures:
            future.result()
