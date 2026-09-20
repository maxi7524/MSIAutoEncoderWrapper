"""Measure artifact construction/loading and batch rendering for one task file.

Run after materializing a campaign:

    uv run python assets/scripts/benchmarks/benchmark_precomputed_synthetic.py \
        --run-directory data/kidney_workspace/configs/entropy-runs/<campaign-id> \
        --phase-name synthetic_axis_pretraining
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import yaml

from msi_autoencoder_wrapper.training.precompute import SyntheticPrecomputePhase
from msi_autoencoder_wrapper.runtime.workflows.wrapper import _build_wrapper


def _load_task(path: Path) -> dict[str, Any]:
    """Load one materialized runtime task."""
    with path.open(encoding="utf-8") as stream:
        task = yaml.safe_load(stream)
    if not isinstance(task, dict):
        raise ValueError("Task file root must be a mapping.")
    return task


def _find_artifact_task(run_directory: Path) -> Path:
    """Find the first artifact task in either supported plan layout."""
    task_directories = (
        run_directory / "plan" / "tasks",
        run_directory / "tasks",
    )
    task_paths = [
        task_path
        for task_directory in task_directories
        for task_path in sorted(task_directory.glob("task_*.yaml"))
    ]
    if not task_paths:
        available = sorted(
            str(path)
            for path in run_directory.glob("**/task_*.yaml")
            if path.is_file()
        )[:5]
        raise FileNotFoundError(
            "No materialized task files found. Checked: "
            f"{', '.join(str(path) for path in task_directories)}. "
            f"Nearby task files: {available or 'none'}."
        )
    for task_path in task_paths:
        task = _load_task(task_path)
        phases = task.get("parameters", {}).get("training", {}).get("phases", [])
        if any(
            phase.get("pretraining", {}).get("kind") == "precomputed_synthetic"
            for phase in phases
        ):
            return task_path
    raise ValueError(
        "The materialized campaign contains no kind=precomputed_synthetic phase."
    )


def _select_phase(task: dict[str, Any], phase_name: str | None) -> dict[str, Any]:
    """Resolve one explicit artifact-backed phase from task training settings."""
    phases = task["parameters"]["training"].get("phases", [])
    candidates = [
        phase
        for phase in phases
        if phase.get("pretraining", {}).get("kind") == "precomputed_synthetic"
    ]
    if phase_name is not None:
        candidates = [
            phase for phase in candidates if phase.get("phase_name") == phase_name
        ]
    if len(candidates) != 1:
        raise ValueError(
            "Select exactly one precomputed phase with --phase-name. "
            f"Available phases: {[phase.get('phase_name') for phase in phases]}."
        )
    return candidates[0]


def main() -> None:
    """Build/load one artifact twice and report batch-rendering throughput."""
    parser = argparse.ArgumentParser(
        description="Benchmark persistent synthetic precompute for a materialized task."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task-file", type=Path)
    source.add_argument(
        "--run-directory",
        type=Path,
        help="Entropy run directory containing plan/tasks/task_*.yaml.",
    )
    parser.add_argument("--phase-name")
    parser.add_argument("--batches", type=int, default=16)
    arguments = parser.parse_args()
    if arguments.batches < 1:
        raise ValueError("--batches must be positive.")

    task_path = (
        arguments.task_file
        if arguments.task_file is not None
        else _find_artifact_task(arguments.run_directory)
    )
    task = _load_task(task_path)
    phase = _select_phase(task, arguments.phase_name)
    wrapper = _build_wrapper(task)
    precompute = SyntheticPrecomputePhase(phase["pretraining"])

    started = time.perf_counter()
    partitions = precompute.build_partitions(wrapper.active_dataset)
    first_seconds = time.perf_counter() - started

    started = time.perf_counter()
    repeated = precompute.build_partitions(wrapper.active_dataset)
    cache_load_seconds = time.perf_counter() - started

    dataset = partitions["train"]
    batch_size = int(phase.get("batch_size", 64))
    rendered = 0
    started = time.perf_counter()
    for start in range(0, min(len(dataset), arguments.batches * batch_size), batch_size):
        rows = list(range(start, min(start + batch_size, len(dataset))))
        batch = dataset.collate_fn(rows)
        rendered += batch.batch_size
    render_seconds = time.perf_counter() - started

    artifact = repeated["train"].artifact
    print(
        json.dumps(
            {
                "artifact_key": artifact.artifact_key,
                "fingerprint": artifact.fingerprint,
                "task_file": str(task_path),
                "population": dataset.population,
                "feature_count": artifact.feature_count,
                "prototype_count": artifact.prototype_count,
                "manifest_samples": len(dataset),
                "first_prepare_seconds": first_seconds,
                "matching_cache_load_seconds": cache_load_seconds,
                "rendered_samples": rendered,
                "render_seconds": render_seconds,
                "rendered_samples_per_second": rendered / max(render_seconds, 1e-12),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
