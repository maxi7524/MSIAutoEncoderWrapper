"""Measure artifact construction/loading and batch rendering for one task file.

Run after materializing a campaign:

    uv run python assets/scripts/benchmarks/benchmark_precomputed_synthetic.py \
        --run-directory data/kidney_workspace/configs/entropy-runs/<campaign-id> \
        --phase-name synthetic_axis_pretraining
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np
import yaml
from pyimzml.ImzMLParser import ImzMLParser

from msi_autoencoder_wrapper.training.precompute import SyntheticPrecomputePhase
from msi_autoencoder_wrapper.runtime.workflows.wrapper import _build_wrapper
from msi_autoencoder_wrapper.runtime import build_plan, load_experiment_config
from msi_autoencoder_wrapper.runtime.workflows.configured import (
    _build_planning_pipeline,
)


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


def _build_spatial_probe(
    config_path: Path,
    *,
    phase_name: str | None,
    mass_min: float,
    mass_max: float,
    spatial_side: int,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Build a real-data precompute probe without materializing a campaign.

    The selected source rows are represented as unit half-open ranges. This
    preserves the dataset's normal annotation and source-ID contracts while
    constraining the probe to one spatial square.
    """
    config = load_experiment_config(config_path)
    task = next(
        planned
        for planned in build_plan(config).tasks
        if any(
            phase.get("pretraining", {}).get("kind") == "precomputed_synthetic"
            for phase in planned.parameters["training"]["phases"]
        )
    )
    factory_parameters = deepcopy(task.parameters["factory_parameters"])
    project_path = Path(factory_parameters["project_path"])
    image_path = Path(factory_parameters["image_path"])
    resolved_image = image_path if image_path.is_absolute() else project_path / image_path
    parser = ImzMLParser(str(resolved_image))
    coordinates = np.asarray(parser.coordinates, dtype=np.int64)
    x_minimum, y_minimum, z_minimum = coordinates.min(axis=0)
    selected = np.flatnonzero(
        (coordinates[:, 0] >= x_minimum)
        & (coordinates[:, 0] < x_minimum + spatial_side)
        & (coordinates[:, 1] >= y_minimum)
        & (coordinates[:, 1] < y_minimum + spatial_side)
        & (coordinates[:, 2] == z_minimum)
    ).astype(np.int64)
    if selected.size < 1:
        raise ValueError("The requested spatial benchmark square is empty.")
    factory_parameters.pop("split_reference_binning", None)
    factory_parameters["binning"] = {
        "strategy": "LinearBinning",
        "parameters": {
            "bin_step": 0.55,
            "x_min": mass_min,
            "x_max": mass_max,
            "aggregation": "sum",
        },
    }
    dataset_parameters = factory_parameters["dataset"]["parameters"]
    dataset_parameters.pop("subset", None)
    dataset_parameters["split"] = {
        "strategy": "random",
        "fractions": {"train": 1.0, "validation": 0.0, "test": 0.0},
    }
    dataset_parameters["source_population"] = {
        "spectrum_ranges": [[int(index), int(index) + 1] for index in selected]
    }
    wrapper, dataset = _build_planning_pipeline(
        factory_parameters,
        split_seed=int(task.reproducibility["common_seeds"]["split"]),
    )
    phase = deepcopy(_select_phase(
        {"parameters": {"training": task.parameters["training"]}},
        phase_name,
    ))
    selected_population = phase["pretraining"]["population"]
    phase["pretraining"]["artifact"]["populations"] = {
        selected_population: phase["pretraining"]["artifact"]["populations"][
            selected_population
        ]
    }
    return wrapper, phase, {
        "spatial_side": spatial_side,
        "source_spectra": int(selected.size),
        "spatial_origin": [int(x_minimum), int(y_minimum), int(z_minimum)],
        "mass_range": [mass_min, mass_max],
    }


def _peak_rss_bytes() -> int:
    """Return process peak resident memory in bytes on Linux."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


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
    source.add_argument(
        "--config",
        type=Path,
        help="Run a bounded real-data spatial probe directly from campaign YAML.",
    )
    parser.add_argument("--phase-name")
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--spatial-side", type=int, default=10)
    parser.add_argument("--mass-min", type=float, default=150.0)
    parser.add_argument("--mass-max", type=float, default=200.0)
    arguments = parser.parse_args()
    if arguments.batches < 1:
        raise ValueError("--batches must be positive.")
    if arguments.spatial_side < 1 or arguments.mass_min >= arguments.mass_max:
        raise ValueError("Probe bounds must define a nonempty spatial and mass range.")

    probe: dict[str, Any] | None = None
    if arguments.config is not None:
        wrapper, phase, probe = _build_spatial_probe(
            arguments.config,
            phase_name=arguments.phase_name,
            mass_min=arguments.mass_min,
            mass_max=arguments.mass_max,
            spatial_side=arguments.spatial_side,
        )
        task_path = None
    else:
        task_path = (
            arguments.task_file
            if arguments.task_file is not None
            else _find_artifact_task(arguments.run_directory)
        )
        task = _load_task(task_path)
        phase = _select_phase(task, arguments.phase_name)
        wrapper = _build_wrapper(task)
    precompute = SyntheticPrecomputePhase(phase["pretraining"])

    rss_before = _peak_rss_bytes()
    started = time.perf_counter()
    partitions = precompute.build_partitions(wrapper.active_dataset)
    first_seconds = time.perf_counter() - started

    started = time.perf_counter()
    repeated = precompute.build_partitions(wrapper.active_dataset)
    cache_load_seconds = time.perf_counter() - started

    dataset = partitions["train"]
    batch_size = int(phase.get("batch_size", 64))
    rendered = 0
    maximum_tic_error = 0.0
    all_finite = True
    all_nonnegative = True
    all_molecule_targets_known = True
    started = time.perf_counter()
    for start in range(0, min(len(dataset), arguments.batches * batch_size), batch_size):
        rows = list(range(start, min(start + batch_size, len(dataset))))
        batch = dataset.collate_fn(rows)
        rendered += batch.batch_size
        spectra = batch.spectra
        maximum_tic_error = max(
            maximum_tic_error,
            float((spectra.sum(dim=1) - 1.0).abs().max().item()),
        )
        all_finite = all_finite and bool(spectra.isfinite().all())
        all_nonnegative = all_nonnegative and bool((spectra >= 0).all())
        all_molecule_targets_known = all_molecule_targets_known and bool(
            batch.targets.masks["molecule"].all()
        )
    render_seconds = time.perf_counter() - started

    artifact = repeated["train"].artifact
    print(
        json.dumps(
            {
                "artifact_key": artifact.artifact_key,
                "fingerprint": artifact.fingerprint,
                "task_file": str(task_path) if task_path is not None else None,
                "probe": probe,
                "population": dataset.population,
                "feature_count": artifact.feature_count,
                "prototype_count": artifact.prototype_count,
                "manifest_samples": len(dataset),
                "first_prepare_seconds": first_seconds,
                "matching_cache_load_seconds": cache_load_seconds,
                "rendered_samples": rendered,
                "render_seconds": render_seconds,
                "rendered_samples_per_second": rendered / max(render_seconds, 1e-12),
                "peak_rss_delta_bytes": _peak_rss_bytes() - rss_before,
                "numerical_validation": {
                    "all_finite": all_finite,
                    "all_nonnegative": all_nonnegative,
                    "maximum_tic_error": maximum_tic_error,
                    "all_molecule_targets_known": all_molecule_targets_known,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
