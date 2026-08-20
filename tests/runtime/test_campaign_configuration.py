"""Tests for the repository architecture and binning campaign."""

from __future__ import annotations

from pathlib import Path

from msi_autoencoder_wrapper.runtime import build_plan, load_experiment_config
from msi_autoencoder_wrapper.runtime.cli import _set_experiment_directory


def test_architecture_binning_campaign_is_paired_across_five_seeds() -> None:
    """The root campaign expands to 60 paired architecture/binning tasks."""
    repository = Path(__file__).resolve().parents[2]
    config = load_experiment_config(
        repository
        / "assets"
        / "experiments"
        / "13_08_26_architecture_and_binning"
        / "architecture_binning_experiment.yaml"
    )

    plan = build_plan(config)

    workspace = repository / "data" / "kidney_workspace"
    factory_parameters = config["task"]["parameters"]["factory_parameters"]
    assert factory_parameters["project_path"] == str(workspace.resolve())
    assert factory_parameters["project_path_anchor"] == "repository"
    assert _set_experiment_directory(config, None) == (
        workspace / "configs" / "execution" / "kidney-architecture-binning"
    ).resolve()
    assert len(plan.tasks) == 60
    assert len(
        {task.reproducibility["derived_run_seeds"]["model_initialization"] for task in plan.tasks}
    ) == 5
    bin_steps = {
        task.parameters["factory_parameters"]["binning"]["parameters"]["bin_step"]
        for task in plan.tasks
    }
    variant_names = {
        task.parameters["factory_parameters"]["variant"]["name"]
        for task in plan.tasks
    }
    assert bin_steps == {0.45, 0.5, 0.55, 1.0}
    inverse_definitions = {
        task.parameters["factory_parameters"]["inverse_binning"]["strategy"]
        for task in plan.tasks
    }
    assert inverse_definitions == {"PassthroughInverseBinner"}
    assert variant_names == {
        "mlp-ae-512-latent-10",
        "mlp-ae-512-256-latent-10",
        "conv1d-ae-32-16-8-latent-10",
    }
    for repetition in range(5):
        repetition_tasks = [task for task in plan.tasks if task.repetition == repetition]
        assert len(repetition_tasks) == 12
        assert len(
            {
                task.reproducibility["derived_run_seeds"]["model_initialization"]
                for task in repetition_tasks
            }
        ) == 1

    # Five repetitions must not reproduce one identical initialization five times.
    repetition_seeds = [
        plan.tasks[repetition].reproducibility["derived_run_seeds"]["model_initialization"]
        for repetition in range(5)
    ]
    assert len(set(repetition_seeds)) == 5
