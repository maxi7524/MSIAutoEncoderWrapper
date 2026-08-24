"""Tests for isolated Slurm array and finalizer scripts."""

from __future__ import annotations

from pathlib import Path

from msi_autoencoder_wrapper.runtime.backends.slurm import (
    build_sbatch_command,
    write_finalize_script,
    write_sbatch_script,
)


def test_slurm_array_is_bounded_and_uses_materialized_tasks(tmp_path: Path) -> None:
    """The scheduler script maps array indices to stable task descriptors."""
    (tmp_path / "tasks").mkdir()
    script = write_sbatch_script(
        tmp_path,
        5,
        {
            "array_parallelism": 2,
            "partition": "gpu",
            "qos": "student_gpu",
            "nodelist": "gpu-node-1",
            "gpus_per_task": 1,
        },
    )
    content = script.read_text(encoding="utf-8")

    assert "#SBATCH --array=0-4%2" in content
    assert "#SBATCH --partition=gpu" in content
    assert "#SBATCH --qos=student_gpu" in content
    assert "#SBATCH --nodelist=gpu-node-1" in content
    assert "task_%06d.yaml" in content
    assert build_sbatch_command(script, parsable=True)[:2] == ["sbatch", "--parsable"]


def test_finalizer_runs_after_any_array_outcome(tmp_path: Path) -> None:
    """Copy-back and cleanup are queued even when an array task fails."""
    script = write_finalize_script(
        tmp_path,
        job_id="1234",
        config_path=tmp_path / "experiment.yaml",
        persistent_directory=tmp_path / "persistent",
        staging_directory=tmp_path / "ram" / "execution-1",
        execution_id="execution-1",
    )
    content = script.read_text(encoding="utf-8")

    assert "#SBATCH --dependency=afterany:1234" in content
    assert " execution_id" not in content
    assert "--execution-id execution-1" in content
