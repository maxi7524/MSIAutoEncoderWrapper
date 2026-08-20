"""Command-line interface for reproducible experiment runtime campaigns."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Sequence

import yaml

from ..utils.logger import get_custom_logger
from .backends import (
    build_sbatch_command,
    run_local_tasks,
    write_finalize_script,
    write_sbatch_script,
)
from .configuration import load_experiment_config
from .workflows import execute_task
from .output import is_completed_task, render_reports, task_fingerprint, update_manifest
from .planning import ExperimentPlan, build_plan, materialize_plan, resolve_plan
from .staging import cleanup_staging_directory, copy_verified, restore_results, stage_plan
from .planning import validate_preflight
from .progress import create_terminal_progress, format_duration, update_progress

logger = get_custom_logger(__name__)

# --------------------------------------------------
# Section: Parser
# --------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI and its internal task command."""
    parser = argparse.ArgumentParser(prog="msi-wrapper")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan", "run", "report"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path)
        if name in {"plan", "run"}:
            command.add_argument("--output", type=Path)
        if name == "run":
            command.add_argument("--dry-run", action="store_true")
            command.add_argument("--test-run", action="store_true")
        if name == "report":
            command.add_argument("--only")
    task = commands.add_parser("task", help=argparse.SUPPRESS)
    task.add_argument("task_file", type=Path)
    task_batch = commands.add_parser("task-batch", help=argparse.SUPPRESS)
    task_batch.add_argument("task_files", type=Path, nargs="+")
    finalize = commands.add_parser("finalize", help=argparse.SUPPRESS)
    finalize.add_argument("config", type=Path)
    finalize.add_argument("--staging-directory", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--execution-id", required=True)
    return parser

# --------------------------------------------------
# Section: Campaign directory
# --------------------------------------------------

def _set_experiment_directory(config: dict, override: Path | None) -> Path:
    """Return the persistent working directory for one experiment campaign.

    :param config: Loaded experiment configuration.
    :type config: dict
    :param override: Explicit CLI output directory, if supplied.
    :type override: pathlib.Path | None
    :return: Absolute directory containing the materialized plan and its outputs.
    :rtype: pathlib.Path
    """
    if override is not None:
        return override.resolve()
    configured = config.get("execution", {}).get("work_directory")
    if configured:
        return (Path(config["_config_directory"]) / configured).resolve()
    workspace = (
        config.get("task", {})
        .get("parameters", {})
        .get("factory_parameters", {})
        .get("project_path")
    )
    if workspace:
        return (
            Path(workspace)
            / "configs"
            / "execution"
            / config["experiment"]["name"]
        ).resolve()
    return (Path.cwd() / "workspace" / "executions" / config["experiment"]["name"]).resolve()


def _load_task(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        task = yaml.safe_load(stream)
    if not isinstance(task, dict):
        raise ValueError(f"Task file must contain a mapping: {path}")
    return task


def _task_label(task: dict) -> str:
    """Create a compact human-readable identity for one planned task."""
    grid = task.get("grid_parameters", {})
    architecture = grid.get("architectures", {})
    architecture_name = architecture.get("name", "unknown-architecture")
    bin_step = grid.get("binning_steps", "?")
    repetition = task.get("repetition", "?")
    return f"{task['task_id']} | {architecture_name} | bin={bin_step} | rep={repetition}"


def _has_complete_plan(directory: Path, plan: ExperimentPlan) -> bool:
    """Return whether a compatible materialized plan can be reused."""
    manifest_path = directory / "resolved-experiment.yaml"
    if not manifest_path.is_file():
        return False
    manifest = _load_task(manifest_path)
    expected_ids = {task.task_id for task in plan.tasks}
    materialized_ids = {task.get("task_id") for task in manifest.get("tasks", [])}
    return (
        manifest.get("runtime_schema_version") == 1
        and manifest.get("config_path") == plan.config_path
        and manifest.get("config_fingerprint") == plan.config_fingerprint
        and materialized_ids == expected_ids
        and all((directory / "tasks" / f"{task_id}.yaml").is_file() for task_id in expected_ids)
    )

# --------------------------------------------------
# Section: Execute plan element
# --------------------------------------------------

def _execute_task_file(path: Path, *, entrypoint: str | None = None, status_directory: str = "status") -> bool:
    """Execute one materialized task and persist its terminal status."""
    task = _load_task(path)
    materialized_fingerprint = task_fingerprint(task)
    if entrypoint is not None:
        task["entrypoint"] = entrypoint
    runtime_root = path.parent.parent
    task["runtime"] = {
        "task_fingerprint": materialized_fingerprint,
        "model_name": task["task_id"],
        "task_label": _task_label(task),
        "checkpoint_path": str(runtime_root / "checkpoints" / f"{task['task_id']}.pt"),
        "progress_path": str(runtime_root / status_directory / f"{task['task_id']}-progress.yaml"),
    }
    manifest = path.parent.parent / status_directory / f"{task['task_id']}.yaml"
    update_manifest(
        manifest,
        task["task_id"],
        {"status": "running", "task": task, "task_fingerprint": materialized_fingerprint},
    )
    try:
        result = execute_task(task)
    except Exception as error:
        update_manifest(manifest, task["task_id"], {"status": "failed", "error": str(error)})
        logger.error("Task '%s' failed.", task["task_id"], exc_info=True)
        return False
    update_manifest(
        manifest,
        task["task_id"],
        {"status": "completed", "result": result, "task_fingerprint": materialized_fingerprint},
    )
    return True


def _run_report_command(config: dict, args: argparse.Namespace) -> None:
    """Render reports without rebuilding or executing a campaign."""
    success = render_reports(
        config.get("reports", []),
        config_directory=Path(config["_config_directory"]),
        manifest_path=(Path(config["_config_directory"]) / "report-manifest.yaml"),
        only=args.only,
        continue_on_error=config.get("reporting", {}).get("continue_on_error", True),
    )
    if not success:
        raise SystemExit(1)


def _prepare_campaign(config: dict, args: argparse.Namespace) -> tuple[ExperimentPlan, Path]:
    """Validate, resolve or reuse, and materialize the campaign plan."""
    messages = validate_preflight(config)
    for message in messages:
        logger.info("%s", message)
    plan = build_plan(config)
    directory = _set_experiment_directory(config, args.output)
    if args.command == "run" and _has_complete_plan(directory, plan):
        logger.info("Reusing %s materialized tasks from %s.", len(plan.tasks), directory)
    else:
        plan = resolve_plan(plan, directory, config["task"].get("plan_entrypoint"))
        materialize_plan(plan, directory)
        logger.info("Materialized %s tasks in %s.", len(plan.tasks), directory)
    return plan, directory


def _run_test_tasks(directory: Path, plan: ExperimentPlan, config: dict) -> None:
    """Probe every unique grid cell without writing trained model artifacts."""
    entrypoint = config["task"].get("test_entrypoint")
    if not isinstance(entrypoint, str):
        raise ValueError("task.test_entrypoint is required for --test-run.")
    task_paths = {
        path.stem.replace("task_", ""): path
        for path in (directory / "tasks").glob("task_*.yaml")
    }
    tested_grid_ids: set[str] = set()
    all_succeeded = True
    for task in plan.tasks:
        if task.grid_id in tested_grid_ids:
            continue
        tested_grid_ids.add(task.grid_id)
        path = task_paths[task.task_id.replace("task_", "")]
        all_succeeded = _execute_task_file(
            path,
            entrypoint=entrypoint,
            status_directory="test-status",
        ) and all_succeeded
    if not all_succeeded:
        raise SystemExit(1)


def _run_local_campaign(
    task_paths: list[Path],
    *,
    directory: Path,
    max_parallel_runs: int,
    completed_before: int,
    total_tasks: int,
) -> None:
    """Execute local tasks while persisting campaign-level progress."""
    progress_path = directory / "progress" / "experiment.yaml"
    pending_total = len(task_paths)
    started = time.monotonic()
    if pending_total == 0:
        update_progress(
            progress_path,
            {"status": "completed", "completed": total_tasks, "total": total_tasks},
        )
        return
    update_progress(
        progress_path,
        {"status": "running", "completed": completed_before, "total": total_tasks},
    )
    experiment_progress = create_terminal_progress(
        total=total_tasks,
        initial=completed_before,
        description="Experiment",
        position=0,
    )
    if max_parallel_runs > 1:
        try:
            run_local_tasks(task_paths, max_parallel_runs)
            experiment_progress.update(pending_total)
            update_progress(
                progress_path,
                {"status": "completed", "completed": total_tasks, "total": total_tasks},
            )
        finally:
            experiment_progress.close()
        return
    try:
        for index, task_path in enumerate(task_paths, start=1):
            task = _load_task(task_path)
            experiment_progress.set_postfix_str(_task_label(task), refresh=False)
            update_progress(
                progress_path,
                {
                    "status": "running",
                    "completed": completed_before + index - 1,
                    "total": total_tasks,
                    "current_task": task["task_id"],
                    "current_task_label": _task_label(task),
                },
            )
            if not _execute_task_file(task_path):
                update_progress(
                    progress_path,
                    {
                        "status": "failed",
                        "completed": completed_before + index - 1,
                        "total": total_tasks,
                        "current_task": task["task_id"],
                    },
                )
                raise SystemExit(1)
            elapsed = time.monotonic() - started
            remaining = (elapsed / index) * (pending_total - index)
            completed = completed_before + index
            experiment_progress.update(1)
            update_progress(
                progress_path,
                {
                    "status": "completed" if completed == total_tasks else "running",
                    "completed": completed,
                    "total": total_tasks,
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": remaining,
                    "last_task": task_path.stem,
                },
            )
            logger.info(
                "Experiment progress: %s/%s tasks | elapsed %s | ETA %s.",
                completed,
                total_tasks,
                format_duration(elapsed),
                format_duration(remaining),
            )
    finally:
        experiment_progress.close()

# --------------------------------------------------
# Section: CLI helpers
# --------------------------------------------------

def _run_single_task_command(args: argparse.Namespace) -> None:
    """Execute the internal command for one materialized task.

    :param args: Parsed arguments containing ``task_file``.
    :type args: argparse.Namespace
    :raises SystemExit: If the task fails.
    """
    if not _execute_task_file(args.task_file):
        raise SystemExit(1)


def _run_task_batch_command(args: argparse.Namespace) -> None:
    """Execute internal worker tasks sequentially in one process.

    :param args: Parsed arguments containing ``task_files``.
    :type args: argparse.Namespace
    :raises SystemExit: If at least one task fails.
    """
    all_succeeded = True
    for task_file in args.task_files:
        all_succeeded = _execute_task_file(task_file) and all_succeeded
    if not all_succeeded:
        raise SystemExit(1)


def _run_finalize_command(args: argparse.Namespace) -> None:
    """Restore one staged Slurm execution and remove its staging directory.

    :param args: Parsed finalization command arguments.
    :type args: argparse.Namespace
    """
    config = load_experiment_config(args.config)
    staging = config.get("execution", {}).get("staging", {})
    try:
        persistent_status = args.output / "status"
        if persistent_status.exists():
            raise FileExistsError(f"Status destination already exists: {persistent_status}")
        staged_status = args.staging_directory / "plan" / "status"
        if staged_status.exists():
            copy_verified(staged_status, persistent_status)
        restore_results(
            args.staging_directory,
            config_directory=Path(config["_config_directory"]),
            results=staging.get("results", []),
        )
        if config.get("reports"):
            render_reports(
                config["reports"],
                config_directory=Path(config["_config_directory"]),
                manifest_path=args.output / "report-manifest.yaml",
                continue_on_error=config.get("reporting", {}).get("continue_on_error", True),
            )
    finally:
        cleanup_staging_directory(args.staging_directory, args.execution_id)


def _run_validate_command(config: dict) -> None:
    """Run preflight validation without resolving or materializing a plan.

    :param config: Loaded experiment configuration.
    :type config: dict
    """
    for message in validate_preflight(config):
        logger.info("%s", message)
    logger.info("Validation completed successfully.")


def _collect_pending_tasks(directory: Path) -> tuple[list[Path], int, int]:
    """Return pending task files and the campaign completion counters.

    :param directory: Materialized experiment directory.
    :type directory: pathlib.Path
    :return: Pending task paths, completed task count, and total task count.
    :rtype: tuple[list[pathlib.Path], int, int]
    """
    all_task_paths = sorted((directory / "tasks").glob("task_*.yaml"))
    pending_task_paths = []
    for task_path in all_task_paths:
        task = _load_task(task_path)
        manifest = directory / "status" / f"{task['task_id']}.yaml"
        if not is_completed_task(manifest, task):
            pending_task_paths.append(task_path)
    completed_before = len(all_task_paths) - len(pending_task_paths)
    logger.info(
        "Execution state: %s completed, %s pending.",
        completed_before,
        len(pending_task_paths),
    )
    return pending_task_paths, completed_before, len(all_task_paths)


def _render_campaign_reports(config: dict, plan: ExperimentPlan, directory: Path) -> None:
    """Render configured reports after successful local campaign execution.

    :param config: Loaded experiment configuration.
    :type config: dict
    :param plan: Resolved experiment plan.
    :type plan: ExperimentPlan
    :param directory: Materialized experiment directory.
    :type directory: pathlib.Path
    :raises SystemExit: If a report fails and reporting is configured as strict.
    """
    if not plan.reports:
        return
    success = render_reports(
        plan.reports,
        config_directory=Path(config["_config_directory"]),
        manifest_path=directory / "report-manifest.yaml",
        continue_on_error=config.get("reporting", {}).get("continue_on_error", True),
    )
    if not success:
        raise SystemExit(1)


def _run_local_backend(
    config: dict,
    plan: ExperimentPlan,
    directory: Path,
    task_paths: list[Path],
    completed_before: int,
    total_tasks: int,
) -> None:
    """Execute pending tasks locally and then render local reports.

    :param config: Loaded experiment configuration.
    :type config: dict
    :param plan: Resolved experiment plan.
    :type plan: ExperimentPlan
    :param directory: Materialized experiment directory.
    :type directory: pathlib.Path
    :param task_paths: Pending materialized task files.
    :type task_paths: list[pathlib.Path]
    :param completed_before: Number of already completed tasks.
    :type completed_before: int
    :param total_tasks: Total number of campaign tasks.
    :type total_tasks: int
    """
    max_parallel_runs = int(plan.execution.get("max_parallel_runs", 1))
    _run_local_campaign(
        task_paths,
        directory=directory,
        max_parallel_runs=max_parallel_runs,
        completed_before=completed_before,
        total_tasks=total_tasks,
    )
    _render_campaign_reports(config, plan, directory)


def _run_slurm_backend(config: dict, plan: ExperimentPlan, directory: Path, task_paths: list[Path]) -> None:
    """Stage, submit, and arrange finalization for one Slurm task array.

    :param config: Loaded experiment configuration.
    :type config: dict
    :param plan: Resolved experiment plan.
    :type plan: ExperimentPlan
    :param directory: Persistent materialized experiment directory.
    :type directory: pathlib.Path
    :param task_paths: Pending materialized task files.
    :type task_paths: list[pathlib.Path]
    :raises ValueError: If staging is disabled for a Slurm execution.
    """
    staging = plan.execution.get("staging", {})
    if not staging.get("enabled", False):
        raise ValueError("Slurm execution requires execution.staging.enabled=true")
    execution_id = f"{plan.experiment_name}-{plan.tasks[0].task_id}"
    staged = stage_plan(
        directory,
        config_directory=Path(config["_config_directory"]),
        staging=staging,
        execution_id=execution_id,
    )
    array_submitted = False
    try:
        staged_plan = staged / "plan"
        script = write_sbatch_script(staged_plan, len(task_paths), plan.execution.get("slurm", {}))
        result = subprocess.run(
            build_sbatch_command(script, parsable=True),
            check=True,
            capture_output=True,
            text=True,
        )
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Cannot parse sbatch job identifier: {result.stdout!r}")
        array_submitted = True
        finalize_script = write_finalize_script(
            staged_plan,
            job_id=job_id,
            config_path=Path(config["_config_path"]),
            persistent_directory=directory,
            staging_directory=staged,
            execution_id=execution_id,
        )
        subprocess.run(build_sbatch_command(finalize_script), check=True)
    except Exception:
        if not array_submitted:
            cleanup_staging_directory(staged, execution_id)
        raise
    logger.info("Submitted Slurm array %s and its dependent finalizer.", job_id)


def _run_campaign_command(config: dict, args: argparse.Namespace) -> None:
    """Plan, test, or execute one public experiment campaign command.

    :param config: Loaded experiment configuration.
    :type config: dict
    :param args: Parsed ``plan`` or ``run`` command arguments.
    :type args: argparse.Namespace
    """
    plan, directory = _prepare_campaign(config, args)
    if args.command == "plan" or args.dry_run:
        return
    if args.test_run:
        _run_test_tasks(directory, plan, config)
        return

    task_paths, completed_before, total_tasks = _collect_pending_tasks(directory)
    backend = plan.execution.get("backend", "local")
    if backend == "local":
        _run_local_backend(
            config,
            plan,
            directory,
            task_paths,
            completed_before,
            total_tasks,
        )
        return
    _run_slurm_backend(config, plan, directory, task_paths)


def _run_internal_command(args: argparse.Namespace) -> bool:
    """Execute one backend-only command and return whether it was handled.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: Whether ``args.command`` was an internal backend command.
    :rtype: bool
    """
    handlers = {
        "task": _run_single_task_command,
        "task-batch": _run_task_batch_command,
        "finalize": _run_finalize_command,
    }
    handler = handlers.get(args.command)
    if handler is None:
        return False
    handler(args)
    return True


# --------------------------------------------------
# Section: Main runner
# --------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch one public campaign command or one internal backend command.

    :param argv: Command-line arguments excluding the executable name.
    :type argv: typing.Sequence[str] | None
    """
    args = build_parser().parse_args(argv)

    # Backend-only commands
    if _run_internal_command(args):
        return

    # Public configuration commands
    config = load_experiment_config(args.config)
    if args.command == "report":
        _run_report_command(config, args)
        return
    if args.command == "validate":
        _run_validate_command(config)
        return

    # Campaign commands: `plan` and `run`
    _run_campaign_command(config, args)


if __name__ == "__main__":
    main()
