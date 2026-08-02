"""Command-line interface for reproducible experiment execution."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import asdict
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
from .entrypoints import execute_task
from .manifests import update_manifest
from .planning import build_plan, materialize_plan
from .reporting import render_reports
from .staging import cleanup_staging_directory, copy_verified, restore_results, stage_plan
from .validation import validate_preflight

logger = get_custom_logger(__name__)


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
        if name == "report":
            command.add_argument("--only")
    task = commands.add_parser("task", help=argparse.SUPPRESS)
    task.add_argument("task_file", type=Path)
    finalize = commands.add_parser("finalize", help=argparse.SUPPRESS)
    finalize.add_argument("config", type=Path)
    finalize.add_argument("--staging-directory", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--execution-id", required=True)
    return parser


def _plan_directory(config: dict, override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    configured = config.get("execution", {}).get("work_directory")
    if configured:
        return (Path(config["_config_directory"]) / configured).resolve()
    return (Path.cwd() / "workspace" / "executions" / config["experiment"]["name"]).resolve()


def _load_task(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        task = yaml.safe_load(stream)
    if not isinstance(task, dict):
        raise ValueError(f"Task file must contain a mapping: {path}")
    return task


def main(argv: Sequence[str] | None = None) -> None:
    """Validate, plan, execute or report an experiment campaign."""
    args = build_parser().parse_args(argv)
    if args.command == "task":
        task = _load_task(args.task_file)
        manifest = args.task_file.parent.parent / "status" / f"{task['task_id']}.yaml"
        update_manifest(manifest, task["task_id"], {"status": "running", "task": task})
        try:
            result = execute_task(task)
        except Exception as error:
            update_manifest(manifest, task["task_id"], {"status": "failed", "error": str(error)})
            raise
        update_manifest(manifest, task["task_id"], {"status": "completed", "result": result})
        return

    if args.command == "finalize":
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
        return

    config = load_experiment_config(args.config)
    if args.command == "report":
        success = render_reports(
            config.get("reports", []),
            config_directory=Path(config["_config_directory"]),
            manifest_path=(Path(config["_config_directory"]) / "report-manifest.yaml"),
            only=args.only,
            continue_on_error=config.get("reporting", {}).get("continue_on_error", True),
        )
        if not success:
            raise SystemExit(1)
        return
    messages = validate_preflight(config)
    for message in messages:
        logger.info("%s", message)
    if args.command == "validate":
        return

    plan = build_plan(config)
    directory = _plan_directory(config, args.output)
    materialize_plan(plan, directory)
    logger.info("Materialized %s tasks in %s.", len(plan.tasks), directory)
    if args.command == "plan" or args.dry_run:
        return
    task_paths = sorted((directory / "tasks").glob("task_*.yaml"))
    backend = plan.execution.get("backend", "local")
    if backend == "local":
        run_local_tasks(task_paths, int(plan.execution.get("max_parallel_runs", 1)))
        if plan.reports:
            success = render_reports(
                plan.reports,
                config_directory=Path(config["_config_directory"]),
                manifest_path=directory / "report-manifest.yaml",
                continue_on_error=config.get("reporting", {}).get("continue_on_error", True),
            )
            if not success:
                raise SystemExit(1)
    else:
        staging = plan.execution.get("staging", {})
        if not staging.get("enabled", False):
            raise ValueError("Slurm execution requires execution.staging.enabled=true")
        execution_id = f"{plan.experiment_name}-{plan.tasks[0].seed}"
        staged = stage_plan(
            directory,
            config_directory=Path(config["_config_directory"]),
            staging=staging,
            execution_id=execution_id,
        )
        array_submitted = False
        try:
            staged_plan = staged / "plan"
            script = write_sbatch_script(
                staged_plan,
                len(task_paths),
                plan.execution.get("slurm", {}),
            )
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


if __name__ == "__main__":
    main()
