"""Verified staging and cleanup for shared node RAM filesystems."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import yaml


MARKER_NAME = ".msi-wrapper-staging"


def file_checksum(path: Path) -> str:
    """Calculate a streaming SHA-256 checksum."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_verified(source: Path, destination: Path) -> None:
    """Copy a file or tree and verify every resulting file checksum."""
    if source.is_dir():
        shutil.copytree(source, destination)
        source_files = [path for path in source.rglob("*") if path.is_file()]
        for source_file in source_files:
            relative = source_file.relative_to(source)
            if file_checksum(source_file) != file_checksum(destination / relative):
                raise OSError(f"Staging checksum mismatch: {relative}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if file_checksum(source) != file_checksum(destination):
            raise OSError(f"Staging checksum mismatch: {source}")


def create_staging_directory(root: Path, execution_id: str) -> Path:
    """Create an owned staging directory below an explicit shared-RAM root."""
    root = root.resolve()
    if root == Path(root.anchor) or not execution_id or Path(execution_id).name != execution_id:
        raise ValueError("Unsafe staging root or execution identifier")
    directory = root / execution_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / MARKER_NAME).write_text(execution_id, encoding="utf-8")
    return directory


def cleanup_staging_directory(directory: Path, execution_id: str) -> None:
    """Remove only a staging directory carrying the matching ownership marker."""
    directory = directory.resolve()
    marker = directory / MARKER_NAME
    if not marker.is_file() or marker.read_text(encoding="utf-8") != execution_id:
        raise ValueError(f"Refusing to clean unowned staging directory: {directory}")
    if directory.name != execution_id or directory == Path(directory.anchor):
        raise ValueError(f"Refusing to clean unsafe staging directory: {directory}")
    shutil.rmtree(directory)


def _replace_paths(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _replace_paths(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_paths(item, replacements) for item in value]
    if isinstance(value, str):
        for source, destination in replacements.items():
            if value == source or value.startswith(source + "/"):
                return destination + value[len(source):]
    return value


def stage_plan(
    plan_directory: Path,
    *,
    config_directory: Path,
    staging: dict[str, Any],
    execution_id: str,
) -> Path:
    """Stage a plan and declared inputs, then rewrite matching task paths."""
    directory = create_staging_directory(Path(staging["root"]), execution_id)
    copy_verified(plan_directory, directory / "plan")
    replacements: dict[str, str] = {}
    for index, value in enumerate(staging.get("inputs", [])):
        definition = {"source": value} if isinstance(value, str) else value
        source = (config_directory / definition["source"]).resolve()
        name = definition.get("name", f"input-{index}-{source.name}")
        destination = directory / "inputs" / name
        copy_verified(source, destination)
        replacements[str(source)] = str(destination)
    if replacements:
        for task_path in (directory / "plan" / "tasks").glob("task_*.yaml"):
            with task_path.open(encoding="utf-8") as stream:
                task = yaml.safe_load(stream)
            with task_path.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(_replace_paths(task, replacements), stream, sort_keys=False)
    return directory


def restore_results(
    staging_directory: Path,
    *,
    config_directory: Path,
    results: list[Any],
) -> None:
    """Copy declared results to persistent storage and verify every file."""
    failures: list[str] = []
    for value in results:
        try:
            if (
                not isinstance(value, dict)
                or "source" not in value
                or "destination" not in value
            ):
                raise ValueError("Every staging result requires source and destination")
            source = (staging_directory / value["source"]).resolve()
            destination = (config_directory / value["destination"]).resolve()
            if destination.exists():
                raise FileExistsError(
                    f"Persistent result destination already exists: {destination}"
                )
            copy_verified(source, destination)
        except (OSError, ValueError) as error:
            failures.append(str(error))
    if failures:
        raise OSError("Failed to restore one or more results: " + "; ".join(failures))
