"""Filesystem-derived planning and materialization of external MSI datasets."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from tqdm.auto import tqdm

from ..imzml import PyImzMLReader
from ..utils.exceptions import DownloadLimitError, raise_validation_error
from ..utils.logger import get_custom_logger
from ..sources.base import DatasetSource
from ..sources.profiles import RotatingDatasetSource, read_source_profiles
from ..validators import validate_imzml_pair, validate_selection
from .annotation_csv import (
    has_complete_annotation_csv,
    write_annotation_csv_pair,
)


logger = get_custom_logger(__name__)


def create_materialization_manifest(
    *,
    selection_path: Path | str,
    datasets_dir: Path | str,
    source_name: str,
    annotation_options: Optional[Mapping[str, Any]] = None,
    dataset_ids: Optional[Sequence[str]] = None,
    manifest_path: Optional[Path | str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Scan local datasets and persist the complete materialization plan.

    :return: Manifest whose ``planned_actions`` are the sole execution plan.
    :rtype: Dict[str, Any]
    """
    selection_file = Path(selection_path).resolve()
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    validate_selection(selection)
    if selection.get("source") != source_name:
        raise ValueError(
            f"Selection source '{selection.get('source')}' does not match "
            f"requested source '{source_name}'."
        )
    resolved_options = _resolve_annotation_options(selection, annotation_options)
    selected = set(dataset_ids or ())
    records = [
        record
        for record in selection.get("datasets", [])
        if not selected or str(record["dataset_id"]) in selected
    ]
    dataset_root = Path(datasets_dir).resolve()
    entries: List[Dict[str, Any]] = []
    for record in records:
        dataset_id = str(record["dataset_id"])
        directory = dataset_root / dataset_id
        data_present = _has_complete_pair(directory, dataset_id)
        annotations_present = has_complete_annotation_csv(directory, dataset_id)
        actions = []
        if not data_present:
            actions.append("download_dataset")
        if not annotations_present:
            actions.append("download_annotations")
        entries.append(
            {
                "dataset_id": dataset_id,
                "name": str(record.get("name", dataset_id)),
                "directory": str(directory),
                "directory_relative_to_invocation": _relative_display(directory),
                "data_present": data_present,
                "annotations_present": annotations_present,
                "planned_actions": actions,
                "execution": {
                    "dataset": "reused" if data_present else "pending",
                    "annotations": "reused" if annotations_present else "pending",
                },
                "final_state": {
                    "data_present": data_present,
                    "annotations_present": annotations_present,
                },
            }
        )
    target = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else selection_file.parent / "materialization.json"
    )
    manifest: Dict[str, Any] = {
        "schema_version": 2,
        "dry_run": bool(dry_run),
        "status": "planned",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "invocation_directory": str(Path.cwd().resolve()),
        "workspace_datasets_path": str(dataset_root),
        "workspace_datasets_path_relative_to_invocation": _relative_display(dataset_root),
        "selection_path": str(selection_file),
        "manifest_path": str(target),
        "source": source_name,
        "annotation_options": resolved_options,
        "datasets": entries,
        "summary": _manifest_summary(entries),
    }
    _write_json_atomic(target, manifest)
    return manifest


def format_materialization_plan(manifest: Mapping[str, Any]) -> str:
    """Return a compact table containing local state, actions, and destinations."""
    headers = ("Dataset ID", "Data", "Annotations", "Actions", "Destination")
    rows = []
    for entry in manifest.get("datasets", []):
        rows.append(
            (
                str(entry["dataset_id"]),
                "yes" if entry["data_present"] else "no",
                "yes" if entry["annotations_present"] else "no",
                ", ".join(entry["planned_actions"]) or "none",
                str(entry["directory_relative_to_invocation"]),
            )
        )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    line = "  ".join(headers[index].ljust(widths[index]) for index in range(len(headers)))
    divider = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
        for row in rows
    ]
    summary = manifest.get("summary", {})
    plan_summary = (
        "Plan: "
        f"{summary.get('selected', len(rows))} selected; "
        f"{summary.get('download_dataset', 0)} dataset downloads remaining; "
        f"{summary.get('download_annotations', 0)} annotation downloads remaining."
    )
    return "\n".join([line, divider, *body, plan_summary])


def materialize_selection(
    *,
    source: DatasetSource,
    manifest: Dict[str, Any],
    profiles_path: Optional[Path | str] = None,
    source_factory: Optional[
        Callable[[str, Mapping[str, str]], DatasetSource]
    ] = None,
) -> List[Path]:
    """Execute one freshly generated materialization manifest.

    :param source: Adapter matching the selection source.
    :type source: DatasetSource
    :param manifest: Plan returned by :func:`create_materialization_manifest`.
        It is updated atomically after every dataset-level state transition.
    :type manifest: Dict[str, Any]
    :param profiles_path: Optional CSV with a mandatory ``key`` column.
    :type profiles_path: pathlib.Path | str | None
    :param source_factory: Factory used to create a source after rotating to a
        new key. ``source`` must use the first CSV key.
    :type source_factory: Callable[[str, Mapping[str, str]], DatasetSource] | None
    :return: Materialized dataset directories. Annotation records are not
        imported into SQLite until cohort composition.
    :rtype: List[pathlib.Path]
    :raises ValueError: If the manifest belongs to another source.

    The file phase stops at the first provider quota error. The annotation
    phase then scans every manifest entry and processes every complete local
    imzML/ibd pair, including pairs located after the failed entry. Annotation
    failures are recorded independently and never discard downloaded files.
    """
    if manifest.get("source") != source.source_name:
        raise ValueError(
            f"Manifest source '{manifest.get('source')}' does not match "
            f"adapter '{source.source_name}'."
        )
    entries = manifest.get("datasets")
    if not isinstance(entries, list):
        raise_validation_error(
            "DatasetMaterialization", "Manifest datasets must be a list."
        )
    resolved_annotation_options = dict(manifest.get("annotation_options", {}))
    target_manifest = Path(str(manifest["manifest_path"]))

    # Provider call routing
    ## Rotate credentials only after an explicit binary-download quota error.
    rotating_source = None
    if profiles_path is not None:
        if source_factory is None:
            raise_validation_error(
                "DatasetMaterialization",
                "source_factory is required when profiles_path is provided.",
            )
        rotating_source = RotatingDatasetSource(
            read_source_profiles(profiles_path),
            source_factory,
            initial_source=source,
        )
    call = (
        rotating_source.call
        if rotating_source is not None
        else lambda method_name, *args, **kwargs: getattr(source, method_name)(
            *args, **kwargs
        )
    )
    materialized: List[Path] = []
    annotation_counts: Dict[str, int] = {}
    file_limit_reached = False
    manifest.update(
        {
            "dry_run": False,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "annotation_counts": annotation_counts,
            "file_limit_reached": False,
            "exhausted_profile_count": 0,
        }
    )
    _persist_execution_manifest(manifest, target_manifest, rotating_source)

    # Dataset file materialization
    ## Execute the immutable plan until the provider refuses binary downloads.
    ## Existing complete pairs were marked reusable by the fresh filesystem scan.
    downloads_remaining = sum(
        "download_dataset" in entry["planned_actions"] for entry in entries
    )
    file_progress = tqdm(
        total=len(entries),
        desc="Dataset files",
        unit="dataset",
        dynamic_ncols=True,
    )
    try:
        for position, entry in enumerate(entries):
            dataset_id = str(entry["dataset_id"])
            destination = Path(str(entry["directory"]))
            file_progress.set_postfix(
                current=dataset_id,
                downloads_remaining=downloads_remaining,
                profiles_remaining=(
                    rotating_source.remaining_profile_count
                    if rotating_source is not None
                    else "unknown"
                ),
            )
            logger.info("Materializing dataset %s", dataset_id)
            if "download_dataset" not in entry["planned_actions"]:
                logger.info(
                    "Reusing downloaded dataset %s from %s",
                    dataset_id,
                    destination,
                )
                entry["execution"]["dataset"] = "reused"
            else:
                entry["execution"]["dataset"] = "downloading"
                _persist_execution_manifest(
                    manifest, target_manifest, rotating_source
                )
                try:
                    call("download_dataset", dataset_id, destination)
                    validate_imzml_pair(destination, dataset_id)
                    entry["execution"]["dataset"] = "downloaded"
                    downloads_remaining -= 1
                except DownloadLimitError:
                    entry["execution"]["dataset"] = "download_limit"
                    file_limit_reached = True
                    manifest["file_limit_reached"] = True
                    for pending in entries[position + 1 :]:
                        if "download_dataset" in pending["planned_actions"]:
                            pending["execution"]["dataset"] = (
                                "not_executed_after_limit"
                            )
                    _persist_execution_manifest(
                        manifest, target_manifest, rotating_source
                    )
                    logger.warning(
                        "Stopping the imzML phase at dataset %s with %s dataset "
                        "downloads remaining; continuing with annotations for "
                        "complete local pairs",
                        dataset_id,
                        downloads_remaining,
                    )
                    break
            _persist_execution_manifest(manifest, target_manifest, rotating_source)
            file_progress.update(1)
    finally:
        file_progress.set_postfix(downloads_remaining=downloads_remaining)
        file_progress.close()

    # Canonical annotation materialization
    ## Rescan all entries after the file phase. This deliberately finds cached
    ## pairs after a quota failure and keeps annotation retrieval independent.
    ## CSVs remain the stable reader boundary; SQLite is created only by compose.
    for entry in entries:
        dataset_id = str(entry["dataset_id"])
        destination = Path(str(entry["directory"]))
        if not _has_complete_pair(destination, dataset_id):
            entry["execution"]["annotations"] = "not_executed_without_data"
            _persist_execution_manifest(manifest, target_manifest, rotating_source)
            continue
        validate_imzml_pair(destination, dataset_id)
        materialized.append(destination)
        if entry["execution"]["dataset"] in {
            "pending",
            "not_executed_after_limit",
        }:
            entry["execution"]["dataset"] = "reused_after_file_phase"
        if "download_annotations" not in entry["planned_actions"]:
            entry["execution"]["annotations"] = "reused"
            logger.info("Reusing annotations for dataset %s", dataset_id)
            _persist_execution_manifest(manifest, target_manifest, rotating_source)
            continue
        reader = PyImzMLReader(validate_imzml_pair(destination, dataset_id))
        entry["execution"]["annotations"] = "downloading"
        _persist_execution_manifest(manifest, target_manifest, rotating_source)
        try:
            provider_annotations = call(
                "get_annotations", dataset_id, resolved_annotation_options
            )
            write_annotation_csv_pair(
                directory=destination,
                dataset_id=dataset_id,
                dataset_name=str(entry.get("name", dataset_id)),
                annotations=provider_annotations,
                reader=reader,
            )
            annotation_count = len(provider_annotations)
        except Exception as error:
            entry["execution"]["annotations"] = (
                f"failed: {type(error).__name__}: {error}"
            )
            logger.error(
                "Annotation materialization failed for dataset %s: %s",
                dataset_id,
                error,
            )
            _persist_execution_manifest(manifest, target_manifest, rotating_source)
            continue
        annotation_counts[dataset_id] = annotation_count
        entry["execution"]["annotations"] = "downloaded"
        _persist_execution_manifest(manifest, target_manifest, rotating_source)

    # Final execution state
    ## Completion may be partial: quota and annotation failures remain explicit.
    annotation_failures = sum(
        str(entry["execution"]["annotations"]).startswith("failed:")
        for entry in entries
    )
    manifest.update(
        {
            "status": (
                "completed_with_gaps"
                if file_limit_reached or annotation_failures
                else "completed"
            ),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _persist_execution_manifest(manifest, target_manifest, rotating_source)
    logger.info(
        "Materialization report: files downloaded=%s, reused=%s, annotation CSVs "
        "downloaded=%s, reused=%s, failed=%s, remaining files=%s; manifest=%s",
        sum(entry["execution"]["dataset"] == "downloaded" for entry in entries),
        sum(
            str(entry["execution"]["dataset"]).startswith("reused")
            for entry in entries
        ),
        sum(entry["execution"]["annotations"] == "downloaded" for entry in entries),
        sum(entry["execution"]["annotations"] == "reused" for entry in entries),
        annotation_failures,
        sum(not entry["final_state"]["data_present"] for entry in entries),
        target_manifest,
    )
    return materialized


def _persist_execution_manifest(
    manifest: Dict[str, Any],
    target: Path,
    rotating_source: Optional[RotatingDatasetSource],
) -> None:
    """Refresh filesystem-derived state and atomically persist execution status.

    :param manifest: Mutable in-memory execution manifest.
    :type manifest: Dict[str, Any]
    :param target: Manifest JSON path.
    :type target: pathlib.Path
    :param rotating_source: Optional quota-aware provider proxy.
    :type rotating_source: RotatingDatasetSource | None

    This helper is intentionally called after every dataset-level transition.
    A terminated process therefore leaves a useful report, while the next run
    still replaces the plan from a new filesystem scan.
    """
    for entry in manifest["datasets"]:
        dataset_id = str(entry["dataset_id"])
        directory = Path(str(entry["directory"]))
        entry["final_state"] = {
            "data_present": _has_complete_pair(directory, dataset_id),
            "annotations_present": has_complete_annotation_csv(directory, dataset_id),
        }
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["exhausted_profile_count"] = (
        rotating_source.exhausted_profile_count if rotating_source is not None else 0
    )
    manifest["final_summary"] = _final_manifest_summary(manifest["datasets"])
    _write_json_atomic(target, manifest)


def _manifest_summary(entries: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Summarize a freshly scanned materialization plan."""
    return {
        "selected": len(entries),
        "data_present": sum(bool(entry["data_present"]) for entry in entries),
        "annotations_present": sum(
            bool(entry["annotations_present"]) for entry in entries
        ),
        "download_dataset": sum(
            "download_dataset" in entry["planned_actions"] for entry in entries
        ),
        "download_annotations": sum(
            "download_annotations" in entry["planned_actions"] for entry in entries
        ),
    }


def _final_manifest_summary(entries: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Summarize final filesystem state after executing a plan."""
    return {
        "data_present": sum(
            bool(entry["final_state"]["data_present"]) for entry in entries
        ),
        "annotations_present": sum(
            bool(entry["final_state"]["annotations_present"]) for entry in entries
        ),
    }


def _relative_display(path: Path) -> str:
    """Return an invocation-relative path when possible, otherwise absolute."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _has_complete_pair(directory: Path, dataset_id: str) -> bool:
    """Return whether a non-empty canonical imzML/ibd pair exists locally."""
    imzml_path = directory / f"{dataset_id}.imzML"
    ibd_path = directory / f"{dataset_id}.ibd"
    return (
        imzml_path.is_file()
        and imzml_path.stat().st_size > 0
        and ibd_path.is_file()
        and ibd_path.stat().st_size > 0
    )


def _resolve_annotation_options(
    selection: Mapping[str, Any],
    annotation_options: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Resolve one annotation FDR across discovery and materialization.

    :param selection: Validated query selection payload.
    :type selection: Mapping[str, Any]
    :param annotation_options: Explicit provider annotation options.
    :type annotation_options: Mapping[str, Any] | None
    :return: Retrieval options containing the selection ``annotation_fdr``.
    :rtype: Dict[str, Any]
    :raises ValidationError: If a legacy key is used or discovery and download
        request different annotation FDR thresholds.
    """
    resolved = dict(annotation_options or {})
    if "fdr" in resolved:
        raise_validation_error(
            "AnnotationOptions",
            "Use 'annotation_fdr' instead of the ambiguous legacy key 'fdr'.",
        )
    selection_filters = selection.get("filters", {})
    selection_fdr = (
        selection_filters.get("annotation_fdr")
        if isinstance(selection_filters, Mapping)
        else None
    )
    requested_fdr = resolved.get("annotation_fdr")
    if selection_fdr is not None and requested_fdr is not None:
        if float(selection_fdr) != float(requested_fdr):
            raise_validation_error(
                "AnnotationOptions",
                "The annotation_fdr used for download must match the value "
                f"stored in the selection ({selection_fdr}).",
            )
    if selection_fdr is not None:
        resolved["annotation_fdr"] = float(selection_fdr)
    return resolved


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a reproducibility manifest without exposing provider secrets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
