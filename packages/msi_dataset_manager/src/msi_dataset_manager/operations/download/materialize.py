"""Execute an immutable dataset materialization manifest."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from tqdm.auto import tqdm

from ...sources.base import AnnotationDatasetSource, DatasetSource
from ...sources.profiles import RotatingDatasetSource, read_source_profiles
from ...sources.source_manager import DatasetSourceManager
from ...utils.exceptions import DownloadLimitError, raise_validation_error
from ...utils.logger import get_custom_logger
from ...validators import validate_imzml_pair
from .manifest import final_manifest_summary, has_complete_pair, write_json_atomic


logger = get_custom_logger(__name__)

# --------------------------------------------------
# Section: Export function 
# --------------------------------------------------

def download_from_manifest(
    *,
    source: DatasetSource,
    manifest: Dict[str, Any],
    metaspace_api_keys_path: Optional[Path | str] = None,
    source_factory: Optional[
        Callable[[str, Mapping[str, str]], DatasetSource]
    ] = None,
) -> List[Path]:
    """Execute one previously created materialization manifest.

    :param source: Adapter matching the manifest source.
    :param manifest: Mutable plan created by ``create_download_manifest``.
    :param metaspace_api_keys_path: Optional additional METASPACE API keys.
    :param source_factory: Factory used after a key has exhausted its quota.
    :return: Directories containing complete local imzML/ibd pairs.
    :raises ValueError: If the manifest source differs from the adapter source.

    Binary downloads stop at a quota boundary. Annotation retrieval still scans
    every manifest entry with a complete local imzML/ibd pair.
    """
    if manifest.get("source") != source.source_name:
        raise ValueError(
            f"Manifest source '{manifest.get('source')}' does not match "
            f"adapter '{source.source_name}'."
        )
    if not isinstance(source, AnnotationDatasetSource):
        raise TypeError(
            f"Dataset source '{source.source_name}' does not implement annotations."
        )
    entries = manifest.get("datasets")
    if not isinstance(entries, list):
        raise_validation_error("DatasetMaterialization", "Manifest datasets must be a list.")
    target_manifest = Path(str(manifest["manifest_path"]))
    annotation_options = dict(manifest.get("annotation_options", {}))
    call, rotating_source, annotation_counts = _prepare_execution(
        source=source,
        manifest=manifest,
        target_manifest=target_manifest,
        metaspace_api_keys_path=metaspace_api_keys_path,
        source_factory=source_factory,
    )
    file_limit_reached = _download_dataset_files(
        call=call,
        entries=entries,
        manifest=manifest,
        target_manifest=target_manifest,
        rotating_source=rotating_source,
    )
    materialized = _download_annotation_files(
        call=call,
        entries=entries,
        manifest=manifest,
        target_manifest=target_manifest,
        rotating_source=rotating_source,
        annotation_options=annotation_options,
        annotation_counts=annotation_counts,
    )
    _finalize_download(
        manifest=manifest,
        entries=entries,
        target_manifest=target_manifest,
        rotating_source=rotating_source,
        file_limit_reached=file_limit_reached,
    )
    return materialized

# --------------------------------------------------
# Section: Main functionality 
# --------------------------------------------------

def _prepare_execution(
    *,
    source: DatasetSource,
    manifest: Dict[str, Any],
    target_manifest: Path,
    metaspace_api_keys_path: Optional[Path | str],
    source_factory: Optional[Callable[[str, Mapping[str, str]], DatasetSource]],
) -> tuple[Callable[..., Any], Optional[RotatingDatasetSource], Dict[str, int]]:
    """Prepare provider routing and persist the initial running state."""
    rotating_source = None
    if metaspace_api_keys_path is not None:
        if source_factory is None:
            raise_validation_error(
                "DatasetMaterialization",
                "source_factory is required when metaspace_api_keys_path is provided.",
            )
        rotating_source = RotatingDatasetSource(
            read_source_profiles(metaspace_api_keys_path),
            source_factory,
            initial_source=source,
        )
    call: Callable[..., Any] = (
        rotating_source.call
        if rotating_source is not None
        else lambda method_name, *args, **kwargs: getattr(source, method_name)(
            *args, **kwargs
        )
    )
    annotation_counts: Dict[str, int] = {}
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
    _update_running_manifest(manifest, target_manifest, rotating_source)
    return call, rotating_source, annotation_counts


def _download_dataset_files(
    *,
    call: Callable[..., Any],
    entries: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    target_manifest: Path,
    rotating_source: Optional[RotatingDatasetSource],
) -> bool:
    """Download planned imzML/ibd pairs and return quota-boundary status."""
    downloads_remaining = sum(
        "download_dataset" in entry["planned_actions"] for entry in entries
    )
    progress = _download_progress(
        total=len(entries),
        initial=len(entries) - downloads_remaining,
    )
    progress.set_postfix(downloads_remaining=downloads_remaining)
    progress.refresh()
    file_limit_reached = False
    try:
        for position, entry in enumerate(entries):
            dataset_id = str(entry["dataset_id"])
            destination = Path(str(entry["directory"]))
            # Binary-file phase
            ## Existing complete pairs were marked reusable when the manifest
            ## was created, so this branch needs no provider API request.
            if "download_dataset" not in entry["planned_actions"]:
                entry["execution"]["dataset"] = "reused"
            else:
                progress.set_postfix(
                    current=dataset_id,
                    downloads_remaining=downloads_remaining,
                    profiles_remaining=(
                        rotating_source.remaining_profile_count
                        if rotating_source is not None
                        else "unknown"
                    ),
                )
                entry["execution"]["dataset"] = "downloading"
                _update_running_manifest(manifest, target_manifest, rotating_source)
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
                            pending["execution"]["dataset"] = "not_executed_after_limit"
                    _update_running_manifest(manifest, target_manifest, rotating_source)
                    logger.warning(
                        "Stopping the imzML phase at dataset %s with %s downloads remaining",
                        dataset_id,
                        downloads_remaining,
                    )
                    break
            _update_running_manifest(manifest, target_manifest, rotating_source)
            if "download_dataset" in entry["planned_actions"]:
                progress.update(1)
    finally:
        progress.set_postfix(downloads_remaining=downloads_remaining)
        progress.close()
    return file_limit_reached


def _download_annotation_files(
    *,
    call: Callable[..., Any],
    entries: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    target_manifest: Path,
    rotating_source: Optional[RotatingDatasetSource],
    annotation_options: Mapping[str, Any],
    annotation_counts: Dict[str, int],
) -> List[Path]:
    """Download annotation CSVs for every complete local dataset pair."""
    annotation_downloads_remaining = sum(
        "download_annotations" in entry["planned_actions"] for entry in entries
    )
    progress = _annotation_progress(
        total=len(entries),
        initial=len(entries) - annotation_downloads_remaining,
    )
    progress.set_postfix(downloads_remaining=annotation_downloads_remaining)
    progress.refresh()
    materialized: List[Path] = []
    try:
        for entry in entries:
            dataset_id = str(entry["dataset_id"])
            destination = Path(str(entry["directory"]))
            if not has_complete_pair(destination, dataset_id):
                entry["execution"]["annotations"] = "not_executed_without_data"
                _update_running_manifest(manifest, target_manifest, rotating_source)
            else:
                imzml_path = validate_imzml_pair(destination, dataset_id)
                materialized.append(destination)
                if entry["execution"]["dataset"] in {"pending", "not_executed_after_limit"}:
                    entry["execution"]["dataset"] = "reused_after_file_phase"
                if "download_annotations" not in entry["planned_actions"]:
                    entry["execution"]["annotations"] = "reused"
                    _update_running_manifest(manifest, target_manifest, rotating_source)
                else:
                    progress.set_postfix(
                        current=dataset_id,
                        downloads_remaining=annotation_downloads_remaining,
                    )
                    entry["execution"]["annotations"] = "downloading"
                    _update_running_manifest(manifest, target_manifest, rotating_source)
                    try:
                        annotation_count = call(
                            "materialize_annotations",
                            dataset_id=dataset_id,
                            dataset_name=str(entry.get("name", dataset_id)),
                            directory=destination,
                            imzml_path=imzml_path,
                            options=annotation_options,
                        )
                    except Exception as error:
                        entry["execution"]["annotations"] = (
                            f"failed: {type(error).__name__}: {error}"
                        )
                        logger.error(
                            "Annotation materialization failed for dataset %s: %s",
                            dataset_id,
                            error,
                        )
                        _update_running_manifest(manifest, target_manifest, rotating_source)
                    else:
                        annotation_counts[dataset_id] = int(annotation_count)
                        entry["execution"]["annotations"] = "downloaded"
                        annotation_downloads_remaining -= 1
                        _update_running_manifest(manifest, target_manifest, rotating_source)
                    progress.update(1)
    finally:
        progress.set_postfix(downloads_remaining=annotation_downloads_remaining)
        progress.close()
    return materialized


def _finalize_download(
    *,
    manifest: Dict[str, Any],
    entries: Sequence[Mapping[str, Any]],
    target_manifest: Path,
    rotating_source: Optional[RotatingDatasetSource],
    file_limit_reached: bool,
) -> None:
    """Persist final status and report aggregate materialization state."""
    annotation_failures = sum(
        str(entry["execution"]["annotations"]).startswith("failed:") for entry in entries
    )
    manifest.update(
        {
            "status": "completed_with_gaps" if file_limit_reached or annotation_failures else "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _update_running_manifest(manifest, target_manifest, rotating_source)
    logger.info(
        "Materialization report: files downloaded=%s, reused=%s, annotation CSVs downloaded=%s, reused=%s, failed=%s, remaining files=%s; manifest=%s",
        sum(entry["execution"]["dataset"] == "downloaded" for entry in entries),
        sum(str(entry["execution"]["dataset"]).startswith("reused") for entry in entries),
        sum(entry["execution"]["annotations"] == "downloaded" for entry in entries),
        sum(entry["execution"]["annotations"] == "reused" for entry in entries),
        annotation_failures,
        sum(not entry["final_state"]["data_present"] for entry in entries),
        target_manifest,
    )

# --------------------------------------------------
# Section: Inner helpers 
# --------------------------------------------------

def _update_running_manifest(
    manifest: Dict[str, Any],
    target: Path,
    rotating_source: Optional[RotatingDatasetSource],
) -> None:
    """Refresh filesystem-derived status and atomically write the manifest."""
    for entry in manifest["datasets"]:
        dataset_id = str(entry["dataset_id"])
        directory = Path(str(entry["directory"]))
        entry["final_state"] = {
            "data_present": has_complete_pair(directory, dataset_id),
            "annotations_present": _has_annotation_export(
                source=str(manifest["source"]),
                directory=directory,
                dataset_id=dataset_id,
            ),
        }
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["exhausted_profile_count"] = (
        rotating_source.exhausted_profile_count if rotating_source is not None else 0
    )
    manifest["final_summary"] = final_manifest_summary(manifest["datasets"])
    write_json_atomic(target, manifest)


def _download_progress(*, total: int, initial: int) -> Any:
    """Create the terminal progress display for binary dataset downloads."""
    return tqdm(
        total=total,
        initial=initial,
        desc="Dataset files",
        unit="dataset",
        dynamic_ncols=True,
    )


def _annotation_progress(*, total: int, initial: int) -> Any:
    """Create the terminal progress display for annotation exports."""
    return tqdm(
        total=total,
        initial=initial,
        desc="Annotations",
        unit="dataset",
        dynamic_ncols=True,
    )


def _has_annotation_export(
    *,
    source: str,
    directory: Path,
    dataset_id: str,
) -> bool:
    """Delegate annotation artifact validation to its registered source class."""
    implementation = DatasetSourceManager.get_annotation_source_class(source)
    return implementation.has_annotation_export(directory, dataset_id)
