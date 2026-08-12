"""Reusable query and materialization operations for external MSI datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..imzml import PyImzMLReader
from ..utils.exceptions import DownloadLimitError, raise_validation_error
from ..utils.logger import get_custom_logger
from ..catalog.sqlite_catalog import DatasetCatalog
from ..sources.base import DatasetSource
from ..sources.profiles import RotatingDatasetSource, read_source_profiles
from ..validators import validate_imzml_pair, validate_selection
from .annotation_csv import (
    has_complete_annotation_csv,
    write_annotation_csv_pair,
)


logger = get_custom_logger(__name__)


def materialize_selection(
    *,
    source: DatasetSource,
    selection_path: Path | str,
    datasets_dir: Path | str,
    catalog: DatasetCatalog,
    annotation_options: Optional[Mapping[str, Any]] = None,
    dataset_ids: Optional[Sequence[str]] = None,
    profiles_path: Optional[Path | str] = None,
    source_factory: Optional[
        Callable[[str, Mapping[str, str]], DatasetSource]
    ] = None,
    manifest_path: Optional[Path | str] = None,
) -> List[Path]:
    """Download selected datasets and persist canonical annotation CSV files.

    :param source: Adapter matching the selection source.
    :type source: DatasetSource
    :param selection_path: Query selection created by :func:`query_to_selection`.
    :type selection_path: pathlib.Path | str
    :param datasets_dir: Workspace datasets directory.
    :type datasets_dir: pathlib.Path | str
    :param catalog: Workspace dataset catalog.
    :type catalog: DatasetCatalog
    :param annotation_options: Provider retrieval options, not experimental
        read-time filters.
    :type annotation_options: Mapping[str, Any] | None
    :param dataset_ids: Optional explicit subset of selected IDs.
    :type dataset_ids: Sequence[str] | None
    :param profiles_path: Optional CSV with a mandatory ``key`` column.
    :type profiles_path: pathlib.Path | str | None
    :param source_factory: Factory used to create a source after rotating to a
        new key. ``source`` must use the first CSV key.
    :type source_factory: Callable[[str, Mapping[str, str]], DatasetSource] | None
    :param manifest_path: Optional output path for the materialization report.
    :type manifest_path: pathlib.Path | str | None
    :return: Materialized dataset directories. Annotation records are not
        imported into SQLite until cohort composition.
    :rtype: List[pathlib.Path]
    :raises ValueError: If the selection belongs to another source.
    """
    selection_file = Path(selection_path)
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    validate_selection(selection)
    if selection.get("source") != source.source_name:
        raise ValueError(
            f"Selection source '{selection.get('source')}' does not match "
            f"adapter '{source.source_name}'."
        )
    resolved_annotation_options = _resolve_annotation_options(
        selection, annotation_options
    )
    selected_ids = set(dataset_ids or [])
    # Dataset file materialization
    ## Keep provider-independent source pairs directly under datasets/<dataset_id>.
    output_root = Path(datasets_dir)
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
    records = [
        record
        for record in selection.get("datasets", [])
        if not selected_ids or str(record["dataset_id"]) in selected_ids
    ]
    materialized: List[Path] = []
    materialized_records: List[tuple[Mapping[str, Any], Path]] = []
    file_statuses: Dict[str, str] = {}
    annotation_statuses: Dict[str, str] = {}
    file_limit_reached = False

    # Dataset file materialization
    ## Download or reuse every imzML/ibd pair before requesting annotations.
    for record in records:
        dataset_id = str(record["dataset_id"])
        destination = output_root / dataset_id
        logger.info("Materializing dataset %s", dataset_id)
        if _has_complete_pair(destination, dataset_id):
            logger.info(
                "Reusing downloaded dataset %s from %s",
                dataset_id,
                destination,
            )
            file_statuses[dataset_id] = "reused"
        else:
            try:
                call("download_dataset", dataset_id, destination)
                file_statuses[dataset_id] = "downloaded"
            except DownloadLimitError:
                file_statuses[dataset_id] = "download_limit"
                file_limit_reached = True
                logger.warning(
                    "Stopping the imzML phase at dataset %s and continuing with "
                    "annotations for complete local pairs",
                    dataset_id,
                )
                break
        validate_imzml_pair(destination, dataset_id)
        metadata_record = record
        metadata = dict(record.get("metadata", {}))
        catalog.upsert_dataset(
            source=source.source_name,
            dataset_id=dataset_id,
            name=str(metadata_record.get("name", record.get("name", dataset_id))),
            metadata=metadata,
            local_path=destination,
            status="materialized",
        )
        materialized.append(destination)
        materialized_records.append((record, destination))

    # Local-pair discovery after quota exhaustion
    ## A failed early download must not hide complete pairs later in selection order.
    if file_limit_reached:
        known_ids = {str(record["dataset_id"]) for record, _ in materialized_records}
        for record in records:
            dataset_id = str(record["dataset_id"])
            destination = output_root / dataset_id
            if dataset_id in known_ids or not _has_complete_pair(destination, dataset_id):
                continue
            validate_imzml_pair(destination, dataset_id)
            catalog.upsert_dataset(
                source=source.source_name,
                dataset_id=dataset_id,
                name=str(record.get("name", dataset_id)),
                metadata=dict(record.get("metadata", {})),
                local_path=destination,
                status="materialized",
            )
            file_statuses[dataset_id] = "reused"
            materialized.append(destination)
            materialized_records.append((record, destination))

    # Canonical annotation materialization
    ## Persist provider-independent CSV files; SQLite is created only by compose.
    annotation_counts: Dict[str, int] = {}
    reused_annotations: List[str] = []
    for record, destination in materialized_records:
        dataset_id = str(record["dataset_id"])
        if has_complete_annotation_csv(destination, dataset_id):
            reused_annotations.append(dataset_id)
            annotation_statuses[dataset_id] = "reused"
            logger.info("Reusing annotations for dataset %s", dataset_id)
            continue
        reader = PyImzMLReader(validate_imzml_pair(destination, dataset_id))
        try:
            provider_annotations = call(
                "get_annotations", dataset_id, resolved_annotation_options
            )
            write_annotation_csv_pair(
                directory=destination,
                dataset_id=dataset_id,
                dataset_name=str(record.get("name", dataset_id)),
                annotations=provider_annotations,
                reader=reader,
            )
            annotation_count = len(provider_annotations)
            catalog.upsert_dataset(
                source=source.source_name,
                dataset_id=dataset_id,
                name=str(record.get("name", dataset_id)),
                metadata=dict(record.get("metadata", {})),
                local_path=destination,
                status="annotation_files_materialized",
            )
        except Exception as error:
            annotation_statuses[dataset_id] = f"failed: {type(error).__name__}: {error}"
            logger.error(
                "Annotation materialization failed for dataset %s: %s",
                dataset_id,
                error,
            )
            continue
        annotation_counts[dataset_id] = annotation_count
        annotation_statuses[dataset_id] = "downloaded"

    target_manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else selection_file.parent / "materialization.json"
    )
    _write_json_atomic(
        target_manifest,
        {
            "schema_version": 1,
            "source": source.source_name,
            "selection_path": str(selection_file.resolve()),
            "selection": selection,
            "annotation_options": resolved_annotation_options,
            "dataset_ids": [str(record["dataset_id"]) for record in records],
            "annotation_counts": annotation_counts,
            "file_statuses": file_statuses,
            "annotation_statuses": annotation_statuses,
            "file_limit_reached": file_limit_reached,
            "reused_annotations": reused_annotations,
            "exhausted_profile_count": (
                rotating_source.exhausted_profile_count
                if rotating_source is not None
                else 0
            ),
        },
    )
    logger.info(
        "Materialization report: files downloaded=%s, reused=%s, annotation CSVs "
        "downloaded=%s, reused=%s, failed=%s, remaining files=%s; manifest=%s",
        sum(value == "downloaded" for value in file_statuses.values()),
        sum(value == "reused" for value in file_statuses.values()),
        sum(value == "downloaded" for value in annotation_statuses.values()),
        sum(value == "reused" for value in annotation_statuses.values()),
        sum(value.startswith("failed:") for value in annotation_statuses.values()),
        len(records) - len(materialized),
        target_manifest,
    )
    return materialized


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
