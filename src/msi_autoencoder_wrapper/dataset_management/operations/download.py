"""Reusable query and materialization operations for external MSI datasets."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pyimzml.ImzMLWriter import ImzMLWriter

from ...readers.strategies.pyimzml_reader import PyImzMLReader
from ...utils.exceptions import raise_validation_error, raise_workspace_error
from ...utils.logger import get_custom_logger
from ..catalog.sqlite_catalog import DatasetCatalog
from ..normalization import normalize_spectrum_annotations
from ..sources.base import DatasetSource
from ..validators import validate_imzml_pair, validate_selection
from .spectrum_selection import select_merge_spectrum_ids


logger = get_custom_logger(__name__)


def materialize_selection(
    *,
    source: DatasetSource,
    selection_path: Path | str,
    datasets_dir: Path | str,
    catalog: DatasetCatalog,
    annotation_options: Optional[Mapping[str, Any]] = None,
    dataset_ids: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Download selected datasets and import their complete annotations.

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
    :return: Materialized dataset directories.
    :rtype: List[pathlib.Path]
    :raises ValueError: If the selection belongs to another source.
    """
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
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
    output_root = Path(datasets_dir) / "sources" / source.source_name
    materialized: List[Path] = []
    for record in selection.get("datasets", []):
        dataset_id = str(record["dataset_id"])
        if selected_ids and dataset_id not in selected_ids:
            continue
        destination = output_root / dataset_id
        logger.info("Materializing dataset %s", dataset_id)
        if _has_complete_pair(destination, dataset_id):
            logger.info(
                "Reusing downloaded dataset %s from %s",
                dataset_id,
                destination,
            )
        else:
            source.download_dataset(dataset_id, destination)
        validate_imzml_pair(destination, dataset_id)
        metadata_record = source.get_dataset_metadata(dataset_id)
        metadata = dict(metadata_record.get("metadata", {}))
        reader = PyImzMLReader(validate_imzml_pair(destination, dataset_id))
        annotations = normalize_spectrum_annotations(
            source.get_annotations(dataset_id, resolved_annotation_options), reader
        )
        catalog.upsert_dataset(
            source=source.source_name,
            dataset_id=dataset_id,
            name=str(metadata_record.get("name", record.get("name", dataset_id))),
            metadata=metadata,
            local_path=destination,
            status="materialized",
        )
        catalog.replace_annotations(
            source=source.source_name,
            dataset_id=dataset_id,
            annotations=annotations,
        )
        materialized.append(destination)
    return materialized


def materialize_and_merge_selection(
    *,
    source: DatasetSource,
    selection_path: Path | str,
    datasets_dir: Path | str,
    catalog: DatasetCatalog,
    output_path: Path | str,
    merged_dataset_id: str,
    row_width: int,
    annotation_options: Optional[Mapping[str, Any]] = None,
    dataset_ids: Optional[Sequence[str]] = None,
    keep_downloads: bool = False,
    unannotated_ratio: Optional[float] = None,
    unannotated_amount: Optional[int] = None,
    random_seed: int = 0,
) -> Path:
    """Download, append, and release one dataset at a time.

    :param source: Adapter matching the query selection.
    :type source: DatasetSource
    :param selection_path: Query selection path.
    :type selection_path: pathlib.Path | str
    :param datasets_dir: Workspace datasets directory.
    :type datasets_dir: pathlib.Path | str
    :param catalog: Workspace dataset catalog.
    :type catalog: DatasetCatalog
    :param output_path: Destination merged imzML path.
    :type output_path: pathlib.Path | str
    :param merged_dataset_id: Stable merged-dataset identifier.
    :type merged_dataset_id: str
    :param row_width: Positive rectangular output width.
    :type row_width: int
    :param annotation_options: Provider retrieval options.
    :type annotation_options: Mapping[str, Any] | None
    :param dataset_ids: Optional selection subset.
    :type dataset_ids: Sequence[str] | None
    :param keep_downloads: Retain source pairs after they are appended. Defaults
        to ``False`` for bounded temporary disk usage.
    :type keep_downloads: bool
    :param unannotated_ratio: Optional unannotated-to-annotated spectrum ratio.
    :type unannotated_ratio: float | None
    :param unannotated_amount: Optional absolute unannotated spectrum count.
    :type unannotated_amount: int | None
    :param random_seed: Reproducible unannotated sampling seed.
    :type random_seed: int
    :return: Written merged imzML path.
    :rtype: pathlib.Path

    This mode intentionally requires a fixed row width because the total number
    of spectra is not known before providers materialize each source dataset.
    A failed run removes only its partial output; cataloged source metadata and
    annotations remain safe to import again.
    """
    if row_width <= 0:
        raise_validation_error("DownloadMerge", "row_width must be greater than zero.")
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    validate_selection(selection)
    if selection.get("source") != source.source_name:
        raise_validation_error(
            "DownloadMerge",
            (
                f"Selection source '{selection.get('source')}' does not match "
                f"adapter '{source.source_name}'."
            ),
        )
    resolved_annotation_options = _resolve_annotation_options(
        selection, annotation_options
    )
    selected_ids = set(dataset_ids or [])
    records = [
        record
        for record in selection.get("datasets", [])
        if not selected_ids or str(record["dataset_id"]) in selected_ids
    ]
    if not records:
        raise_validation_error("DownloadMerge", "The dataset selection is empty.")

    output = Path(output_path).with_suffix(".imzML")
    if output.exists() or output.with_suffix(".ibd").exists():
        raise_workspace_error("DownloadMerge", f"Output pair already exists for '{output}'.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.partial.imzML")
    temporary_ibd = temporary.with_suffix(".ibd")
    temporary.unlink(missing_ok=True)
    temporary_ibd.unlink(missing_ok=True)
    staging_root = Path(datasets_dir) / ".staging" / source.source_name
    retained_root = Path(datasets_dir) / "sources" / source.source_name
    mappings: List[Dict[str, object]] = []
    merged_index = 0

    try:
        with ImzMLWriter(str(temporary), mode="processed") as writer:
            for record in records:
                dataset_id = str(record["dataset_id"])
                retained = retained_root / dataset_id
                staging = staging_root / dataset_id

                # Dataset-level resume
                ## Prefer the canonical workspace copy before contacting the provider.
                if _has_complete_pair(retained, dataset_id):
                    destination = retained
                    uses_retained_source = True
                    logger.info(
                        "Reusing downloaded dataset %s from %s",
                        dataset_id,
                        destination,
                    )
                elif retained.exists():
                    # Complete an existing canonical directory in place so the
                    # provider can skip either source file that is already present.
                    destination = retained
                    uses_retained_source = True
                    source.download_dataset(dataset_id, destination)
                else:
                    destination = staging
                    uses_retained_source = False
                    # Keep partial staging contents: the provider client can skip
                    # an existing member of the imzML/ibd pair and fetch its companion.
                    source.download_dataset(dataset_id, destination)
                validate_imzml_pair(destination, dataset_id)
                metadata_record = source.get_dataset_metadata(dataset_id)
                imzml_path = validate_imzml_pair(destination, dataset_id)
                reader = PyImzMLReader(imzml_path)
                annotations = normalize_spectrum_annotations(
                    source.get_annotations(dataset_id, resolved_annotation_options), reader
                )
                metadata = dict(metadata_record.get("metadata", {}))
                catalog.upsert_dataset(
                    source=source.source_name,
                    dataset_id=dataset_id,
                    name=str(metadata_record.get("name", record.get("name", dataset_id))),
                    metadata=metadata,
                    local_path=(retained_root / dataset_id) if keep_downloads else None,
                    status="materialized" if keep_downloads else "merged",
                )
                catalog.replace_annotations(
                    source=source.source_name,
                    dataset_id=dataset_id,
                    annotations=annotations,
                )
                selected_spectrum_ids = record.get("spectrum_ids")
                spectrum_ids = (
                    select_merge_spectrum_ids(
                        candidate_ids=list(range(reader.GetNumberOfSpectra())),
                        annotated_ids=catalog.get_annotated_spectrum_ids(
                            source=source.source_name,
                            dataset_id=dataset_id,
                        ),
                        unannotated_ratio=unannotated_ratio,
                        unannotated_amount=unannotated_amount,
                        random_seed=random_seed,
                        seed_namespace=f"{source.source_name}:{dataset_id}",
                    )
                    if selected_spectrum_ids is None
                    else [int(value) for value in selected_spectrum_ids]
                )
                for source_spectrum_id in spectrum_ids:
                    if source_spectrum_id < 0 or source_spectrum_id >= reader.GetNumberOfSpectra():
                        raise_validation_error(
                            "DownloadMerge",
                            f"Dataset '{dataset_id}' has invalid spectrum ID {source_spectrum_id}.",
                        )
                    mass_axis, intensities = reader.GetSpectrum(source_spectrum_id)
                    writer.addSpectrum(
                        mass_axis,
                        intensities,
                        (
                            merged_index % row_width + 1,
                            merged_index // row_width + 1,
                            1,
                        ),
                        userParams=[
                            {"name": "source", "value": source.source_name},
                            {"name": "source_dataset_id", "value": dataset_id},
                            {"name": "source_spectrum_id", "value": str(source_spectrum_id)},
                        ],
                    )
                    mappings.append(
                        {
                            "source": source.source_name,
                            "source_dataset_id": dataset_id,
                            "source_spectrum_id": source_spectrum_id,
                            "merged_spectrum_index": merged_index,
                        }
                    )
                    merged_index += 1
                if keep_downloads and not uses_retained_source:
                    retained.parent.mkdir(parents=True, exist_ok=True)
                    if retained.exists():
                        shutil.rmtree(retained)
                    destination.replace(retained)
                elif not uses_retained_source:
                    shutil.rmtree(destination)
        if merged_index == 0:
            raise_validation_error("DownloadMerge", "Selected datasets contain no spectra.")
        temporary.replace(output)
        temporary_ibd.replace(output.with_suffix(".ibd"))
    except Exception:
        temporary.unlink(missing_ok=True)
        temporary_ibd.unlink(missing_ok=True)
        raise

    catalog.register_merged_dataset(merged_dataset_id, output)
    catalog.replace_spectrum_mappings(merged_dataset_id, mappings)
    logger.info(
        "Downloaded and merged %s spectra from %s datasets into %s",
        merged_index,
        len(records),
        output,
    )
    return output


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
