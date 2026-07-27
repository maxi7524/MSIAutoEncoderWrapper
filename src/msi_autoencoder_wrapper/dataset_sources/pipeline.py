"""Reusable query and materialization operations for external MSI datasets."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from pyimzml.ImzMLWriter import ImzMLWriter

from ..annotations.validators import validate_annotation_record
from ..readers.strategies.pyimzml_reader import PyImzMLReader
from ..utils.exceptions import raise_validation_error, raise_workspace_error
from ..utils.logger import get_custom_logger
from ..workspace.dataset_catalog import DatasetCatalog
from .base_source import DatasetSource
from .validators import validate_imzml_pair, validate_selection, validate_source_record


logger = get_custom_logger(__name__)


def query_to_selection(
    *,
    source: DatasetSource,
    filters: Mapping[str, Any],
    catalog: DatasetCatalog,
    selection_path: Path | str,
) -> List[Dict[str, Any]]:
    """Query dataset metadata, update SQLite, and write a reproducible selection.

    :param source: External provider adapter.
    :type source: DatasetSource
    :param filters: Native provider-side discovery filters.
    :type filters: Mapping[str, Any]
    :param catalog: Workspace dataset catalog.
    :type catalog: DatasetCatalog
    :param selection_path: Destination JSON selection file.
    :type selection_path: pathlib.Path | str
    :return: Selected dataset records.
    :rtype: List[Dict[str, Any]]
    """
    records = source.search_datasets(filters)
    selection_records: List[Dict[str, Any]] = []
    for record in records:
        validate_source_record(record)
        dataset_id = str(record["dataset_id"])
        name = str(record.get("name", dataset_id))
        metadata = dict(record.get("metadata", {}))
        catalog.upsert_dataset(
            source=source.source_name,
            dataset_id=dataset_id,
            name=name,
            metadata=metadata,
        )
        selection_records.append(
            {
                "source": source.source_name,
                "dataset_id": dataset_id,
                "name": name,
                "metadata": metadata,
            }
        )
    selection = {
        "schema_version": 1,
        "source": source.source_name,
        "filters": dict(filters),
        "datasets": selection_records,
    }
    target = Path(selection_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Wrote query selection with %s datasets to %s", len(records), target)
    return selection_records


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
    selected_ids = set(dataset_ids or [])
    output_root = Path(datasets_dir) / "sources" / source.source_name
    materialized: List[Path] = []
    for record in selection.get("datasets", []):
        dataset_id = str(record["dataset_id"])
        if selected_ids and dataset_id not in selected_ids:
            continue
        destination = output_root / dataset_id
        logger.info("Materializing dataset %s", dataset_id)
        source.download_dataset(dataset_id, destination)
        validate_imzml_pair(destination, dataset_id)
        metadata_record = source.get_dataset_metadata(dataset_id)
        metadata = dict(metadata_record.get("metadata", {}))
        reader = PyImzMLReader(validate_imzml_pair(destination, dataset_id))
        annotations = _normalize_spectrum_annotations(
            source.get_annotations(dataset_id, annotation_options), reader
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
                destination = staging_root / dataset_id
                if destination.exists():
                    shutil.rmtree(destination)
                source.download_dataset(dataset_id, destination)
                validate_imzml_pair(destination, dataset_id)
                metadata_record = source.get_dataset_metadata(dataset_id)
                imzml_path = validate_imzml_pair(destination, dataset_id)
                reader = PyImzMLReader(imzml_path)
                annotations = _normalize_spectrum_annotations(
                    source.get_annotations(dataset_id, annotation_options), reader
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
                    list(range(reader.GetNumberOfSpectra()))
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
                if keep_downloads:
                    retained = retained_root / dataset_id
                    retained.parent.mkdir(parents=True, exist_ok=True)
                    if retained.exists():
                        shutil.rmtree(retained)
                    destination.replace(retained)
                else:
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


def _normalize_spectrum_annotations(
    annotations: Sequence[Mapping[str, Any]],
    reader: PyImzMLReader,
) -> List[Dict[str, Any]]:
    """Map provider ion-image coordinates to canonical ``spectrum_id`` links.

    :param annotations: Provider records, optionally containing ``ion_image``.
    :type annotations: Sequence[Mapping[str, Any]]
    :param reader: Reader defining stable spectrum IDs and coordinates.
    :type reader: PyImzMLReader
    :return: Canonical records suitable for SQLite storage.
    :rtype: List[Dict[str, Any]]

    Positive finite ion-image values are stored sparsely. The full imzML
    spectrum remains the source of intensity data; this relation is only the
    spatial molecular annotation supplied by the external database.
    """
    normalized: List[Dict[str, Any]] = []
    for annotation in annotations:
        record = dict(annotation)
        validate_annotation_record(record)
        ion_image_value = record.pop("ion_image", None)
        if ion_image_value is None:
            normalized.append(record)
            continue
        ion_image = np.asarray(ion_image_value)
        spectrum_ids: List[int] = []
        spectrum_values: Dict[int, float] = {}
        for spectrum_id in range(reader.GetNumberOfSpectra()):
            x_position, y_position, _ = reader.GetSpectrumPosition(spectrum_id)
            row = y_position - 1
            column = x_position - 1
            if row < 0 or column < 0 or row >= ion_image.shape[0] or column >= ion_image.shape[1]:
                continue
            intensity = float(ion_image[row, column])
            if np.isfinite(intensity) and intensity > 0:
                spectrum_ids.append(spectrum_id)
                spectrum_values[spectrum_id] = intensity
        record["spectrum_ids"] = spectrum_ids
        record["spectrum_values"] = spectrum_values
        normalized.append(record)
    return normalized
