"""Reusable discovery and materialization stages for external MSI datasets."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pyimzml.ImzMLWriter import ImzMLWriter

from ..readers.strategies.pyimzml_reader import PyImzMLReader
from ..utils.exceptions import raise_validation_error, raise_workspace_error
from ..utils.logger import get_custom_logger
from ..workspace.dataset_catalog import DatasetCatalog
from .base_source import DatasetSource


logger = get_custom_logger(__name__)


def discover_to_manifest(
    *,
    source: DatasetSource,
    filters: Mapping[str, Any],
    catalog: DatasetCatalog,
    manifest_path: Path | str,
) -> List[Dict[str, Any]]:
    """Discover dataset metadata, update the catalog, and write a manifest.

    :param source: External provider adapter.
    :type source: DatasetSource
    :param filters: Native provider-side discovery filters.
    :type filters: Mapping[str, Any]
    :param catalog: Workspace dataset catalog.
    :type catalog: DatasetCatalog
    :param manifest_path: Destination JSON manifest.
    :type manifest_path: pathlib.Path | str
    :return: Manifest dataset records.
    :rtype: List[Dict[str, Any]]
    """
    records = source.search_datasets(filters)
    manifest_records: List[Dict[str, Any]] = []
    for record in records:
        dataset_id = str(record["dataset_id"])
        name = str(record.get("name", dataset_id))
        metadata = dict(record.get("metadata", {}))
        catalog.upsert_dataset(
            source=source.source_name,
            dataset_id=dataset_id,
            name=name,
            metadata=metadata,
        )
        manifest_records.append(
            {
                "source": source.source_name,
                "dataset_id": dataset_id,
                "name": name,
                "metadata": metadata,
            }
        )
    manifest = {
        "schema_version": 1,
        "source": source.source_name,
        "filters": dict(filters),
        "datasets": manifest_records,
    }
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Wrote discovery manifest with %s datasets to %s", len(records), target)
    return manifest_records


def materialize_manifest(
    *,
    source: DatasetSource,
    manifest_path: Path | str,
    datasets_dir: Path | str,
    catalog: DatasetCatalog,
    annotation_options: Optional[Mapping[str, Any]] = None,
    dataset_ids: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Download manifest datasets and import their complete annotations.

    :param source: Adapter matching the manifest source.
    :type source: DatasetSource
    :param manifest_path: Discovery manifest created by
        :func:`discover_to_manifest`.
    :type manifest_path: pathlib.Path | str
    :param datasets_dir: Workspace datasets directory.
    :type datasets_dir: pathlib.Path | str
    :param catalog: Workspace dataset catalog.
    :type catalog: DatasetCatalog
    :param annotation_options: Provider retrieval options, not experimental
        read-time filters.
    :type annotation_options: Mapping[str, Any] | None
    :param dataset_ids: Optional explicit subset of manifest IDs.
    :type dataset_ids: Sequence[str] | None
    :return: Materialized dataset directories.
    :rtype: List[pathlib.Path]
    :raises ValueError: If the manifest belongs to another source.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("source") != source.source_name:
        raise ValueError(
            f"Manifest source '{manifest.get('source')}' does not match "
            f"adapter '{source.source_name}'."
        )
    selected_ids = set(dataset_ids or [])
    output_root = Path(datasets_dir) / "sources" / source.source_name
    materialized: List[Path] = []
    for record in manifest.get("datasets", []):
        dataset_id = str(record["dataset_id"])
        if selected_ids and dataset_id not in selected_ids:
            continue
        destination = output_root / dataset_id
        logger.info("Materializing dataset %s", dataset_id)
        source.download_dataset(dataset_id, destination)
        metadata_record = source.get_dataset_metadata(dataset_id)
        metadata = dict(metadata_record.get("metadata", {}))
        annotations = source.get_annotations(dataset_id, annotation_options)
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
        (destination / "metadata.json").write_text(
            json.dumps(metadata_record, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        (destination / "annotations.json").write_text(
            json.dumps(annotations, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        materialized.append(destination)
    return materialized


def materialize_and_merge_manifest(
    *,
    source: DatasetSource,
    manifest_path: Path | str,
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

    :param source: Adapter matching the discovery manifest.
    :type source: DatasetSource
    :param manifest_path: Discovery manifest path.
    :type manifest_path: pathlib.Path | str
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
    :param dataset_ids: Optional manifest subset.
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
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("source") != source.source_name:
        raise_validation_error(
            "DownloadMerge",
            (
                f"Manifest source '{manifest.get('source')}' does not match "
                f"adapter '{source.source_name}'."
            ),
        )
    selected_ids = set(dataset_ids or [])
    records = [
        record
        for record in manifest.get("datasets", [])
        if not selected_ids or str(record["dataset_id"]) in selected_ids
    ]
    if not records:
        raise_validation_error("DownloadMerge", "The manifest selection is empty.")

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
                metadata_record = source.get_dataset_metadata(dataset_id)
                annotations = source.get_annotations(dataset_id, annotation_options)
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
                imzml_path = destination / f"{dataset_id}.imzML"
                reader = PyImzMLReader(imzml_path)
                selected_spatial_ids = record.get("spatial_ids")
                spatial_ids = (
                    list(range(reader.GetNumberOfSpectra()))
                    if selected_spatial_ids is None
                    else [int(value) for value in selected_spatial_ids]
                )
                for source_spatial_id in spatial_ids:
                    if source_spatial_id < 0 or source_spatial_id >= reader.GetNumberOfSpectra():
                        raise_validation_error(
                            "DownloadMerge",
                            f"Dataset '{dataset_id}' has invalid spatial ID {source_spatial_id}.",
                        )
                    mass_axis, intensities = reader.GetSpectrum(source_spatial_id)
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
                            {"name": "source_spatial_id", "value": str(source_spatial_id)},
                        ],
                    )
                    mappings.append(
                        {
                            "source": source.source_name,
                            "source_dataset_id": dataset_id,
                            "source_spatial_id": source_spatial_id,
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
                    (retained / "metadata.json").write_text(
                        json.dumps(metadata_record, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8",
                    )
                    (retained / "annotations.json").write_text(
                        json.dumps(annotations, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8",
                    )
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
