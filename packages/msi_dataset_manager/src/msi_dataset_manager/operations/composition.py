"""Compose a reproducible cohort dataset from canonical local source folders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from ..catalog import DatasetCatalog
from ..layout import DatasetWorkspaceLayout
from ..utils.exceptions import raise_validation_error
from ..utils.logger import get_custom_logger
from ..validators import validate_imzml_pair
from .annotation_csv import annotation_csv_paths, has_complete_annotation_csv
from .cohort_annotations import build_cohort_annotation_index
from .import_local import import_local_dataset
from .merge import ImzMLMergeInput, ImzMLMerger


logger = get_custom_logger(__name__)


def compose_cohort(
    *,
    workspace_path: Path | str,
    cohort_id: str,
    source: str,
    dataset_ids: Sequence[str],
    row_width: Optional[int] = None,
    max_fdr: Optional[float] = None,
    minimum_dataset_occurrence: int = 1,
    unannotated_ratio: Optional[float] = None,
    unannotated_amount: Optional[int] = None,
    random_seed: int = 0,
    config: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Build one merged cohort and its self-contained result catalogue.

    :param workspace_path: Workspace containing ``datasets`` and ``configs``.
    :type workspace_path: pathlib.Path | str
    :param cohort_id: Output dataset and cohort-catalog identifier.
    :type cohort_id: str
    :param source: Provider metadata key shared by input datasets.
    :type source: str
    :param dataset_ids: Ordered identifiers of canonical local datasets.
    :type dataset_ids: Sequence[str]
    :param row_width: Optional output image width.
    :type row_width: int | None
    :param max_fdr: Optional annotation FDR threshold used for masks and spectra.
    :type max_fdr: float | None
    :param minimum_dataset_occurrence: Minimum datasets containing a molecule.
    :type minimum_dataset_occurrence: int
    :param unannotated_ratio: Optional unannotated-to-annotated ratio.
    :type unannotated_ratio: float | None
    :param unannotated_amount: Optional absolute unannotated count. ``None``
        retains all available unannotated spectra; ``0`` retains none.
    :type unannotated_amount: int | None
    :param random_seed: Reproducible sampling seed.
    :type random_seed: int
    :param config: Additional configuration fields retained as provenance.
    :type config: Mapping[str, Any] | None
    :return: Merged imzML path.
    :rtype: pathlib.Path

    Composition is the only stage that creates SQLite. It validates canonical
    source pairs, imports their paired annotation CSVs into one common schema,
    merges selected spectra, stores source-to-merged index provenance, builds
    cohort-level annotation masks, and finally writes normalized configuration.
    """
    # Composition request validation
    ordered_ids = [str(value) for value in dataset_ids]
    if not ordered_ids or len(ordered_ids) != len(set(ordered_ids)):
        raise_validation_error("Composition", "dataset_ids must be non-empty and unique.")

    layout = DatasetWorkspaceLayout(workspace_path)
    # Result catalogue initialization
    ## This is the only SQLite catalogue in the workflow. Query and download
    ## use JSON artifacts and the filesystem instead of maintaining shadow state.
    catalog = DatasetCatalog(layout.composed_catalog_path(cohort_id))
    inputs = []
    available_ids = []
    missing_ids = []

    # Canonical input validation and cohort-local annotation import
    ## Source pairs remain shared under datasets/<source_id>. Paired source CSVs
    ## are normalized by the annotation reader into the common SQLite schema.
    for dataset_id in ordered_ids:
        directory = layout.dataset_dir(dataset_id)
        if not (
            (directory / f"{dataset_id}.imzML").is_file()
            and (directory / f"{dataset_id}.ibd").is_file()
        ):
            missing_ids.append(dataset_id)
            logger.warning("Skipping unavailable local dataset %s", dataset_id)
            continue
        imzml_path = validate_imzml_pair(directory, dataset_id)
        available_ids.append(dataset_id)
        if has_complete_annotation_csv(directory, dataset_id):
            annotations_path, intensities_path = annotation_csv_paths(directory, dataset_id)
            import_local_dataset(
                catalog=catalog,
                source=source,
                dataset_id=dataset_id,
                name=dataset_id,
                imzml_path=imzml_path,
                annotations_path=annotations_path,
                pixel_intensities_path=intensities_path,
            )
        else:
            catalog.upsert_dataset(
                source=source,
                dataset_id=dataset_id,
                name=dataset_id,
                metadata={},
                local_path=directory,
                status="materialized_without_annotations",
            )
        inputs.append(ImzMLMergeInput(source, dataset_id, imzml_path))
    if not inputs:
        raise_validation_error(
            "Composition", "None of the requested datasets has a complete local imzML pair."
        )

    # Reproducible composition configuration
    ## Record both requested and actually available datasets before merging.
    normalized: Dict[str, Any] = {
        **dict(config or {}),
        "schema_version": 1,
        "cohort_id": cohort_id,
        "source": source,
        "requested_dataset_ids": ordered_ids,
        "dataset_ids": available_ids,
        "missing_dataset_ids": missing_ids,
        "row_width": row_width,
        "max_fdr": max_fdr,
        "minimum_dataset_occurrence": int(minimum_dataset_occurrence),
        "unannotated_ratio": unannotated_ratio,
        "unannotated_amount": unannotated_amount,
        "random_seed": int(random_seed),
    }
    # Spectrum merge and provenance
    ## ImzMLMerger writes the result pair and source-to-output spectrum mappings.
    output = layout.imzml_path(cohort_id)
    ImzMLMerger(catalog).merge(
        inputs=inputs,
        output_path=output,
        merged_dataset_id=cohort_id,
        row_width=row_width,
        max_fdr=max_fdr,
        unannotated_ratio=unannotated_ratio,
        unannotated_amount=unannotated_amount,
        random_seed=random_seed,
    )
    # Cohort annotation index
    ## Derive occurrence/FDR masks from the now-complete common SQLite catalogue.
    build_cohort_annotation_index(
        catalog=catalog,
        source=source,
        dataset_ids=available_ids,
        config=normalized,
        output_path=output.parent / "annotation_index.json",
    )
    # Final composition artifact
    _write_json_atomic(layout.composition_path(cohort_id), normalized)
    return output


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically persist one formatted JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
