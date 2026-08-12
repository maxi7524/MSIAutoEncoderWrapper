"""Build cohort-level molecule occurrence and FDR masks from SQLite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from ...utils.exceptions import raise_validation_error
from ..catalog.sqlite_catalog import DatasetCatalog


def build_cohort_annotation_index(
    *,
    catalog: DatasetCatalog,
    source: str,
    dataset_ids: Sequence[str],
    config: Mapping[str, Any],
    output_path: Path | str | None = None,
) -> Dict[str, Any]:
    """Build molecule occurrence masks for a materialized dataset cohort.

    :param catalog: Canonical annotation catalogue.
    :type catalog: DatasetCatalog
    :param source: Provider source shared by the cohort.
    :type source: str
    :param dataset_ids: Ordered cohort dataset identifiers.
    :type dataset_ids: Sequence[str]
    :param config: Composition filters. Supported keys are ``max_fdr`` and
        ``minimum_dataset_occurrence``.
    :type config: Mapping[str, Any]
    :param output_path: Optional JSON artifact path.
    :type output_path: pathlib.Path | str | None
    :return: Molecule observations and reusable cohort masks.
    :rtype: Dict[str, Any]
    """
    ordered_ids = [str(value) for value in dataset_ids]
    if not ordered_ids or len(ordered_ids) != len(set(ordered_ids)):
        raise_validation_error(
            "CohortAnnotations", "dataset_ids must be non-empty and unique."
        )
    max_fdr = config.get("max_fdr")
    minimum_occurrence = int(config.get("minimum_dataset_occurrence", 1))
    if minimum_occurrence <= 0:
        raise_validation_error(
            "CohortAnnotations", "minimum_dataset_occurrence must be positive."
        )

    identities: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    for dataset_id in ordered_ids:
        annotations = catalog.get_annotations(
            source=source,
            dataset_id=dataset_id,
            filters={"max_fdr": max_fdr} if max_fdr is not None else None,
        )
        for annotation in annotations:
            identity = (
                str(annotation.get("sumFormula") or annotation.get("formula") or ""),
                str(annotation.get("adduct") or ""),
                str(annotation.get("database_name") or annotation.get("database") or ""),
                str(annotation.get("database_version") or ""),
            )
            molecule = identities.setdefault(
                identity,
                {
                    "formula": identity[0],
                    "adduct": identity[1],
                    "database_name": identity[2],
                    "database_version": identity[3],
                    "observations": [],
                },
            )
            molecule["observations"].append(
                {"dataset_id": dataset_id, "fdr": annotation.get("fdr")}
            )

    molecules = []
    for molecule in identities.values():
        observed_ids = {item["dataset_id"] for item in molecule["observations"]}
        occurrence_mask = [dataset_id in observed_ids for dataset_id in ordered_ids]
        molecules.append(
            {
                **molecule,
                "dataset_occurrence_count": len(observed_ids),
                "occurrence_mask": occurrence_mask,
                "single_dataset_only": len(observed_ids) == 1,
                "selected": len(observed_ids) >= minimum_occurrence,
            }
        )
    molecules.sort(
        key=lambda item: (
            item["formula"], item["adduct"], item["database_name"], item["database_version"]
        )
    )
    result = {
        "schema_version": 1,
        "source": source,
        "dataset_ids": ordered_ids,
        "config": dict(config),
        "molecules": molecules,
        "masks": {
            "all": [True] * len(molecules),
            "single_dataset": [item["single_dataset_only"] for item in molecules],
            "minimum_dataset_occurrence": [item["selected"] for item in molecules],
        },
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    return result
