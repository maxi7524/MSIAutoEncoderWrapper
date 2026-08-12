"""Tests for cohort-level annotation masks."""

from __future__ import annotations

from msi_dataset_manager.catalog import DatasetCatalog
from msi_dataset_manager.operations import (
    build_cohort_annotation_index,
)


def test_cohort_index_retains_fdr_and_marks_single_dataset_molecules(tmp_path) -> None:
    """Local composition filters preserve observations while creating masks."""
    catalog = DatasetCatalog(tmp_path / "catalog.sqlite")
    for dataset_id in ("one", "two"):
        catalog.upsert_dataset(
            source="fake",
            dataset_id=dataset_id,
            name=dataset_id,
            metadata={},
        )
    catalog.replace_annotations(
        source="fake",
        dataset_id="one",
        annotations=[
            {"id": "shared-one", "sumFormula": "C1", "adduct": "+H", "fdr": 0.05},
            {"id": "unique", "sumFormula": "C2", "adduct": "+H", "fdr": 0.02},
        ],
    )
    catalog.replace_annotations(
        source="fake",
        dataset_id="two",
        annotations=[
            {"id": "shared-two", "sumFormula": "C1", "adduct": "+H", "fdr": 0.08},
            {"id": "rejected-fdr", "sumFormula": "C3", "adduct": "+H", "fdr": 0.2},
        ],
    )

    result = build_cohort_annotation_index(
        catalog=catalog,
        source="fake",
        dataset_ids=["one", "two"],
        config={"max_fdr": 0.1, "minimum_dataset_occurrence": 2},
    )

    assert [item["formula"] for item in result["molecules"]] == ["C1", "C2"]
    assert result["molecules"][0]["occurrence_mask"] == [True, True]
    assert [item["fdr"] for item in result["molecules"][0]["observations"]] == [
        0.05,
        0.08,
    ]
    assert result["masks"]["single_dataset"] == [False, True]
    assert result["masks"]["minimum_dataset_occurrence"] == [True, False]
