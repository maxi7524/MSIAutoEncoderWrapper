"""Tests for the final merged annotation representation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from msi_dataset_manager.annotations import AnnotationReader
from msi_dataset_manager.annotations.blobs import (
    decode_pixel_indices,
    encode_pixel_indices,
)
from msi_dataset_manager.annotations.merge import (
    AnnotationMergeInput,
    MergedAnnotationWriter,
)
from msi_dataset_manager.sources.base import SourceAnnotationExport


def _export(dataset_id: str, records: list[dict]) -> SourceAnnotationExport:
    return SourceAnnotationExport(
        source="metaspace",
        dataset_id=dataset_id,
        schema_version=2,
        metadata={"condition": f"condition-{dataset_id}"},
        records=records,
    )


def test_pixel_blob_round_trip_is_sorted_and_unique() -> None:
    """Pixel blobs retain exact indices without storing per-pixel SQL rows."""
    assert decode_pixel_indices(encode_pixel_indices([1203, 14, 18, 18])) == [
        14,
        18,
        1203,
    ]


def test_writer_and_reader_preserve_classes_references_and_segments(
    tmp_path: Path,
) -> None:
    """The merged store separates global classes from source-specific evidence."""
    records_a = [
        {
            "source_annotation_id": "db-a:C6H12O6+H",
            "formula": "C6H12O6",
            "adduct": "+H",
            "mz": 181.0707,
            "fdr": 0.05,
            "database_id": "db-a",
            "database_name": "HMDB",
            "database_version": "1",
            "source_record": {"provider": "record-a"},
            "spectrum_ids": [0, 2],
        },
        {
            "source_annotation_id": "db-a:local-only",
            "formula": "C2H4O2",
            "adduct": "-H",
            "mz": 59.0139,
            "fdr": 0.05,
            "source_record": {},
            "spectrum_ids": [2],
        },
    ]
    records_b = [
        {
            "source_annotation_id": "db-b:C6H12O6+H",
            "formula": "C6H12O6",
            "adduct": "+H",
            "mz": 181.0711,
            "fdr": 0.1,
            "database_id": "db-b",
            "database_name": "Core",
            "database_version": "2",
            "source_record": {"provider": "record-b"},
            "spectrum_ids": [5],
        }
    ]
    inputs = [
        AnnotationMergeInput(
            source="metaspace",
            dataset_id="a",
            name="A",
            imzml_path=tmp_path / "a.imzML",
            annotation_export=_export("a", records_a),
            metadata={"tissue": "kidney"},
        ),
        AnnotationMergeInput(
            source="metaspace",
            dataset_id="b",
            name="B",
            imzml_path=tmp_path / "b.imzML",
            annotation_export=_export("b", records_b),
            metadata={"tissue": "kidney"},
        ),
    ]
    mappings = [
        {
            "source": "metaspace",
            "source_dataset_id": "a",
            "source_spectrum_id": 0,
            "merged_spectrum_index": 0,
        },
        {
            "source": "metaspace",
            "source_dataset_id": "a",
            "source_spectrum_id": 2,
            "merged_spectrum_index": 1,
        },
        {
            "source": "metaspace",
            "source_dataset_id": "b",
            "source_spectrum_id": 5,
            "merged_spectrum_index": 2,
        },
    ]
    store = MergedAnnotationWriter().write(
        path=tmp_path / "merged.sqlite",
        inputs=inputs,
        pixel_mappings=mappings,
        filtering_metadata={"max_fdr": 0.1},
        max_fdr=0.1,
    )

    reader = AnnotationReader.load(type="merge", path=store)
    annotations = reader.get_annotations()
    assert [(item["formula"], item["adduct"]) for item in annotations] == [
        ("C2H4O2", "-H"),
        ("C6H12O6", "+H"),
    ]
    assert annotations[0]["spectrum_ids"] == [1]
    assert annotations[1]["spectrum_ids"] == [0, 1, 2]
    assert reader.get_spectrum_metadata(1)["source_spectrum_id"] == 2
    assert reader.get_spectrum_metadata(2)["dataset_id"] == "b"
    assert reader.get_spectrum_groups([0, 1, 2]) == [("a",), ("a",), ("b",)]
    reference = reader.get_spectrum_annotations(2)[0]
    assert reference["mz"] == 181.0711
    assert reference["database_name"] == "Core"
    assert "intensity" not in reference

    with sqlite3.connect(store) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        merged_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(merged_annotations)")
        }
    assert tables == {
        "datasets_metadata",
        "pixel_segments",
        "merged_annotations",
        "reference_annotations_0001",
        "reference_annotations_0002",
    }
    assert "pixel_intensities_blob" not in merged_columns
