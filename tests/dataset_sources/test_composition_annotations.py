"""Workflow test for composition and its final annotation store."""

from __future__ import annotations

import shutil
from pathlib import Path

from msi_dataset_manager.annotations import AnnotationReader
from msi_dataset_manager.operations.composition import compose_cohort
from msi_dataset_manager.sources.base import SourceAnnotationExport
from msi_dataset_manager.sources.source_manager import DatasetSourceManager


def test_composition_writes_the_final_annotation_store(
    tmp_path: Path,
    msi_fixture_path: Path,
    monkeypatch,
) -> None:
    """Composition uses source records and publishes one readable SQLite file."""
    workspace = tmp_path / "workspace"
    directory = workspace / "datasets" / "source-a"
    directory.mkdir(parents=True)
    image_path = directory / "source-a.imzML"
    shutil.copy2(msi_fixture_path, image_path)
    shutil.copy2(msi_fixture_path.with_suffix(".ibd"), image_path.with_suffix(".ibd"))
    export = SourceAnnotationExport(
        source="metaspace",
        dataset_id="source-a",
        schema_version=2,
        metadata={"condition": "disease"},
        records=[
            {
                "source_annotation_id": "db:C6H12O6+H",
                "formula": "C6H12O6",
                "adduct": "+H",
                "mz": 181.0707,
                "fdr": 0.05,
                "database_name": "HMDB",
                "source_record": {"provider": "record"},
                "spectrum_ids": [0, 2],
            }
        ],
    )
    monkeypatch.setattr(
        DatasetSourceManager,
        "read_annotation_export",
        staticmethod(lambda **_: export),
    )
    manifest = {
        "source": "metaspace",
        "requested_dataset_ids": ["source-a"],
        "dataset_ids": ["source-a"],
        "missing_dataset_ids": [],
        "available_inputs": [
            {
                "source": "metaspace",
                "dataset_id": "source-a",
                "directory": str(directory),
                "imzml_path": str(image_path),
                "annotations_present": True,
            }
        ],
    }

    merged_path = compose_cohort(
        workspace_path=workspace,
        cohort_id="cohort",
        source="metaspace",
        dataset_ids=["source-a"],
        unannotated_amount=0,
        max_fdr=0.1,
        composition_manifest=manifest,
    )

    reader = AnnotationReader.load(
        type="merge",
        path=merged_path.with_suffix(".sqlite"),
        image_path=merged_path,
    )
    assert reader.get_annotations()[0]["spectrum_ids"] == [0, 1]
    assert reader.get_spectrum_annotations(1)[0]["mz"] == 181.0707
    assert reader.get_spectrum_metadata(1)["source_spectrum_id"] == 2
