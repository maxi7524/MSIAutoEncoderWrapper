"""Offline contract tests for the METASPACE source adapter."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from msi_autoencoder_wrapper.dataset_management.sources.strategies.metaspace import (
    MetaspaceDatasetSource,
    _records_from_table,
)
from msi_autoencoder_wrapper.utils.exceptions import (
    DownloadLimitError,
    ExternalServiceError,
)


class FakeMetaspaceDataset:
    """Minimal object matching the official client's dataset surface."""

    id = "dataset-a"
    name = "Dataset A"
    metadata = {"organism": "mouse", "condition": "healthy"}
    polarity = "Positive"
    status = "FINISHED"
    image_size = {"x": 2, "y": 2}
    database_details = [SimpleNamespace(id=1, name="HMDB", version="v4")]

    def __init__(self) -> None:
        self.result_calls: List[Dict[str, Any]] = []

    def results(self, **kwargs: Any) -> List[Dict[str, Any]]:
        self.result_calls.append(kwargs)
        return [
            {
                "id": "one",
                "sumFormula": "C6H12O6",
                "adduct": "+H",
                "fdr": 0.1,
            }
        ]

    def all_annotation_images(self, **kwargs: Any) -> List[Any]:
        images = AnnotationImages(
            [np.array([[1.0, 0.0], [0.0, 0.0]])],
            formula="C6H12O6",
            adduct="+H",
        )
        return [images]

    def download_to_dir(self, path: Path, base_name: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{base_name}.imzML").write_text("imzml", encoding="utf-8")
        (path / f"{base_name}.ibd").write_bytes(b"ibd")


class FakeMetaspaceClient:
    """Capture discovery filters and return a stable fake dataset."""

    def __init__(self) -> None:
        self.value = FakeMetaspaceDataset()
        self.filters: Dict[str, Any] = {}
        self._gqclient = FakeGraphQLClient()

    def datasets(self, **filters: Any) -> List[FakeMetaspaceDataset]:
        self.filters = filters
        return [self.value]

    def dataset(self, id: str) -> FakeMetaspaceDataset:
        assert id == self.value.id
        return self.value


class AnnotationImages(list):
    """List-like METASPACE image group carrying molecule identity."""

    def __init__(self, values: List[Any], *, formula: str, adduct: str) -> None:
        super().__init__(values)
        self.formula = formula
        self.adduct = adduct


class FakeGraphQLClient:
    """Return aggregate metadata without ion images or binary downloads."""

    def listQuery(
        self,
        field_name: str,
        query: str,
        variables: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        assert field_name == "allDatasets"
        assert variables["fdrLevels"] == [10]
        return [
            {
                "id": "dataset-a",
                "opticalImage": "optical-image-id",
                "annotationCounts": [
                    {
                        "databaseId": 1,
                        "dbName": "HMDB",
                        "dbVersion": "v4",
                        "counts": [{"level": 10, "n": 12}],
                        "isTargeted": False,
                        "total": 20,
                    }
                ],
            }
        ]

    def getAnnotations(self, **kwargs: Any) -> List[Dict[str, Any]]:
        return [
            {
                "dataset": {"id": "dataset-a"},
                "sumFormula": "C6H12O6",
                "adduct": "+H",
            }
        ]


def test_metaspace_adapter_preserves_metadata_and_imports_broad_annotations(
    tmp_path: Path,
) -> None:
    """The adapter passes native filters and adds database provenance."""
    client = FakeMetaspaceClient()
    source = MetaspaceDatasetSource(client=client)

    records = source.filter(
        {"organism": "Mus musculus", "include_molecule_stats": True}
    )
    annotations = source.get_annotations("dataset-a")
    destination = source.download_dataset("dataset-a", tmp_path / "dataset-a")

    assert client.filters == {
        "organism": "Mus musculus",
        "status": "FINISHED",
    }
    assert records[0]["metadata"]["condition"] == "healthy"
    assert records[0]["metadata"]["databases"] == [
        {"name": "HMDB", "version": "v4", "id": 1}
    ]
    assert records[0]["metadata"]["annotation_count"] == 12
    assert records[0]["metadata"]["molecule_count"] == 1
    assert records[0]["metadata"]["unique_molecule_count"] == 1
    assert records[0]["metadata"]["has_optical_image"] is True
    assert annotations[0]["database_name"] == "HMDB"
    assert annotations[0]["database_version"] == "v4"
    assert client.value.result_calls == [{"database": ("HMDB", "v4"), "fdr": 0.1}]
    assert annotations[0]["ion_image"] == [[1.0, 0.0], [0.0, 0.0]]
    assert (destination / "dataset-a.imzML").is_file()


def test_metaspace_dataframe_index_is_preserved_as_molecule_identity() -> None:
    """Formula and adduct indices from the official client survive normalization."""
    table = pd.DataFrame(
        [{"fdr": 0.1, "intensity": 2.0}],
        index=pd.MultiIndex.from_tuples(
            [("C6H12O6", "+H")], names=["formula", "adduct"]
        ),
    )

    assert _records_from_table(table) == [
        {"formula": "C6H12O6", "adduct": "+H", "fdr": 0.1, "intensity": 2.0}
    ]


def test_spatial_statistics_count_each_annotated_pixel_once() -> None:
    """Overlapping molecular images contribute one count per spatial position."""
    class SpatialDataset(FakeMetaspaceDataset):
        database_details = [
            SimpleNamespace(id=1, name="HMDB", version="v4"),
            SimpleNamespace(id=2, name="LipidMaps", version="2024"),
        ]
        _info = {
            "acquisitionGeometry": json.dumps(
                {
                    "acquisition_grid": {"count_x": 2, "count_y": 2},
                    "pixel_count": 4,
                }
            )
        }

        def all_annotation_images(self, **kwargs: Any) -> List[List[np.ndarray]]:
            assert kwargs["fdr"] == 0.1
            assert kwargs["only_first_isotope"] is True
            assert kwargs["scale_intensity"] is False
            if kwargs["database"] == ("HMDB", "v4"):
                return [
                    [np.array([[2.0, 0.0], [0.0, 0.0]])],
                    [np.array([[0.0, 8.0], [0.0, 0.0]])],
                ]
            return [
                [np.array([[0.0, 3.0], [0.0, 0.0]])],
                [np.array([[0.0, 0.0], [5.0, 0.0]])],
            ]

    client = FakeMetaspaceClient()
    client.value = SpatialDataset()
    source = MetaspaceDatasetSource(client=client)

    records = source.filter(
        {
            "dataset_ids": ["dataset-a"],
            "annotation_fdr": 0.1,
            "include_spatial_annotation_stats": True,
        }
    )

    metadata = records[0]["metadata"]
    assert metadata["annotated_pixel_count"] == 3
    assert metadata["unannotated_pixel_count"] == 1
    assert metadata["annotated_pixel_fraction"] == 0.75
    assert metadata["spatial_annotation_count"] == 4
    assert metadata["spatial_annotation_database_count"] == 2
    assert metadata["annotation_fdr"] == 0.1
    assert metadata["spatial_stats_status"] == "complete"


def test_metaspace_available_values_are_derived_from_dataset_metadata() -> None:
    """Explorer choices include values currently present in METASPACE metadata."""
    client = FakeMetaspaceClient()
    source = MetaspaceDatasetSource(client=client)

    values = source.get_available_values("organism")

    assert values == [{"value": "mouse", "label": "mouse", "count": 1}]
    assert client.filters == {"status": "FINISHED"}
    with pytest.raises(ValueError, match="free text or quantitative"):
        source.get_available_values("molecule")


def test_download_reports_metaspace_access_message(tmp_path: Path) -> None:
    """Missing signed links produce the actionable METASPACE response."""
    class RestrictedDataset(FakeMetaspaceDataset):
        def download_links(self) -> Dict[str, Any]:
            return {"message": "Please Sign in to download."}

    client = FakeMetaspaceClient()
    client.value = RestrictedDataset()
    source = MetaspaceDatasetSource(client=client)

    with pytest.raises(ExternalServiceError, match="Please Sign in to download"):
        source.download_dataset("dataset-a", tmp_path / "dataset-a")


def test_download_limit_sentinel_has_a_dedicated_error(tmp_path: Path) -> None:
    """METASPACE quota responses are detected before transfer threads start."""
    class LimitedDataset(FakeMetaspaceDataset):
        def download_links(self) -> Dict[str, Any]:
            return {
                "files": [
                    {
                        "filename": "Download_Limit_Reached.txt",
                        "link": "https://example.invalid/quota-message",
                    }
                ]
            }

    client = FakeMetaspaceClient()
    client.value = LimitedDataset()
    source = MetaspaceDatasetSource(client=client)

    with pytest.raises(DownloadLimitError, match="quota has been reached"):
        source.download_dataset("dataset-a", tmp_path / "dataset-a")
    assert list((tmp_path / "dataset-a").iterdir()) == []


def test_spatial_retrieval_rejects_annotations_without_ion_images() -> None:
    """Requested pixel annotations cannot silently lose molecular results."""
    class MissingImageDataset(FakeMetaspaceDataset):
        def all_annotation_images(self, **kwargs: Any) -> List[Any]:
            return []

    client = FakeMetaspaceClient()
    client.value = MissingImageDataset()
    source = MetaspaceDatasetSource(client=client)

    with pytest.raises(ExternalServiceError, match="without matching ion images"):
        source.get_annotations(
            "dataset-a",
            {"annotation_fdr": 0.1, "include_spatial": True},
        )


def test_existing_complete_download_is_reused_without_contacting_dataset(
    tmp_path: Path,
) -> None:
    """A non-empty local pair bypasses METASPACE download requests."""
    class NoDownloadClient(FakeMetaspaceClient):
        def dataset(self, id: str) -> FakeMetaspaceDataset:
            raise AssertionError("Existing local files must bypass the provider.")

    destination = tmp_path / "dataset-a"
    destination.mkdir()
    (destination / "dataset-a.imzML").write_text("existing", encoding="utf-8")
    (destination / "dataset-a.ibd").write_bytes(b"existing")

    source = MetaspaceDatasetSource(client=NoDownloadClient())

    assert source.download_dataset("dataset-a", destination) == destination
