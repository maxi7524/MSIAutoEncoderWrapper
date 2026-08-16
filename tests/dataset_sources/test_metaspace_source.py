"""Offline contract tests for the METASPACE source adapter."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from pyimzml.ImzMLWriter import ImzMLWriter

from msi_dataset_manager.exploration.dataset_explorer import (
    DatasetExplorer,
)
from msi_dataset_manager.sources.strategies.metaspace import (
    MetaspaceDatasetSource,
)
from msi_dataset_manager.sources.strategies.metaspace.source import _records_from_table
from msi_dataset_manager.sources.strategies.metaspace.metadata import (
    attach_api_metadata,
    summarise_records,
)
from msi_dataset_manager.sources.strategies.metaspace.parameters import (
    FREE_TEXT_VALUE_GROUPS,
    split_filters,
)
from msi_dataset_manager.utils.exceptions import (
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
        self.download_calls: List[tuple[Path, str]] = []

    def isotope_images(self, *args: Any, **kwargs: Any) -> List[np.ndarray]:
        assert kwargs["only_first_isotope"] is True
        assert kwargs["scale_intensity"] is True
        assert kwargs["hotspot_clipping"] is True
        return [np.array([[1.0, 0.0], [0.0, 0.0]])]

    def results(self, **kwargs: Any) -> List[Dict[str, Any]]:
        self.result_calls.append(kwargs)
        return [
            {
                "id": "one",
                "sumFormula": "C6H12O6",
                "adduct": "+H",
                "mz": 181.0707,
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
        self.download_calls.append((path, base_name))
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{base_name}.imzML").write_text("imzml", encoding="utf-8")
        (path / f"{base_name}.ibd").write_bytes(b"ibd")


class FakeMetaspaceClient:
    """Capture discovery filters and return a stable fake dataset."""

    def __init__(self) -> None:
        self.value = FakeMetaspaceDataset()
        self.filters: Dict[str, Any] = {}
        self._gqclient = FakeGraphQLClient()
        self.value._gqclient = self._gqclient

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

    def __init__(self) -> None:
        self.annotation_filters: List[Dict[str, Any]] = []

    def listQuery(
        self,
        field_name: str,
        query: str,
        variables: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        assert field_name == "allDatasets"
        if "datasetMetadata" in query:
            return [{"id": "dataset-a"}]
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
        if "annotationFilter" in kwargs:
            self.annotation_filters.append(dict(kwargs["annotationFilter"]))
            return [
                {
                    "dataset": {"id": "dataset-a", "name": "Dataset A"},
                    "sumFormula": "C6H12O6",
                    "adduct": "+H",
                    "ion": "C6H12O6+H",
                    "mz": 181.0707,
                    "fdrLevel": 0.1,
                    "possibleCompounds": [
                        {
                            "name": "Glucose",
                            "information": [{"databaseId": 1}],
                        }
                    ],
                    "isotopeImages": [
                        {
                            "mz": 181.0707,
                            "url": "https://example.invalid/image",
                            "minIntensity": 0.0,
                            "maxIntensity": 1.0,
                            "totalIntensity": 1.0,
                        }
                    ],
                }
            ]
        return [
            {
                "dataset": {"id": "dataset-a"},
                "sumFormula": "C6H12O6",
                "adduct": "+H",
            }
        ]


def test_declared_filter_mapping_is_explicit_and_rejects_unknown_keys() -> None:
    """Canonical values and provider field names have one declared source."""
    native, local = split_filters({"name": "liver", "condition": "Wild type"})

    assert native == {"nameMask": "liver", "status": "FINISHED"}
    assert local == {"condition": "Wild type", "annotation_fdr": 0.1}
    assert FREE_TEXT_VALUE_GROUPS["Wildtype"] == ("Wildtype", "Wild type", "Wtype")
    with pytest.raises(ValueError, match="Unknown METASPACE filters"):
        split_filters({"typo": "value"})


def test_mz_bounds_are_normalized_to_an_early_coverage_filter() -> None:
    """Notebook-friendly bounds produce the canonical range configuration."""
    native, local = split_filters({"mz_min": 200, "mz_max": 900})

    assert native == {"status": "FINISHED"}
    assert local["mz_range"] == {"min": 200.0, "max": 900.0, "mode": "covers"}
    with pytest.raises(ValueError, match="provided together"):
        split_filters({"mz_min": 200})


def test_api_metadata_contains_sizes_settings_pixels_and_mass_range() -> None:
    """Public GraphQL fields reproduce the annotation-settings panel and m/z range."""
    class MetadataGraph:
        def listQuery(self, field_name: str, query: str, variables: Dict[str, Any]) -> List[Dict[str, Any]]:
            assert "diagnostics { type data }" in query
            assert "error" not in query
            return [
                {
                    "id": "dataset-a",
                    "configJson": json.dumps({"image_generation": {"ppm": 5}, "analysis_version": 1}),
                    "sizeHash": json.dumps({"imzml_size": 80_058_778, "ibd_size": 314_981_417}),
                    "acquisitionGeometry": json.dumps({"pixel_count": 32_576}),
                    "diagnostics": [
                        {
                            "type": "IMZML_METADATA",
                            "data": json.dumps(
                                {
                                    "min_mz": 100.0,
                                    "max_mz": 1000.0,
                                    "n_spectra": 32_576,
                                }
                            ),
                        }
                    ],
                }
            ]

    records = [{"dataset_id": "dataset-a", "metadata": {}}]
    attach_api_metadata(SimpleNamespace(_gqclient=MetadataGraph()), records)
    metadata = records[0]["metadata"]

    assert metadata["mz_tolerance_ppm"] == 5.0
    assert metadata["analysis_method"] == "Original MSM"
    assert metadata["pixel_count"] == 32_576
    assert metadata["total_size_bytes"] == 395_040_195
    assert (metadata["mz_min"], metadata["mz_max"]) == (100.0, 1000.0)


def test_metaspace_summary_intersects_mass_ranges_and_unions_methods() -> None:
    """The summary reports cohort storage, common m/z axis and all methods."""
    summary = summarise_records(
        [
            {"metadata": {"total_size_bytes": 100, "mz_min": 50, "mz_max": 900, "analysis_method": "Original MSM", "ionisation_source": "MALDI", "analyzer": {"type": "Orbitrap"}}},
            {"metadata": {"total_size_bytes": 200, "mz_min": 100, "mz_max": 1000, "analysis_method": "METASPACE-ML", "ionisation_source": "DESI", "analyzer": {"type": "Orbitrap"}}},
        ]
    )

    assert summary == {
        "dataset_count": 2,
        "total_size_bytes": 300,
        "size_complete": True,
        "mz_intersection_min": 100.0,
        "mz_intersection_max": 900.0,
        "analysis_methods": ["METASPACE-ML", "Original MSM"],
        "measurement_methods": ["DESI", "MALDI", "Orbitrap"],
    }


def test_explorer_appends_summary_only_when_requested() -> None:
    """The default table stays unchanged and ``summarise=True`` adds a final row."""
    explorer = DatasetExplorer(MetaspaceDatasetSource(client=FakeMetaspaceClient()))

    ordinary = explorer.search()
    summarised = explorer.results(summarise=True)

    assert ordinary["dataset_id"].tolist() == ["dataset-a"]
    assert summarised["dataset_id"].tolist() == ["dataset-a", "SUMMARY"]
    assert summarised.iloc[-1]["name"] == "1 datasets (partial size data)"


def test_metaspace_adapter_preserves_metadata_and_imports_broad_annotations(
    tmp_path: Path,
) -> None:
    """The adapter passes native filters and adds database provenance."""
    client = FakeMetaspaceClient()
    source = MetaspaceDatasetSource(client=client)

    records = source.filter(
        {"organism": "MOUSE", "include_molecule_stats": True}
    )
    destination = source.download_dataset("dataset-a", tmp_path / "dataset-a")
    with ImzMLWriter(str(destination / "dataset-a.imzML"), mode="processed") as writer:
        for y in (1, 2):
            for x in (1, 2):
                writer.addSpectrum(
                    np.asarray([181.0707]),
                    np.asarray([1.0]),
                    (x, y, 1),
                )
    source.materialize_annotations(
        dataset_id="dataset-a",
        dataset_name="Dataset A",
        directory=destination,
        imzml_path=destination / "dataset-a.imzML",
    )
    annotations = source.read_annotation_export(
        dataset_id="dataset-a",
        directory=destination,
        imzml_path=destination / "dataset-a.imzML",
    ).records

    # "organism" is unstructured free text on METASPACE (e.g. "Mus musculus
    # (mouse)" vs "Mouse" vs "mouse"), so it is matched locally and
    # case-insensitively and the cache restricts the API query to matching IDs.
    assert client.filters == {"idMask": ["dataset-a"], "status": "FINISHED"}
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
    assert client._gqclient.annotation_filters[-1]["databaseId"] == 1
    assert client.value.result_calls == []
    assert annotations[0]["spectrum_values"] == {0: 1.0}
    assert client.value.download_calls == [(destination, "dataset-a")]
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


def test_download_only_source_does_not_load_discovery_catalogue() -> None:
    """Frozen selections avoid a full catalogue query for every API key."""
    class DownloadOnlyClient:
        def datasets(self, **filters: Any) -> List[Any]:
            raise AssertionError("download-only construction queried the catalogue")

    source = MetaspaceDatasetSource(
        client=DownloadOnlyClient(),
        load_catalog=False,
    )

    assert source._available_values_cache == []


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

    assert values == [
        {"value": "Mouse", "label": "Mouse", "count": 1, "variants": "mouse (1)"}
    ]
    assert client.filters == {"status": "FINISHED"}
    with pytest.raises(ValueError, match="free text or quantitative"):
        source.get_available_values("molecule")


def test_available_values_group_case_only_organism_part_spelling_variants() -> None:
    """'Root' and 'root' collapse into one row through `_canonical_free_text`."""

    class MultiDatasetClient(FakeMetaspaceClient):
        def datasets(self, **filters: Any) -> List[FakeMetaspaceDataset]:
            self.filters = filters
            first = FakeMetaspaceDataset()
            first.id = "dataset-a"
            first.metadata = {"Sample_Information": {"Organism_Part": "Root"}}
            second = FakeMetaspaceDataset()
            second.id = "dataset-b"
            second.metadata = {"Sample_Information": {"Organism_Part": "root"}}
            return [first, second]

    source = MetaspaceDatasetSource(client=MultiDatasetClient())

    values = source.get_available_values("organism_part")

    assert values == [
        {
            "value": "Root",
            "label": "Root",
            "count": 2,
            "variants": "Root (1), root (1)",
        }
    ]


def test_condition_filter_tolerates_metaspace_free_text_spelling_variants() -> None:
    """'Wild type' must match a dataset whose submitter typed 'Wildtype'."""
    client = FakeMetaspaceClient()
    client.value.metadata = {"organism": "mouse", "condition": "Wildtype"}
    source = MetaspaceDatasetSource(client=client)

    records = source.filter({"condition": "Wild type"})

    assert [record["dataset_id"] for record in records] == ["dataset-a"]
    assert client.filters == {"idMask": ["dataset-a"], "status": "FINISHED"}


def test_condition_filter_rejects_unrelated_free_text_values() -> None:
    """Tolerant matching must not merge genuinely different condition labels."""
    client = FakeMetaspaceClient()
    client.value.metadata = {"organism": "mouse", "condition": "Diseased"}
    source = MetaspaceDatasetSource(client=client)

    records = source.filter({"condition": "Wild type"})

    assert records == []
    rejected = source.get_rejected_records()
    assert len(rejected) == 1
    assert "condition" in rejected[0]["reason"]


def test_organism_filter_accepts_a_list_of_synonyms() -> None:
    """A list of query terms is OR-matched against the free-text field."""
    client = FakeMetaspaceClient()
    client.value.metadata = {"organism": "Mus musculus (mouse)", "condition": "n/a"}
    source = MetaspaceDatasetSource(client=client)

    records = source.filter({"organism": ["Rat", "Mouse"]})

    assert [record["dataset_id"] for record in records] == ["dataset-a"]


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


def test_download_transfers_the_validated_links_without_requesting_them_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quota-sensitive signed links are fetched exactly once per dataset."""
    class SignedLinkDataset(FakeMetaspaceDataset):
        def __init__(self) -> None:
            super().__init__()
            self.link_calls = 0

        def download_links(self) -> Dict[str, Any]:
            self.link_calls += 1
            if self.link_calls > 1:
                return {
                    "files": [
                        {
                            "filename": "Download_Limit_Reached.txt",
                            "link": "https://example.invalid/quota",
                        }
                    ]
                }
            return {
                "files": [
                    {"filename": "source.imzml", "link": "https://example/imzml"},
                    {"filename": "source.ibd", "link": "https://example/ibd"},
                ]
            }

    class Response:
        headers = {"content-length": "4"}

        def __init__(self, url: str) -> None:
            self.content = b"xml!" if url.endswith("imzml") else b"ibd!"

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int) -> List[bytes]:
            return [self.content]

    dataset = SignedLinkDataset()
    client = FakeMetaspaceClient()
    client.value = dataset
    monkeypatch.setattr(
        "msi_dataset_manager.sources.strategies.metaspace.source.requests.get",
        lambda url, stream: Response(url),
    )

    destination = MetaspaceDatasetSource(client=client).download_dataset(
        "dataset-a", tmp_path / "dataset-a"
    )

    assert dataset.link_calls == 1
    assert (destination / "dataset-a.imzML").read_bytes() == b"xml!"
    assert (destination / "dataset-a.ibd").read_bytes() == b"ibd!"
    assert list(destination.glob("*.part")) == []


def test_spatial_retrieval_rejects_annotations_without_first_isotope_data() -> None:
    """Requested pixel annotations cannot silently lose molecular results."""
    class MissingImageGraphQLClient(FakeGraphQLClient):
        def getAnnotations(self, **kwargs: Any) -> List[Dict[str, Any]]:
            records = super().getAnnotations(**kwargs)
            if "annotationFilter" in kwargs:
                records[0]["isotopeImages"] = []
            return records

    client = FakeMetaspaceClient()
    client._gqclient = MissingImageGraphQLClient()
    client.value._gqclient = client._gqclient
    source = MetaspaceDatasetSource(client=client)

    with pytest.raises(ExternalServiceError, match="no first-isotope image"):
        source.materialize_annotations(
            dataset_id="dataset-a",
            dataset_name="Dataset A",
            directory=Path("unused"),
            imzml_path=Path("unused.imzML"),
            options={"annotation_fdr": 0.1, "include_spatial": True},
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
