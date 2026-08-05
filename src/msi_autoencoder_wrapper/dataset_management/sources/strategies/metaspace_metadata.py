"""METASPACE GraphQL metadata enrichment and cohort summaries."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional

from ....utils.logger import get_custom_logger


logger = get_custom_logger(__name__)


_DETAIL_QUERY = """
    query datasetMetadata($filter: DatasetFilter, $offset: Int, $limit: Int) {
      allDatasets(filter: $filter, offset: $offset, limit: $limit) {
        id
        configJson
        sizeHash
        acquisitionGeometry
        diagnostics { type data }
      }
    }
"""


def attach_api_metadata(client: Any, records: List[Dict[str, Any]]) -> None:
    """Attach public configuration, size and m/z diagnostics in place."""
    graph = getattr(client, "_gqclient", None)
    if graph is None or not hasattr(graph, "listQuery") or not records:
        return
    details: List[Dict[str, Any]] = []
    for start in range(0, len(records), 100):
        dataset_ids = "|".join(
            str(record["dataset_id"])
            for record in records[start : start + 100]
        )
        details.extend(
            graph.listQuery(
                "allDatasets",
                _DETAIL_QUERY,
                {"filter": {"ids": dataset_ids}},
            )
        )
    by_id = {str(detail["id"]): detail for detail in details}
    for record in records:
        detail = by_id.get(str(record["dataset_id"]))
        if detail is not None:
            values = _normalise_detail(detail)
            record["metadata"].update(
                {key: value for key, value in values.items() if value is not None}
            )


def _normalise_detail(detail: Mapping[str, Any]) -> Dict[str, Any]:
    config = _json_mapping(detail.get("configJson"))
    size_hash = _json_mapping(detail.get("sizeHash"))
    geometry = _json_mapping(detail.get("acquisitionGeometry"))
    diagnostic = next(
        (
            item
            for item in detail.get("diagnostics") or []
            if item.get("type") == "IMZML_METADATA"
        ),
        {},
    )
    diagnostic_data = _mapping(diagnostic.get("data"))
    image_generation = _mapping(config.get("image_generation"))
    analysis_version = config.get("analysis_version")
    imzml_size = _optional_int(size_hash.get("imzml_size"))
    ibd_size = _optional_int(size_hash.get("ibd_size"))
    sizes = [value for value in (imzml_size, ibd_size) if value is not None]
    pixel_count = _optional_int(geometry.get("pixel_count"))
    if pixel_count is None:
        pixel_count = _optional_int(diagnostic_data.get("n_spectra"))
    return {
        "mz_tolerance_ppm": _optional_float(image_generation.get("ppm")),
        "analysis_version": analysis_version,
        "analysis_method": _analysis_method(analysis_version),
        "pixel_count": pixel_count,
        "imzml_size_bytes": imzml_size,
        "ibd_size_bytes": ibd_size,
        "total_size_bytes": sum(sizes) if sizes else None,
        "download_size_bytes": sum(sizes) if sizes else None,
        "mz_min": _optional_float(diagnostic_data.get("min_mz")),
        "mz_max": _optional_float(diagnostic_data.get("max_mz")),
    }


def summarise_records(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate storage, m/z intersection and measurement methods."""
    metadata = [_mapping(record.get("metadata")) for record in records]
    sizes = [item.get("total_size_bytes") for item in metadata]
    minima = [
        float(item["mz_min"])
        for item in metadata
        if item.get("mz_min") is not None
    ]
    maxima = [
        float(item["mz_max"])
        for item in metadata
        if item.get("mz_max") is not None
    ]
    methods = sorted(
        {str(item["analysis_method"]) for item in metadata if item.get("analysis_method")}
    )
    measurement_methods = sorted(
        {
            value
            for item in metadata
            for value in _measurement_method_values(item)
        }
    )
    intersection_min = max(minima) if len(minima) == len(records) and records else None
    intersection_max = min(maxima) if len(maxima) == len(records) and records else None
    if (
        intersection_min is not None
        and intersection_max is not None
        and intersection_min > intersection_max
    ):
        intersection_min = intersection_max = None
    return {
        "dataset_count": len(records),
        "total_size_bytes": sum(int(size) for size in sizes if size is not None),
        "size_complete": bool(records) and all(size is not None for size in sizes),
        "mz_intersection_min": intersection_min,
        "mz_intersection_max": intersection_max,
        "analysis_methods": methods,
        "measurement_methods": measurement_methods,
    }


def _json_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return _mapping(parsed)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _analysis_method(analysis_version: Any) -> Optional[str]:
    if analysis_version is None:
        return None
    return "Original MSM" if analysis_version == 1 else "METASPACE-ML"


def _measurement_method_values(metadata: Mapping[str, Any]) -> List[str]:
    values: List[str] = []
    ionisation_source = metadata.get("ionisation_source")
    if ionisation_source:
        values.append(str(ionisation_source))
    analyzer = _mapping(metadata.get("analyzer"))
    if analyzer.get("type"):
        values.append(str(analyzer["type"]))
    return values


def _optional_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
