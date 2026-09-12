"""Normalized local metadata artifacts for source datasets and cohorts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .layout import DatasetWorkspaceLayout


METADATA_SCHEMA_VERSION = 1


def normalize_dataset_metadata(
    *,
    source: str,
    dataset_id: str,
    name: str,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Normalize provider metadata without discarding provider-specific fields.

    :param source: Registered source identifier.
    :type source: str
    :param dataset_id: Stable provider dataset identifier.
    :type dataset_id: str
    :param name: Human-readable dataset name.
    :type name: str
    :param metadata: Provider metadata from a frozen selection or local export.
    :type metadata: Mapping[str, Any] | None
    :return: Versioned normalized dataset metadata artifact.
    :rtype: dict[str, Any]
    """
    raw = dict(metadata or {})
    analyzer = raw.get("analyzer")
    analyzer_mapping = analyzer if isinstance(analyzer, Mapping) else {}
    canonical = {
        "polarity": raw.get("polarity"),
        "organism": raw.get("organism"),
        "organism_part": raw.get("organism_part"),
        "condition": raw.get("condition"),
        "growth_conditions": raw.get("growth_conditions"),
        "analyzer_type": raw.get("analyzer_type") or analyzer_mapping.get("type"),
        "ionisation_source": raw.get("ionisation_source"),
        "maldi_matrix": raw.get("maldi_matrix"),
        "mz_min": raw.get("mz_min"),
        "mz_max": raw.get("mz_max"),
        "mz_tolerance_ppm": raw.get("mz_tolerance_ppm"),
        "databases": raw.get("databases", ()),
    }
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "source": str(source),
        "dataset_id": str(dataset_id),
        "name": str(name),
        "metadata": canonical,
        "provider_metadata": raw,
    }


def write_dataset_metadata(
    *,
    workspace_path: Path | str,
    source: str,
    dataset_id: str,
    name: str,
    metadata: Mapping[str, Any] | None,
) -> Path:
    """Write one normalized source-dataset metadata artifact atomically."""
    layout = DatasetWorkspaceLayout(workspace_path)
    target = layout.dataset_metadata_path(dataset_id)
    artifact = {
        **normalize_dataset_metadata(
            source=source,
            dataset_id=dataset_id,
            name=name,
            metadata=metadata,
        ),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(target, artifact)
    return target


def read_dataset_metadata(
    *,
    workspace_path: Path | str,
    dataset_id: str,
) -> dict[str, Any]:
    """Read one current normalized source-dataset metadata artifact.

    :raises ValueError: If the artifact is absent or uses another schema.
    """
    target = DatasetWorkspaceLayout(workspace_path).dataset_metadata_path(dataset_id)
    if not target.is_file():
        raise ValueError(f"Dataset metadata does not exist: '{target}'.")
    value = json.loads(target.read_text(encoding="utf-8"))
    if value.get("schema_version") != METADATA_SCHEMA_VERSION:
        raise ValueError("Unsupported dataset metadata schema version.")
    return value


def write_cohort_metadata(
    *,
    workspace_path: Path | str,
    cohort_id: str,
    datasets: Sequence[Mapping[str, Any]],
) -> Path:
    """Write source metadata retained by a composed cohort atomically."""
    layout = DatasetWorkspaceLayout(workspace_path)
    target = layout.cohort_metadata_path(cohort_id)
    normalized = [dict(dataset) for dataset in datasets]
    _write_json_atomic(
        target,
        {
            "schema_version": METADATA_SCHEMA_VERSION,
            "cohort_id": str(cohort_id),
            "written_at": datetime.now(timezone.utc).isoformat(),
            "datasets": normalized,
            "metadata": _cohort_metadata(normalized),
        },
    )
    return target


def read_candidate_metadata(
    *,
    workspace_path: Path | str,
    dataset_id: str,
) -> dict[str, Any]:
    """Read canonical metadata for either a source dataset or a cohort."""
    layout = DatasetWorkspaceLayout(workspace_path)
    dataset_path = layout.dataset_metadata_path(dataset_id)
    if dataset_path.is_file():
        return read_dataset_metadata(workspace_path=workspace_path, dataset_id=dataset_id)
    cohort_path = layout.cohort_metadata_path(dataset_id)
    if not cohort_path.is_file():
        raise ValueError(f"No normalized metadata exists for dataset '{dataset_id}'.")
    artifact = json.loads(cohort_path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != METADATA_SCHEMA_VERSION:
        raise ValueError("Unsupported cohort metadata schema version.")
    return artifact


def _cohort_metadata(datasets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive metadata usable for one candidate catalogue over a cohort."""
    values = [dict(item.get("metadata", {})) for item in datasets]
    polarities = {item.get("polarity") for item in values if item.get("polarity")}
    minima = [float(item["mz_min"]) for item in values if item.get("mz_min") is not None]
    maxima = [float(item["mz_max"]) for item in values if item.get("mz_max") is not None]
    return {
        "polarity": next(iter(polarities)) if len(polarities) == 1 else None,
        "mz_min": max(minima) if len(minima) == len(values) and values else None,
        "mz_max": min(maxima) if len(maxima) == len(values) and values else None,
        "dataset_count": len(datasets),
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Write one JSON mapping atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
