"""Query external dataset metadata without downloading spectra."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from ...utils.logger import get_custom_logger
from ..catalog.sqlite_catalog import DatasetCatalog
from ..sources.base import DatasetSource
from ..validators import validate_source_record


logger = get_custom_logger(__name__)


def query_to_selection(
    *, source: DatasetSource, filters: Mapping[str, Any], catalog: DatasetCatalog,
    selection_path: Path | str,
) -> List[Dict[str, Any]]:
    """Query metadata, update SQLite, and write a reproducible selection."""
    provider_filters = dict(filters)
    excluded_ids = {
        str(value) for value in provider_filters.get("exclude_dataset_ids", ())
    }
    if source.source_name != "metaspace":
        provider_filters.pop("exclude_dataset_ids", None)
    records = [
        record
        for record in source.filter(provider_filters)
        if str(record.get("dataset_id")) not in excluded_ids
    ]
    selected: List[Dict[str, Any]] = []
    for record in records:
        validate_source_record(record)
        dataset_id = str(record["dataset_id"])
        name = str(record.get("name", dataset_id))
        metadata = dict(record.get("metadata", {}))
        catalog.upsert_dataset(
            source=source.source_name, dataset_id=dataset_id, name=name, metadata=metadata
        )
        selected.append({"source": source.source_name, "dataset_id": dataset_id,
                         "name": name, "metadata": metadata})
    target = Path(selection_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"schema_version": 1, "source": source.source_name,
                                  "filters": dict(filters), "datasets": selected},
                                 ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote query selection with %s datasets to %s", len(records), target)
    return selected
