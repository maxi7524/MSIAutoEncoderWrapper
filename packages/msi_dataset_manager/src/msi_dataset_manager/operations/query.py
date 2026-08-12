"""Query external dataset metadata without downloading spectra."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from ..utils.logger import get_custom_logger
from ..sources.base import DatasetSource
from ..validators import validate_source_record


logger = get_custom_logger(__name__)


def query_to_selection(
    *, source: DatasetSource, filters: Mapping[str, Any],
    selection_path: Path | str,
) -> List[Dict[str, Any]]:
    """Query provider metadata and write a reproducible JSON selection.

    :param source: Provider adapter used only for metadata discovery.
    :type source: DatasetSource
    :param filters: Provider and local selection criteria.
    :type filters: Mapping[str, Any]
    :param selection_path: Destination ``selection.json`` path.
    :type selection_path: pathlib.Path | str
    :return: Validated records written to the selection.
    :rtype: List[Dict[str, Any]]

    Query does not create operational state. The selection freezes provider
    results and metadata; download state belongs to the freshly regenerated
    materialization manifest, while SQLite belongs exclusively to composition.
    """
    # Provider discovery
    ## Keep generic exclusions outside adapters that do not implement them.
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
        selected.append(
            {
                "source": source.source_name,
                "dataset_id": dataset_id,
                "name": name,
                "metadata": metadata,
            }
        )

    # Frozen selection artifact
    ## This file contains criteria and provider results, never download status.
    target = Path(selection_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": source.source_name,
                "filters": dict(filters),
                "datasets": selected,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote query selection with %s datasets to %s", len(records), target)
    return selected
