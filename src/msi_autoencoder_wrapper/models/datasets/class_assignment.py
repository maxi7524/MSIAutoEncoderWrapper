"""Deterministic class assignment helpers for dataset metadata and molecules."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence


def build_class_mapping(
    values: Iterable[str],
    explicit_mapping: Mapping[str, int] | None = None,
) -> Dict[str, int]:
    """Return a stable class mapping for semantic annotation values.

    :param values: Semantic values available in the local dataset.
    :type values: Iterable[str]
    :param explicit_mapping: Optional user-defined value-to-index mapping.
    :type explicit_mapping: Mapping[str, int] | None
    :return: Explicit mapping copy or alphabetically ordered generated mapping.
    :rtype: Dict[str, int]

    No class zero or unknown class is injected implicitly. Unknown-value policy
    belongs to the experiment configuration and can be added explicitly later.
    """
    if explicit_mapping is not None:
        return {str(value): int(index) for value, index in explicit_mapping.items()}
    return {
        value: index
        for index, value in enumerate(sorted({str(value) for value in values}))
    }


def metadata_values(metadata: Mapping[str, Any], field: str) -> Sequence[str]:
    """Extract all non-null values of one field from source dataset metadata.

    :param metadata: One source record or a merged metadata record.
    :type metadata: Mapping[str, Any]
    :param field: Metadata field selected as a model target.
    :type field: str
    :return: Values in source order.
    :rtype: Sequence[str]
    """
    records = metadata.get("datasets", metadata.get("sources"))
    if records is None:
        records = [metadata]
    values = []
    for record in records:
        if not record:
            continue
        source_metadata = record.get("metadata", record)
        value = source_metadata.get(field)
        if isinstance(value, (list, tuple, set)):
            values.extend(str(item) for item in value if item is not None)
        elif value is not None:
            values.append(str(value))
    return values


def molecule_key(annotation: Mapping[str, Any]) -> str:
    """Return the canonical formula/adduct label for one molecular annotation."""
    formula = annotation.get("formula", annotation.get("sumFormula", ""))
    adduct = annotation.get("adduct", "")
    return f"{formula}|{adduct}"
