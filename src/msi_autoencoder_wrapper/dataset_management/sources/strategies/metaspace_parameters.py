"""Declared METASPACE filter names and free-text value mappings."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple


# Public wrapper name -> exact ``SMInstance.datasets`` argument. Keeping every
# translation here makes the provider boundary reviewable and prevents an
# unknown wrapper option from being passed to METASPACE by accident.
API_FILTER_FIELDS = {
    "name": "nameMask",
    "dataset_ids": "idMask",
    "submitter_id": "submitter_id",
    "group_id": "group_id",
    "project_id": "project_id",
    "polarity": "polarity",
    "analyzer_type": "analyzer_type",
    "ionisation_source": "ionisation_source",
    "maldi_matrix": "maldi_matrix",
    "status": "status",
}


# Canonical values use PascalCase.  Each tuple explicitly lists every raw
# METASPACE spelling that belongs to the group.  Filtering by either the
# canonical value or any listed raw value selects the complete group.
FREE_TEXT_VALUE_GROUPS = {
    "Wildtype": (
        "Wildtype",
        "Wild type",
        "Wtype",
    ),
    "Mouse": (
        "Mouse",
        "Mus musculus (mouse)",
        "Mus musculus",
    ),
    "Liver": (
        "Liver",
    ),
}


# These are intentionally not grouped: they carry additional biological
# meaning or are ambiguous, rather than being spelling variants.
#
# * ``wildtype_unlabeled``: distinct labelling status.
# * ``wildtype and knock out``: mixed condition.
# * ``Homo sapiens (human) | Mus musculus (mouse)``: mixed species.
# * ``liver and kidney`` and similar values: multi-organ samples.


FREE_TEXT_FILTER_FIELDS = {
    "organism": "organism",
    "organism_part": "organism_part",
    "condition": "condition",
    "growth_conditions": "growth_conditions",
}


LOCAL_FILTERS = {
    "annotation_fdr",
    "include_molecule_stats",
    "include_spatial_annotation_stats",
    "min_annotation_count",
    "max_annotation_count",
    "min_molecule_count",
    "min_unique_molecule_count",
    "has_optical_image",
    *FREE_TEXT_FILTER_FIELDS,
}


def filter_schema() -> Dict[str, Any]:
    """Return the complete public METASPACE filter schema."""
    schema: Dict[str, Any] = {
        key: {"type": "string", "api_field": api_field}
        for key, api_field in API_FILTER_FIELDS.items()
    }
    schema.update(
        {
            "dataset_ids": {
                "type": "string | list[string]",
                "api_field": API_FILTER_FIELDS["dataset_ids"],
            },
            "molecule": {
                "type": "string",
                "api_field": "hasAnnotationMatching.compoundQuery",
            },
            "polarity": {
                "type": "Positive | Negative",
                "api_field": API_FILTER_FIELDS["polarity"],
                "choices": ["Positive", "Negative"],
            },
            "status": {
                "type": "string",
                "api_field": API_FILTER_FIELDS["status"],
                "default": "FINISHED",
            },
            "annotation_fdr": {"type": "float", "default": 0.1, "local": True},
            "min_annotation_count": {"type": "integer | null", "local": True},
            "max_annotation_count": {"type": "integer | null", "local": True},
            "include_molecule_stats": {
                "type": "boolean",
                "default": False,
                "local": True,
            },
            "include_spatial_annotation_stats": {
                "type": "boolean",
                "default": False,
                "local": True,
            },
            "min_molecule_count": {"type": "integer | null", "local": True},
            "min_unique_molecule_count": {"type": "integer | null", "local": True},
            "has_optical_image": {
                "type": "boolean",
                "local": True,
                "choices": [True, False],
            },
            "exclude_dataset_ids": {
                "type": "list[string]",
                "description": "Local exclusions applied after provider discovery.",
            },
        }
    )
    for key in FREE_TEXT_FILTER_FIELDS:
        schema[key] = {
            "type": "string | list[string]",
            "local": True,
            "description": (
                "Matched against explicit PascalCase canonical groups in "
                "FREE_TEXT_VALUE_GROUPS; canonical and raw variants select "
                "the whole group."
            ),
        }
    return schema


def split_filters(filters: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate and separate provider filters from local constraints."""
    unknown = set(filters) - set(filter_schema())
    if unknown:
        raise ValueError(f"Unknown METASPACE filters: {sorted(unknown)}")
    options = dict(filters)
    options.pop("exclude_dataset_ids", None)
    local = {key: options.pop(key) for key in list(options) if key in LOCAL_FILTERS}
    fdr = float(local.get("annotation_fdr", 0.1))
    molecule = options.pop("molecule", None)
    native = {
        API_FILTER_FIELDS[key]: value
        for key, value in options.items()
        if value is not None
    }
    native.setdefault("status", "FINISHED")
    if molecule:
        native["hasAnnotationMatching"] = {
            "compoundQuery": str(molecule),
            "fdrLevel": fdr,
        }
    local.setdefault("annotation_fdr", fdr)
    return native, local


def normalize_free_text(value: Any) -> str:
    """Case-fold and collapse whitespace for exact alias lookup."""
    return " ".join(str(value or "").split()).casefold()


_CANONICAL_BY_VARIANT = {
    normalize_free_text(variant): canonical
    for canonical, variants in FREE_TEXT_VALUE_GROUPS.items()
    for variant in variants
}


def canonical_free_text(value: Any) -> str:
    """Return a declared PascalCase group or the normalized raw value."""
    normalized = normalize_free_text(value)
    return _CANONICAL_BY_VARIANT.get(normalized, normalized)
