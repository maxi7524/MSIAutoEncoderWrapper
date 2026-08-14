"""Build one merged MSI cohort from locally materialized source datasets."""

from .annotations import build_cohort_annotation_index
from .compose import compose_cohort, create_composition_manifest

__all__ = [
    "build_cohort_annotation_index",
    "compose_cohort",
    "create_composition_manifest",
]
