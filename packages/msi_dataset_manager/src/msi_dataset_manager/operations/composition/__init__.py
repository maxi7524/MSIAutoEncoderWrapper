"""Build one merged MSI cohort from locally materialized source datasets."""

from .compose import compose_cohort, create_composition_manifest

__all__ = [
    "compose_cohort",
    "create_composition_manifest",
]
