"""Public dataset-management operations."""

from .download import download_from_manifest
from .composition import compose_cohort
from .query import query_to_selection
from .split import grouped_dataset_split

__all__ = [
    "grouped_dataset_split",
    "download_from_manifest",
    "compose_cohort",
    "query_to_selection",
]
