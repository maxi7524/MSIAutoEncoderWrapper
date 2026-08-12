"""Public dataset-management operations."""

from .download import materialize_selection
from .cohort_annotations import build_cohort_annotation_index
from .composition import compose_cohort
from .query import query_to_selection
from .split import grouped_dataset_split

__all__ = [
    "grouped_dataset_split",
    "materialize_selection",
    "build_cohort_annotation_index",
    "compose_cohort",
    "query_to_selection",
]
