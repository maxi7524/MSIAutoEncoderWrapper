"""Public model-dataset contracts and strategies."""

from .base_dataset import MSIBaseDataset
from .annotations import AnnotationAwareDatasetMixin
from .dataset_manager import DatasetManager
from .subsetting import DatasetSubsetter
from .strategies.cohort_dataset import CohortLatentDataset, CohortPixelDataset
from .strategies.pixel_dataset import PixelDataset

__all__ = [
    "CohortLatentDataset",
    "CohortPixelDataset",
    "AnnotationAwareDatasetMixin",
    "DatasetManager",
    "DatasetSubsetter",
    "MSIBaseDataset",
    "PixelDataset",
]
