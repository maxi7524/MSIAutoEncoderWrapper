"""Dataset strategy registration package."""

from msi_autoencoder_wrapper.utils.module_search import discover_modules

discover_modules(__name__, recursive=False)
from .cohort_dataset import CohortDataset, CohortLatentDataset, CohortPixelDataset

__all__ = ["CohortDataset", "CohortLatentDataset", "CohortPixelDataset"]
